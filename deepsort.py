
import supervision as sv
import cv2
import numpy as np
from ultralytics import YOLO
import torch 

# --- 0. Configuration and Paths ---
# Define paths to your video and model
SOURCE_VIDEO_PATH = "15sec_input_720p.mp4"
OUTPUT_VIDEO_PATH = "output/annotated_output_deepsort_optimized.mp4"  
MODEL_PATH = "model/best.pt" 

BALL_CLASS_ID = 0 


# Confidence threshold for your YOLO model
YOLO_CONF_THRESHOLD = 0.87   

# --- 1. Load Models and Initialize Components ---


print(f"Loading YOLO model from: {MODEL_PATH}")
detection_model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully.")

# Initialize the DeepSORT tracker using deep_sort_realtime

from deep_sort_realtime.deepsort_tracker import DeepSort

print("Initializing DeepSORT tracker...")
tracker = DeepSort(
    max_age=250,                    # INCREASED: Keep tracks alive longer during occlusions
    n_init=4,                       # INCREASED: Require 3 consecutive detections (more stable)
    max_iou_distance=0.5,          # REDUCED: Stricter spatial matching
    max_cosine_distance=0.25,       # REDUCED: Much stricter appearance matching
    nn_budget=300,                  # INCREASED: More appearance samples per player
    embedder="mobilenet",           
    half=True,                      
    bgr=True,                       
    embedder_gpu=True               
)
print("DeepSORT tracker initialized.")

# Get video properties for output video writer
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {SOURCE_VIDEO_PATH}") 
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

print(f"Video properties: {width}x{height} @ {fps} fps, Total frames: {total_frames}")

# Define video writer for the output annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
if not out.isOpened():
    print(f"Error: Could not create video writer for {OUTPUT_VIDEO_PATH}")
    exit()


# --- 2. Initialize Annotators for Visualization ---

box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700', '#32CD32', '#FF4500']),
    thickness=2
)

label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700', '#32CD32', '#FF4500']),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_position=sv.Position.TOP_CENTER,
    text_thickness=2,
    text_scale=0.6
)

triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'), # Gold for the ball
    base=25,
    height=20,
    outline_thickness=2
)

# --- 3. Tracking Loop Initialization ---
frame_count = 0
total_unique_players_tracked = set() # To store all unique player IDs encountered
player_frame_counts = {} # To track how many frames each player ID appears

print("Starting video processing loop...")

# Create empty Detections for fallback cases
def create_empty_detections():
    return sv.Detections(
        xyxy=np.zeros((0, 4)),
        confidence=np.zeros(0),
        class_id=np.zeros(0, dtype=int),
        tracker_id=np.zeros(0, dtype=int)
    )


# This function helps in cleaning up raw YOLO detections before sending to the tracker.
def filter_detections_for_tracking(detections: sv.Detections, min_area=1000, max_area=60000, 
                                   min_aspect_ratio=1.2, max_aspect_ratio=4.0, 
                                   nms_threshold=0.35):
    """
    Filters raw detections based on area, aspect ratio, and applies NMS.
    This helps the tracker by providing cleaner inputs.
    Parameters adjusted to better handle soccer players with varying poses.
    """
    if len(detections) == 0:
        return detections
    
    # Calculate areas
    areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
            (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    
    # Filter by area
    area_mask = (areas >= min_area) & (areas <= max_area)
    
    # Calculate aspect ratios (height / width)
    widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    aspect_ratios = heights / widths
    
    # Filter by aspect ratio (typical for standing humans)
    aspect_mask = (aspect_ratios >= min_aspect_ratio) & (aspect_ratios <= max_aspect_ratio)
    
    # Combine spatial filters
    valid_spatial_mask = area_mask & aspect_mask
    filtered_detections = detections[valid_spatial_mask]
    
    # Apply Non-Max Suppression to reduce overlapping detections for the same object
    # This is applied *before* tracking to ensure the tracker gets one detection per object
    if len(filtered_detections) > 0:
        filtered_detections = filtered_detections.with_nms(threshold=nms_threshold, class_agnostic=True)
    
    return filtered_detections


# --- 5. Main Video Processing Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # End of video stream

    frame_count += 1

    # Step 5.1: Perform object detection with your YOLO model
    # The 'conf' parameter filters detections by confidence score
    results = detection_model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)[0] 
    detections = sv.Detections.from_ultralytics(results)

    # Step 5.2: Separate detections into players/referees and the bal
    ball_detections = detections[detections.class_id == BALL_CLASS_ID]
    
    # Pad ball bounding boxes slightly for better visualization, if needed
    if len(ball_detections) > 0:
        ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10) # Smaller pad for ball

    player_detections = detections[detections.class_id != BALL_CLASS_ID]

    # Step 5.3: Apply filtering to player detections for better tracking input
    # This removes unlikely detections (too small, too large, wrong aspect ratio)
    # and handles overlapping initial detections.
    player_detections = filter_detections_for_tracking(
        player_detections, 
        min_area=600, 
        max_area=60000,
        min_aspect_ratio=1.3, 
        max_aspect_ratio=4.0,
        nms_threshold=0.30 
    )

    # Step 5.4: Update DeepSORT tracker with current frame's player detections
    # Convert detections to DeepSORT format: (bbox, confidence, class)
    # where bbox is in the format [x, y, w, h] 
    det_list = []
    if len(player_detections) > 0:
        for i in range(len(player_detections.xyxy)):
            bbox = player_detections.xyxy[i]
            x1, y1, x2, y2 = bbox
            w, h = x2-x1, y2-y1
            confidence = player_detections.confidence[i]
            class_id = player_detections.class_id[i]
            # Only include detections with reasonable width and height (avoid partial players at frame edges)
            if w > 20 and h > 50:  
                det_list.append(([x1, y1, w, h], confidence, class_id))
    
    # DeepSORT update returns list of Track objects
    tracks = tracker.update_tracks(det_list, frame=frame)
    
    # Convert tracks to supervision format for consistency with the rest of the code
    track_boxes = []
    track_ids = []
    track_confidences = []
    track_class_ids = []
    
    for track in tracks:
        # Only include confirmed tracks with higher confidence in appearance embedding
        if not track.is_confirmed() or track.time_since_update > 10:
            continue
            
        ltrb = track.to_ltrb()  # Get the box in (left, top, right, bottom) format
        x1, y1, x2, y2 = ltrb
        
        # Skip tiny bounding boxes that are likely false positives
        w, h = x2-x1, y2-y1
        if w < 15 or h < 40:
            continue
            
        track_boxes.append([x1, y1, x2, y2])
        track_ids.append(track.track_id)
        
        # Safely get confidence from track or use default
        confidence = getattr(track, 'det_conf', None)
        # If confidence is None or not a number, use default
        if confidence is None or not isinstance(confidence, (int, float)):
            confidence = 0.5
        track_confidences.append(confidence)
        track_class_ids.append(getattr(track, 'det_class', 1))  # Default to class 1 if not available
    
    # Create supervision Detections from the tracks
    if track_boxes:
        tracked_players = sv.Detections(
            xyxy=np.array(track_boxes),
            confidence=np.array(track_confidences),
            class_id=np.array(track_class_ids),
            tracker_id=np.array(track_ids)
        )
    else:
        # Empty detections
        tracked_players = create_empty_detections()

    # Step 5.5: Update player statistics for analysis
    if len(tracked_players) > 0:
        for i, tracker_id in enumerate(tracked_players.tracker_id):
            total_unique_players_tracked.add(tracker_id) # Record all unique IDs
            
            # Update frame count for this player
            if tracker_id not in player_frame_counts:
                player_frame_counts[tracker_id] = 0
            player_frame_counts[tracker_id] += 1

    # Step 5.6: Prepare labels for annotation
    player_labels = []
    if len(tracked_players) > 0:
        for i, tracker_id in enumerate(tracked_players.tracker_id):
            confidence = tracked_players.confidence[i]
            # Get the stability (how many frames this ID has been tracked)
            stability = player_frame_counts.get(tracker_id, 0)
            # Handle None confidence values
            if confidence is None:
                confidence_str = "N/A"
            else:
                confidence_str = f"{confidence:.2f}"
            player_labels.append(f"P#{tracker_id} ({confidence_str}) [{stability}f]")

    # Step 5.7: Print debug information (every few frames or at start/end)
    if frame_count % fps == 0 or frame_count <= 5 or frame_count == total_frames:
        active_players = len(tracked_players)
        stable_players_count = sum(1 for pid, count in player_frame_counts.items() if count >= fps) # Players tracked for at least 1 second
        
        current_ids = sorted(list(tracked_players.tracker_id)) if active_players > 0 else 'None'
        print(f"Frame {frame_count:4d}/{total_frames}: Active Players: {active_players:2d} | "
              f"Current IDs: {current_ids}")
        if active_players > 0:
            # Handle potential None values in confidence
            confidence_strs = []
            for conf in tracked_players.confidence:
                if conf is None:
                    confidence_strs.append('N/A')
                else:
                    confidence_strs.append(f'{conf:.2f}')
            print(f"  Confidences: {confidence_strs}")
        print(f"  Total Unique IDs seen: {len(total_unique_players_tracked)} | Stable (>={fps} frames): {stable_players_count}")
        print("-" * 30)

    # Step 5.8: Annotate the frame with bounding boxes and labels
    annotated_frame = frame.copy()

    # Annotate players (with boxes and labels)
    if len(tracked_players) > 0:
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_players
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_players,
            labels=player_labels
        )
    
    # Annotate the ball (with a triangle)
    if len(ball_detections) > 0:
        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections
        )

    # Add general info text on the frame
    info_text_lines = [
        f"Frame: {frame_count}/{total_frames}",
        f"Active Players: {len(tracked_players)}",
        f"Unique IDs: {len(total_unique_players_tracked)}",
        f"Stable (1s+): {sum(1 for count in player_frame_counts.values() if count >= fps)}"
    ]
    for i, text_line in enumerate(info_text_lines):
        cv2.putText(annotated_frame, text_line, (10, 30 + i * 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Step 5.9: Write the annotated frame to the output video
    out.write(annotated_frame)






# --- 6. Release Resources and Final Analysis ---
cap.release()
out.release()


print(f"\n--- Video Processing Complete ---")
print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
print(f"Total unique players identified: {len(total_unique_players_tracked)}")
print(f"All Player IDs encountered: {sorted(list(total_unique_players_tracked))}")

# Final stability analysis based on how long each ID was tracked
# Define 'stable' as being tracked for at least 1 second (25 frames)
STABILITY_THRESHOLD_FRAMES = fps 
stable_players = {pid: count for pid, count in player_frame_counts.items() if count >= STABILITY_THRESHOLD_FRAMES}
transient_players = {pid: count for pid, count in player_frame_counts.items() if count < STABILITY_THRESHOLD_FRAMES}

print(f"\n--- STABILITY ANALYSIS ---")
print(f"Number of stable players ({STABILITY_THRESHOLD_FRAMES}+ frames): {len(stable_players)}")
print(f"Number of transient players (<{STABILITY_THRESHOLD_FRAMES} frames): {len(transient_players)}")

if len(total_unique_players_tracked) > 0:
    tracking_efficiency = (len(stable_players) / len(total_unique_players_tracked)) * 100
    print(f"Tracking efficiency (Stable / Total Unique): {tracking_efficiency:.1f}%")
else:
    print("No players were tracked.")

if stable_players:
    print(f"\n--- Top 10 Most Stable Players (by frames tracked) ---")
    sorted_stable = sorted(stable_players.items(), key=lambda item: item[1], reverse=True)
    for pid, count in sorted_stable[:10]:
        print(f"  Player #{pid}: {count} frames ({count/total_frames*100:.1f}% of video)")

if transient_players:
    print(f"\n--- Top 10 Most Transient Players (likely false positives or brief appearances) ---")
    sorted_transient = sorted(transient_players.items(), key=lambda item: item[1], reverse=True)
    for pid, count in sorted_transient[:10]:
        print(f"  Player #{pid}: {count} frames")

# --- 7. Recommendations for parameter tuning (based on typical DeepSORT issues) ---
print(f"\n--- PARAMETER OPTIMIZATION SUGGESTIONS ---")
if len(total_unique_players_tracked) > 25: # Example threshold for too many IDs in soccer
    print("  • If too many unique IDs (over-fragmentation), consider: ")
    print("    - Increasing `DEEPSORT_MAX_AGE` further (e.g., 150-200) to allow tracks to persist longer.")
    print("    - Decreasing `max_cosine_distance` (e.g., 0.1) for stricter appearance matching.")
    print("    - Increasing your nn_budget value for more appearance samples per ID.")
    print("    - Adjusting the confirmation criteria in the track extraction loop.")
    print("    - Trying a more powerful feature extractor in DeepSORT (e.g., 'resnet' instead of 'mobilenet').")
elif len(stable_players) < 20: # Example threshold for too few stable IDs in soccer
    print("  • If too few stable players (missing tracks), consider: ")
    print("    - Decreasing `YOLO_CONF_THRESHOLD` further (e.g., 0.75) to detect more players.")
    print("    - Decreasing `DEEPSORT_MIN_HITS` to 1 to allow tracks to be established more easily.")
    print("    - Increasing `max_cosine_distance` (e.g., 0.2-0.3) for more lenient appearance matching.")
    print("    - Adjusting the filter_detections_for_tracking parameters to include more detections.")

print("\nTracking optimization complete. Review the output video and logs for performance assessment.")


