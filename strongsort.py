import supervision as sv
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict, deque
import time
import os
from pathlib import Path
from boxmot.trackers.strongsort.strongsort import StrongSort
 
# --- 0. Configuration and Paths ---
SOURCE_VIDEO_PATH = "15sec_input_720p.mp4"
OUTPUT_VIDEO_PATH = "output/annotated_output_strongsort_reidentification.mp4" 
MODEL_PATH = "model/best.pt" 

BALL_CLASS_ID = 0 
YOLO_CONF_THRESHOLD = 0.85   

# --- 1. Load Models and Initialize StrongSORT ---
print(f"Loading YOLO model from: {MODEL_PATH}")
detection_model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully.")

    
   

# Initialize StrongSORT with optimized parameters for re-identification
print("Initializing StrongSORT tracker...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert string path to Path object for StrongSORT
weights_path = Path(os.path.abspath('./osnet_x0_25_msmt17.pt'))
print(f"Using weights from: {weights_path}")

tracker = StrongSort( 
    reid_weights=weights_path,
    device=device,
    half=torch.cuda.is_available(),
    max_cos_dist=0.25,
    max_iou_dist=0.6,
    max_age=200,
    n_init=3,
    nn_budget=200
)

print("StrongSORT tracker initialized.")

# Get video properties
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {SOURCE_VIDEO_PATH}") 
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

print(f"Video properties: {width}x{height} @ {fps} fps, Total frames: {total_frames}")

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
if not out.isOpened():
    print(f"Error: Could not create video writer for {OUTPUT_VIDEO_PATH}")
    exit()

# --- 2. Initialize Annotators ---
# Enhanced color palette for better visualization
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
          '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(colors),
    thickness=3
)

label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(colors),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_position=sv.Position.TOP_CENTER,
    text_thickness=2,
    text_scale=0.7
)

triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=30,
    height=25,
    outline_thickness=2
)

# --- 3. Re-identification Enhancement Variables ---
frame_count = 0
player_registry = {}  # Store player information for re-identification
player_frame_counts = defaultdict(int)
player_last_seen = {}
player_confidence_history = defaultdict(list)
reidentification_events = []  # Track re-identification events

# Enhanced filtering function for better tracking
def filter_detections_for_tracking(detections: sv.Detections, 
                                 min_area=800, max_area=50000,
                                 min_aspect_ratio=1.2, max_aspect_ratio=4.5,
                                 nms_threshold=0.20):
    """Enhanced filtering for StrongSORT with re-identification focus"""
    if len(detections) == 0:
        return detections
    
    # Calculate areas and aspect ratios
    areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
            (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    
    widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    aspect_ratios = heights / widths
    
    # Enhanced spatial filtering
    border_margin = 20  # Pixels from border
    x_min_valid = detections.xyxy[:, 0] > border_margin
    y_min_valid = detections.xyxy[:, 1] > border_margin
    x_max_valid = detections.xyxy[:, 2] < (width - border_margin)
    y_max_valid = detections.xyxy[:, 3] < (height - border_margin)
    
    # Combine all filters
    valid_mask = (
        (areas >= min_area) & (areas <= max_area) &
        (aspect_ratios >= min_aspect_ratio) & (aspect_ratios <= max_aspect_ratio) &
        x_min_valid & y_min_valid & x_max_valid & y_max_valid
    )
    
    filtered_detections = detections[valid_mask]
    
    # Apply NMS
    if len(filtered_detections) > 0:
        filtered_detections = filtered_detections.with_nms(
            threshold=nms_threshold, 
            class_agnostic=True
        )
    
    return filtered_detections

def create_empty_detections():
    """Create empty detections object"""
    return sv.Detections(
        xyxy=np.zeros((0, 4)),
        confidence=np.zeros(0),
        class_id=np.zeros(0, dtype=int),
        tracker_id=np.zeros(0, dtype=int)
    )

def update_player_registry(tracker_id, bbox, confidence, frame_num):
    """Update player registry for re-identification analysis"""
    if tracker_id not in player_registry:
        player_registry[tracker_id] = {
            'first_seen': frame_num,
            'last_seen': frame_num,
            'total_frames': 1,
            'avg_confidence': confidence,
            'bbox_history': [bbox],
            'reappearances': 0
        }
    else:
        # Check for re-appearance (gap in tracking)
        if frame_num - player_registry[tracker_id]['last_seen'] > fps // 2:  # Gap > 0.5 seconds
            player_registry[tracker_id]['reappearances'] += 1
            reidentification_events.append({
                'player_id': tracker_id,
                'frame': frame_num,
                'gap_frames': frame_num - player_registry[tracker_id]['last_seen']
            })
            print(f"🔄 Re-identification: Player #{tracker_id} reappeared at frame {frame_num} "
                  f"after {frame_num - player_registry[tracker_id]['last_seen']} frame gap")
        
        # Update registry
        player_registry[tracker_id]['last_seen'] = frame_num
        player_registry[tracker_id]['total_frames'] += 1
        player_registry[tracker_id]['bbox_history'].append(bbox)
        
        # Update average confidence
        old_avg = player_registry[tracker_id]['avg_confidence']
        total_frames = player_registry[tracker_id]['total_frames']
        player_registry[tracker_id]['avg_confidence'] = (
            (old_avg * (total_frames - 1) + confidence) / total_frames
        )

# --- 4. Main Processing Loop ---
print("Starting enhanced re-identification processing...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Object detection
    results = detection_model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Separate ball and player detections
    ball_detections = detections[detections.class_id == BALL_CLASS_ID]
    if len(ball_detections) > 0:
        ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=15)
    
    player_detections = detections[detections.class_id != BALL_CLASS_ID]
    
    # Enhanced filtering for better re-identification
    player_detections = filter_detections_for_tracking(
        player_detections,
        min_area=1000,
        max_area=6000,
        min_aspect_ratio=1.1,
        max_aspect_ratio=4.8,
        nms_threshold=0.20
    )
    
    # Update StrongSORT tracker
    if len(player_detections) > 0:
        # Convert to format expected by StrongSORT
        dets = np.zeros((len(player_detections), 6))
        dets[:, 0:4] = player_detections.xyxy
        dets[:, 4] = player_detections.confidence
        dets[:, 5] = player_detections.class_id
        
        # StrongSORT update - note that boxmot expects detections in xyxy format
        outputs = tracker.update(dets, frame)
        
        if len(outputs) > 0:
            # Extract tracking results
            tracked_boxes = outputs[:, :4]  # [x1, y1, x2, y2]
            tracked_ids = outputs[:, 4].astype(int)
            tracked_scores = outputs[:, 5] if outputs.shape[1] > 5 else np.ones(len(outputs)) * 0.5
            tracked_classes = outputs[:, 6].astype(int) if outputs.shape[1] > 6 else np.ones(len(outputs), dtype=int)
            
            # Create supervision detections
            tracked_players = sv.Detections(
                xyxy=tracked_boxes,
                confidence=tracked_scores,
                class_id=tracked_classes,
                tracker_id=tracked_ids
            )
            
            # Update player registry and trails
            for i, (bbox, tracker_id, confidence) in enumerate(
                zip(tracked_boxes, tracked_ids, tracked_scores)
            ):
                update_player_registry(tracker_id, bbox, confidence, frame_count)
                player_frame_counts[tracker_id] += 1
                
                player_last_seen[tracker_id] = frame_count
                player_confidence_history[tracker_id].append(confidence)
        else:
            tracked_players = create_empty_detections()
    else:
        tracked_players = create_empty_detections()
    
    # Create enhanced labels with re-identification info
    player_labels = []
    if len(tracked_players) > 0:
        for i, tracker_id in enumerate(tracked_players.tracker_id):
            confidence = tracked_players.confidence[i]
            registry_info = player_registry.get(tracker_id, {})
            
            # Enhanced label with re-identification info
            reapp_count = registry_info.get('reappearances', 0)
            total_frames = registry_info.get('total_frames', 0)
            avg_conf = registry_info.get('avg_confidence', confidence)
            
            # Create informative label
            label_parts = [f"P#{tracker_id}"]
            if reapp_count > 0:
                label_parts.append(f"🔄{reapp_count}")  # Re-identification indicator
            label_parts.append(f"({avg_conf:.2f})")
            label_parts.append(f"[{total_frames}f]")
            
            player_labels.append(" ".join(label_parts))
    
    # Enhanced frame annotation
    annotated_frame = frame.copy()
    
    # Draw player boxes and labels
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
    
    # Draw ball
    if len(ball_detections) > 0:
        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections
        )
    
    # Enhanced info display
    active_players = len(tracked_players)
    total_unique = len(player_registry)
    total_reidentifications = sum(info.get('reappearances', 0) for info in player_registry.values())
    stable_players = sum(1 for count in player_frame_counts.values() if count >= fps)
    
    info_lines = [
        f"Frame: {frame_count}/{total_frames}",
        f"Active: {active_players} | Unique: {total_unique}",
        f"Re-IDs: {total_reidentifications} | Stable: {stable_players}",
        f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}"
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(annotated_frame, line, (10, 30 + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Progress indicator
    progress = int((frame_count / total_frames) * 100)
    cv2.rectangle(annotated_frame, (10, height - 30), (10 + progress * 3, height - 10), 
                 (0, 255, 0), -1)
    cv2.putText(annotated_frame, f"{progress}%", (10 + progress * 3 + 10, height - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Write frame
    out.write(annotated_frame)
    
    # Progress reporting
    if frame_count % (fps * 2) == 0:  # Every 2 seconds
        print(f"📊 Frame {frame_count}/{total_frames} | "
              f"Active: {active_players} | Unique: {total_unique} | "
              f"Re-IDs: {total_reidentifications}")

# --- 5. Cleanup and Final Analysis ---
cap.release()
out.release()

print(f"\n🎉 Processing Complete!")
print(f"📹 Output saved: {OUTPUT_VIDEO_PATH}")
print(f"🎯 Total unique players: {len(player_registry)}")
print(f"🔄 Total re-identifications: {sum(info.get('reappearances', 0) for info in player_registry.values())}")

# Detailed re-identification analysis
print(f"\n📋 RE-IDENTIFICATION ANALYSIS")
print("=" * 50)

for player_id, info in sorted(player_registry.items()):
    duration = (info['last_seen'] - info['first_seen'] + 1) / fps
    reapp_count = info.get('reappearances', 0)
    avg_conf = info.get('avg_confidence', 0)
    
    status = "🟢 Stable" if info['total_frames'] >= fps else "🟡 Brief"
    if reapp_count > 0:
        status += f" | 🔄 {reapp_count} re-IDs"
    
    print(f"Player #{player_id:2d}: {status}")
    print(f"  Duration: {duration:.1f}s | Frames: {info['total_frames']} | Avg Conf: {avg_conf:.3f}")
    
    if reapp_count > 0:
        print(f"  ✅ Successfully re-identified {reapp_count} times")

# Re-identification events summary
if reidentification_events:
    print(f"\n🔄 RE-IDENTIFICATION EVENTS ({len(reidentification_events)} total)")
    print("-" * 50)
    
    # Show last 10 events or all if fewer than 10
    events_to_show = reidentification_events[-10:] if len(reidentification_events) > 10 else reidentification_events
    
    for event in events_to_show:
        gap_seconds = event['gap_frames'] / fps
        print(f"  Frame {event['frame']:4d}: Player #{event['player_id']} "
              f"(gap: {gap_seconds:.1f}s)")

# Performance metrics
total_possible_reids = len([e for e in reidentification_events if e['gap_frames'] > fps//2])
successful_reids = len(reidentification_events)
reid_success_rate = (successful_reids / max(1, total_possible_reids)) * 100

print(f"\n📈 PERFORMANCE METRICS")
print(f"Re-identification Success Rate: {reid_success_rate:.1f}%")
print(f"Average Track Duration: {np.mean([info['total_frames'] for info in player_registry.values()]) / fps:.1f}s")
print(f"Tracking Efficiency: {(len([p for p in player_registry.values() if p['total_frames'] >= fps]) / len(player_registry)) * 100:.1f}%")

print(f"\n✨ StrongSORT Re-identification Complete!")