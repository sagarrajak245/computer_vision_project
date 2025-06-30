import supervision as sv
import cv2
import numpy as np
import os

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Paths
SOURCE_VIDEO_PATH = "15sec_input_720p.mp4"  
OUTPUT_VIDEO_PATH = "output/annotated_output_bytetrack.mp4"  
BALL_ID = 0

from ultralytics import YOLO

PLAYER_DETECTION_MODEL = YOLO('model/best.pt')

# Enhanced annotators with better visibility
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700', '#32CD32', '#FF4500']),
    thickness=3
)

label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700', '#32CD32', '#FF4500']),
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

# BALANCED TRACKER PARAMETERS - Less strict but still stable
tracker = sv.ByteTrack(
    track_activation_threshold=0.82,     # Slightly lower than model confidence
    lost_track_buffer=50,                # 2 seconds buffer (25fps * 2)
    minimum_matching_threshold=0.75,     # Balanced matching threshold
    minimum_consecutive_frames=2,        # Require only 2 consecutive frames
    frame_rate=25
)
tracker.reset()

# Get video properties
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video properties: {width}x{height} @ {fps} fps")

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

frame_count = 0
total_unique_players = set()
player_stats = {}  # Track how long each player ID appears

def filter_detections_by_stability(detections, min_area=600, max_area=60000):
    """Balanced filtering - less aggressive"""
    if len(detections) == 0:
        return detections
    
    # Calculate areas
    areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
            (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    
    # More lenient area filtering
    area_mask = (areas >= min_area) & (areas <= max_area)
    
    # More lenient aspect ratio filtering
    widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    aspect_ratios = heights / widths
    aspect_mask = (aspect_ratios >= 1.0) & (aspect_ratios <= 5.0)  # More lenient
    
    # Combine filters
    valid_mask = area_mask & aspect_mask
    
    return detections[valid_mask]

for frame in frame_generator:
    frame_count += 1
    
    # Step 1: Inference with balanced confidence
    results = PLAYER_DETECTION_MODEL(frame, conf=0.78)  # Lower to catch more players
    detections = sv.Detections.from_ultralytics(results[0])
    
    # Step 2: Separate ball detections
    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=15)
    
    # Step 3: Enhanced player detection filtering
    player_detections = detections[detections.class_id != BALL_ID]
    
    # Apply stability-based filtering
    player_detections = filter_detections_by_stability(player_detections)
    
    # More balanced NMS to reduce duplicates but keep players
    player_detections = player_detections.with_nms(threshold=0.38, class_agnostic=True)
    
    # Sort by confidence to prioritize high-confidence detections
    if len(player_detections) > 0:
        sorted_indices = np.argsort(player_detections.confidence)[::-1]
        player_detections = player_detections[sorted_indices]
        
        # Allow more players per frame but still limit
        if len(player_detections) > 25:  # Slightly higher limit
            player_detections = player_detections[:25]
    
    player_detections.class_id -= 1
    
    # Step 4: Enhanced tracking
    tracked_players = tracker.update_with_detections(player_detections)
    
    # Step 5: Update player statistics
    if len(tracked_players) > 0:
        for tracker_id in tracked_players.tracker_id:
            if tracker_id not in player_stats:
                player_stats[tracker_id] = 0
            player_stats[tracker_id] += 1
            total_unique_players.add(tracker_id)
    
    # Step 6: Create labels with stability info
    labels = []
    if len(tracked_players) > 0:
        for i, tracker_id in enumerate(tracked_players.tracker_id):
            confidence = tracked_players.confidence[i]
            stability = player_stats[tracker_id]
            labels.append(f"P#{tracker_id} ({confidence:.2f}) [{stability}f]")
    
    # Enhanced debug info with stability metrics
    if frame_count % 25 == 0 or frame_count <= 5:
        active_players = len(tracked_players)
        stable_players = sum(1 for pid, count in player_stats.items() if count >= 8)  # Lower threshold
        
        print(f"Frame {frame_count:3d}: {active_players:2d} active | "
              f"IDs: {sorted(list(tracked_players.tracker_id)) if active_players > 0 else 'None'}")
        print(f"            Total unique: {len(total_unique_players)} | "
              f"Stable (10+ frames): {stable_players}")
        if active_players > 0:
            print(f"            Confidences: {[f'{c:.2f}' for c in tracked_players.confidence]}")
        print("---")
    
    # Step 7: Enhanced annotation
    annotated_frame = frame.copy()
    
    # Annotate players
    if len(tracked_players) > 0:
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_players
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_players,
            labels=labels
        )
    
    # Annotate ball
    if len(ball_detections) > 0:
        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections
        )
    
    # Add comprehensive frame info
    info_text = [
        f"Frame: {frame_count}/375",
        f"Active: {len(tracked_players)}",
        f"Total IDs: {len(total_unique_players)}",
        f"Stable: {sum(1 for count in player_stats.values() if count >= 8)}"  # Lower threshold
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(annotated_frame, text, (10, 30 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Step 8: Write frame
    out.write(annotated_frame)

cap.release()
out.release()

# Final analysis with adjusted stability metrics
stable_players = {pid: count for pid, count in player_stats.items() if count >= 8}  # Lower threshold
transient_players = {pid: count for pid, count in player_stats.items() if count < 8}

print(f"\n✅ Video processing complete!")
print(f"📹 Output saved to: {OUTPUT_VIDEO_PATH}")
print(f"👥 Total unique players tracked: {len(total_unique_players)}")
print(f"🎯 All Player IDs: {sorted(list(total_unique_players))}")
print(f"\n📊 STABILITY ANALYSIS:")
print(f"   ✅ Stable players (8+ frames): {len(stable_players)}")
print(f"   ⚠️  Transient players (<8 frames): {len(transient_players)}")
print(f"   🎯 Tracking efficiency: {len(stable_players)/len(total_unique_players)*100:.1f}%")

if stable_players:
    print(f"\n🏆 MOST STABLE PLAYERS:")
    sorted_stable = sorted(stable_players.items(), key=lambda x: x[1], reverse=True)
    for pid, count in sorted_stable[:10]:  # Top 10
        print(f"   Player #{pid}: {count} frames ({count/375*100:.1f}% of video)")

if transient_players:
    print(f"\n⚠️  TRANSIENT PLAYERS (likely false positives):")
    sorted_transient = sorted(transient_players.items(), key=lambda x: x[1], reverse=True)
    for pid, count in sorted_transient[:15]:  # Show problematic ones
        print(f"   Player #{pid}: {count} frames")

# Balanced recommendations based on results
print(f"\n🔧 OPTIMIZATION SUGGESTIONS:")
if len(total_unique_players) > 30:
    print("   • Too many IDs - increase track_activation_threshold to 0.84")
elif len(total_unique_players) < 15:
    print("   • Too few IDs - decrease track_activation_threshold to 0.80")
if len(stable_players) < 12:
    print("   • Low stable player count - decrease conf threshold to 0.75")
elif len(stable_players) > 22:
    print("   • Too many stable players - increase conf threshold to 0.82")
if len(transient_players) > len(stable_players) * 1.5:
    print("   • Too many transient IDs - increase minimum_consecutive_frames to 3")

