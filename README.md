# Player Tracking and Re-identification System

## Objective
Given a 15-second video (`15sec_input_720p.mp4`), identify each player and ensure that players who go out of frame and reappear are assigned the same identity as before. The solution simulates real-time re-identification and player tracking using three different approaches.

## Problem Statement
- **Challenge**: Maintain consistent player IDs throughout the video
- **Key Requirement**: Players who exit and re-enter the frame must retain their original IDs
- **Real-world Application**: Sports analytics, player performance tracking, tactical analysis

---

## 📦 Model Weights

This project uses custom-trained weights for best performance. Please download them from the links below:
# create model folder and then inside download these models-

### 🎯 1️⃣ YOLO Player/Ball Detection Model

* **Description**: Fine-tuned YOLO model (`best.pt`) for detecting tennis players and the ball.
* **Download Link**: [best.pt on Google Drive](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

---

### 🧭 2️⃣ Player Re-Identification Model

* **Description**: OSNet-based appearance model for player ReID (used in StrongSORT / DeepSORT tracking).
* **Download Link**: [OSNet Weights on Google Drive](https://drive.google.com/file/d/1QjEu6KJNcPLLij2nKMlJ6Xlke0V9IkdI/view?usp=sharing)

---

> ⚠️ Note: These files are large and hosted externally due to GitHub's file size limitations. Please download them manually and place them in the appropriate directory specified in the code.

---

## Method 1: ByteTrack IOU Method

### Overview
ByteTrack is a simple yet effective multi-object tracking algorithm that relies on Intersection over Union (IoU) matching and Kalman filtering for object tracking. It's particularly effective for tracking objects in crowded scenes without requiring appearance features.

### How ByteTrack Solves the Problem

**Core Principle:**
ByteTrack addresses the re-identification challenge through a two-stage association strategy:
1. **High-confidence Detection Matching**: Primary association using high-confidence detections
2. **Low-confidence Recovery**: Secondary association using low-confidence detections to recover lost tracks

**Key Algorithm Components:**

#### 1. Track State Management
```python
# Three track states for robust tracking
tracker = sv.ByteTrack(
    track_activation_threshold=0.82,     # Threshold for track activation
    lost_track_buffer=50,                # Buffer frames to keep lost tracks
    minimum_matching_threshold=0.75,     # IoU threshold for matching
    minimum_consecutive_frames=2,        # Frames needed for track confirmation
    frame_rate=25
)
```

#### 2. Detection Preprocessing
```python
def filter_detections_by_stability(detections, min_area=600, max_area=60000):
    """Enhanced filtering for stable player detection"""
    # Area-based filtering
    areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
            (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    area_mask = (areas >= min_area) & (areas <= max_area)
    
    # Aspect ratio filtering for human-like shapes
    widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    aspect_ratios = heights / widths
    aspect_mask = (aspect_ratios >= 1.0) & (aspect_ratios <= 5.0)
    
    return detections[area_mask & aspect_mask]
```

#### 3. Core Tracking Process
```python
# Main tracking pipeline
for frame in frame_generator:
    # Step 1: Object Detection with confidence threshold
    results = PLAYER_DETECTION_MODEL(frame, conf=0.78)
    detections = sv.Detections.from_ultralytics(results[0])
    
    # Step 2: Filter and process detections
    player_detections = detections[detections.class_id != BALL_ID]
    player_detections = filter_detections_by_stability(player_detections)
    player_detections = player_detections.with_nms(threshold=0.38)
    
    # Step 3: Update tracker with processed detections
    tracked_players = tracker.update_with_detections(player_detections)
    
    # Step 4: Maintain player statistics for analysis
    for tracker_id in tracked_players.tracker_id:
        player_stats[tracker_id] = player_stats.get(tracker_id, 0) + 1
```

### Key Features Solving Re-identification

#### 1. **Lost Track Buffer System**
- Maintains tracks for 50 frames (2 seconds) after losing detection
- Allows players to be re-associated when they reappear
- Prevents ID switching for temporary occlusions

#### 2. **Cascade Matching Strategy**
- Prioritizes recently active tracks for matching
- Uses IoU-based distance metric for spatial consistency
- Implements hierarchical matching to handle crowded scenes

#### 3. **Kalman Filter Prediction**
- Predicts player positions during brief disappearances
- Maintains motion continuity for smooth tracking
- Helps in associating detections with predicted positions

### Performance Analysis

#### Results Summary
```
📊 STABILITY ANALYSIS:
✅ Stable players (8+ frames): 31
⚠️ Transient players (<8 frames): 4
🎯 Tracking efficiency: 88.6%

🏆 MOST STABLE PLAYERS:
Player #10: 300 frames (80.0% of video)
Player #15: 297 frames (79.2% of video)
Player #3: 211 frames (56.3% of video)
Player #6: 203 frames (54.1% of video)
Player #24: 203 frames (54.1% of video)
Player #9: 189 frames (50.4% of video)
Player #16: 145 frames (38.7% of video)
Player #13: 142 frames (37.9% of video)
Player #21: 123 frames (32.8% of video)
Player #22: 121 frames (32.3% of video)

⚠️ TRANSIENT PLAYERS (likely false positives):
Player #14: 4 frames
Player #8: 3 frames
Player #35: 3 frames
Player #18: 1 frames
```

#### Performance Metrics
- **Total Unique Players**: 35 players tracked
- **Stable Tracking Rate**: 88.6% (31/35 players stable)
- **Frame Coverage**: Top players tracked for 50-80% of video duration
- **False Positive Rate**: 11.4% (4 transient detections)

#### Strengths
- ✅ High stability for main players (80% video coverage for top players)
- ✅ Effective handling of player re-entry scenarios
- ✅ Minimal ID switching due to robust IoU matching
- ✅ Real-time performance suitable for live applications

#### Limitations
- ⚠️ Some false positive detections (transient players)
- ⚠️ Relies purely on spatial information (no appearance features)
- ⚠️ May struggle with identical player appearances in close proximity

### Optimization Recommendations
Based on the analysis results:

```
🔧 OPTIMIZATION SUGGESTIONS:
• Too many IDs - increase track_activation_threshold to 0.84
• Too many stable players - increase conf threshold to 0.82
```

**Parameter Tuning Guidelines:**
- **Increase `track_activation_threshold`** to reduce false positives
- **Increase `conf` threshold** for more selective detection
- **Adjust `lost_track_buffer`** based on typical occlusion duration
- **Fine-tune `minimum_matching_threshold`** for optimal IoU matching

---

## Method 2: YOLO + DeepSORT

### Overview
DeepSORT enhances the original SORT algorithm by incorporating deep appearance features for more robust re-identification. This method combines YOLO detection with CNN-based appearance modeling to maintain consistent player identities even during extended occlusions.

### How DeepSORT Solves the Problem

**Core Innovation:**
DeepSORT addresses the re-identification challenge through appearance-based matching combined with motion prediction:

1. **Deep Appearance Features**: Uses CNN embeddings to capture player appearance
2. **Cascade Matching**: Hierarchical association prioritizing recently active tracks
3. **Extended Track Management**: Longer track persistence during occlusions
4. **Appearance-Motion Fusion**: Combines spatial IoU and appearance similarity

### Key Algorithm Components

#### 1. Enhanced Tracker Configuration
```python
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(
    max_age=250,                    # Extended track persistence (10 seconds)
    n_init=4,                       # Require 4 consecutive detections for stability
    max_iou_distance=0.5,          # Spatial matching threshold
    max_cosine_distance=0.25,       # Strict appearance matching threshold
    nn_budget=300,                  # Appearance samples per track
    embedder="mobilenet",           # CNN feature extractor
    half=True,                      # FP16 optimization
    bgr=True,                       # Color format
    embedder_gpu=True               # GPU acceleration
)
```

#### 2. Advanced Detection Preprocessing
```python
def filter_detections_for_tracking(detections, min_area=1000, max_area=60000, 
                                   min_aspect_ratio=1.2, max_aspect_ratio=4.0, 
                                   nms_threshold=0.35):
    """Enhanced filtering for DeepSORT input optimization"""
    # Area-based filtering for realistic player sizes
    areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
            (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    area_mask = (areas >= min_area) & (areas <= max_area)
    
    # Human-like aspect ratio filtering
    widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    aspect_ratios = heights / widths
    aspect_mask = (aspect_ratios >= min_aspect_ratio) & (aspect_ratios <= max_aspect_ratio)
    
    # Apply spatial filters and NMS
    filtered_detections = detections[area_mask & aspect_mask]
    return filtered_detections.with_nms(threshold=nms_threshold, class_agnostic=True)
```

#### 3. Core Tracking Pipeline
```python
# Main tracking loop with appearance-based re-identification
for frame_count, frame in enumerate(video_frames):
    # Step 1: YOLO Detection with high confidence
    results = detection_model(frame, conf=0.87, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Step 2: Filter player detections
    player_detections = detections[detections.class_id != BALL_CLASS_ID]
    player_detections = filter_detections_for_tracking(player_detections)
    
    # Step 3: Convert to DeepSORT format
    det_list = []
    for i in range(len(player_detections.xyxy)):
        bbox = player_detections.xyxy[i]
        x1, y1, x2, y2 = bbox
        w, h = x2-x1, y2-y1
        if w > 20 and h > 50:  # Minimum size validation
            det_list.append(([x1, y1, w, h], player_detections.confidence[i], 
                           player_detections.class_id[i]))
    
    # Step 4: Update DeepSORT tracker with appearance features
    tracks = tracker.update_tracks(det_list, frame=frame)
    
    # Step 5: Extract confirmed tracks with quality control
    track_boxes, track_ids = [], []
    for track in tracks:
        if track.is_confirmed() and track.time_since_update <= 10:
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = ltrb
            if (x2-x1) >= 15 and (y2-y1) >= 40:  # Size validation
                track_boxes.append([x1, y1, x2, y2])
                track_ids.append(track.track_id)
```

### Key Features Solving Re-identification

#### 1. **Deep Appearance Modeling**
- **MobileNet CNN**: Extracts robust appearance features for each detection
- **Feature Embedding**: Creates 128-dimensional appearance vectors
- **Cosine Distance Matching**: Compares appearance similarity between tracks and detections
- **Appearance Budget**: Maintains multiple appearance samples per track for robustness

#### 2. **Extended Track Persistence**
- **Long-term Memory**: Keeps tracks alive for 250 frames (10 seconds)
- **Confirmation Strategy**: Requires 4 consecutive detections for track initialization
- **State Management**: Tracks transition through tentative → confirmed → deleted states

#### 3. **Cascade Matching Strategy**
```python
# Hierarchical matching prioritizes recently active tracks
# 1. Recent tracks (high priority) - appearance + motion
# 2. Older tracks (medium priority) - primarily appearance
# 3. New track initialization (low priority)
```

#### 4. **Quality Control Mechanisms**
- High confidence threshold (0.87) for initial detection
- Strict appearance matching (cosine distance < 0.25)
- Size validation to filter false positives
- Track confirmation requirements

### Performance Analysis

#### Results Summary
```
📊 DEEPSORT STABILITY ANALYSIS:
✅ Stable players (25+ frames): 27
⚠️ Transient players (<25 frames): 0
🎯 Tracking efficiency: 100.0%

🏆 MOST STABLE PLAYERS:
Player #1: 329 frames (87.7% of video)
Player #90: 321 frames (85.6% of video)
Player #75: 291 frames (77.6% of video)
Player #82: 268 frames (71.5% of video)
Player #11: 251 frames (66.9% of video)
Player #9: 230 frames (61.3% of video)
Player #34: 213 frames (56.8% of video)
Player #100: 209 frames (55.7% of video)
Player #95: 204 frames (54.4% of video)
Player #86: 186 frames (49.6% of video)

✅ NO TRANSIENT PLAYERS: Perfect stability achieved
```

#### Performance Metrics
- **Total Unique Players**: 27 players tracked
- **Perfect Stability**: 100% tracking efficiency (27/27 players stable)
- **Frame Coverage**: Top players tracked for 50-87% of video duration
- **Zero False Positives**: No transient detections
- **Excellent Persistence**: Players tracked consistently throughout video

#### Comparison with ByteTrack
| Metric | ByteTrack | DeepSORT | Improvement |
|---------|-----------|----------|-------------|
| Total Players | 35 | 27 | -23% (fewer false IDs) |
| Stable Players | 31 | 27 | Better quality |
| Tracking Efficiency | 88.6% | 100.0% | +11.4% |
| Transient Players | 4 | 0 | Perfect elimination |
| Top Player Coverage | 80.0% | 87.7% | +7.7% |

#### Strengths
- ✅ **Perfect Stability**: 100% of tracked players are stable (no false positives)
- ✅ **Extended Coverage**: Top players tracked for up to 87.7% of video
- ✅ **Robust Re-identification**: Appearance features handle complex occlusions
- ✅ **Quality Control**: Strict confirmation criteria eliminate noise
- ✅ **Consistent Identity**: No ID switching during re-entry scenarios

#### Limitations
- ⚠️ **Computational Cost**: Higher processing due to CNN feature extraction
- ⚠️ **Parameter Sensitivity**: Requires careful tuning of appearance thresholds
- ⚠️ **Initial Detection Dependency**: Relies on high-quality YOLO detections
- ⚠️ **Memory Usage**: Stores appearance features for multiple samples per track

### Optimization Achieved

**Parameter Tuning Results:**
The implementation demonstrates excellent parameter optimization:

```python
# Optimized Configuration
YOLO_CONF_THRESHOLD = 0.87        # High quality detections
max_age = 250                     # Extended track persistence
n_init = 4                        # Stable track confirmation
max_cosine_distance = 0.25        # Strict appearance matching
nn_budget = 300                   # Rich appearance memory
```

**Key Optimization Strategies:**
1. **High Detection Threshold**: 0.87 confidence eliminates weak detections
2. **Extended Track Persistence**: 250 frames (10 seconds) handles long occlusions
3. **Strict Appearance Matching**: 0.25 cosine distance prevents ID confusion
4. **Quality Gate**: Multi-frame confirmation prevents noise tracks

### DeepSORT vs ByteTrack Trade-offs

**DeepSORT Advantages:**
- Superior re-identification through appearance features
- Perfect stability (100% vs 88.6% efficiency)
- Zero false positive tracks
- Better handling of identical appearances
- Robust to lighting and pose variations

**ByteTrack Advantages:**
- Lower computational requirements
- Simpler parameter tuning
- Real-time performance guarantee
- No dependency on appearance model quality

---


## Method 3: YOLO + StrongSORT (BoxMOT + OSNet)

### Overview

**StrongSORT** is an advanced multi-object tracking framework that enhances DeepSORT by integrating a stronger appearance model (**OSNet**) and an improved association strategy. It is designed to handle complex player re-identification scenarios including occlusions, varying poses, and lighting changes.

### How StrongSORT Solves the Problem

**Core Innovations:**

* **OSNet Appearance Model**: Extracts rich, discriminative features to distinguish between players.
* **BoxMOT Integration**: Plug-and-play flexible tracker with smarter matching.
* **Long-Term Tracking Memory**: Keeps player IDs alive even during long disappearances.
* **IoU + Appearance Fusion**: Robust association using both spatial and appearance metrics.

---
##
Get free ReID model file:

##You can download it automatically using the BoxMOT CLI:
```bash
boxmot track --reid-model osnet_x0_25_msmt17.pt
```

### Key Algorithm Components

#### 1. StrongSORT Tracker Configuration

```python
from strongsort import StrongSORT

tracker = StrongSORT(
    model_weights_path='osnet_x0_25_msmt17.pt',  # OSNet weights
    device='cuda',                               # GPU acceleration
    max_age=300,                                 # 12 seconds track persistence
    n_init=3,                                    # 3 detections to confirm a track
    max_iou_distance=0.7,                        # IoU threshold
    max_cosine_distance=0.2,                     # Strict appearance matching
    embedder_model='osnet_x0_25',                # Appearance model
    half=True                                    # Mixed precision
)
```

#### 2. Detection Filtering for StrongSORT

```python
def filter_detections_for_strongsort(detections, min_area=800, max_area=70000,
                                     min_aspect_ratio=1.0, max_aspect_ratio=5.0,
                                     nms_threshold=0.35):
    """Filter realistic player bounding boxes."""
    areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
            (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    area_mask = (areas >= min_area) & (areas <= max_area)

    widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    aspect_ratios = heights / widths
    aspect_mask = (aspect_ratios >= min_aspect_ratio) & (aspect_ratios <= max_aspect_ratio)

    filtered_detections = detections[area_mask & aspect_mask]
    return filtered_detections.with_nms(threshold=nms_threshold, class_agnostic=True)
```

#### 3. Main Tracking Loop

```python
for frame_idx, frame in enumerate(video_frames):
    # Step 1: YOLO Detection
    results = detection_model(frame, conf=0.85, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Step 2: Filter detections
    player_detections = detections[detections.class_id != BALL_CLASS_ID]
    player_detections = filter_detections_for_strongsort(player_detections)

    # Step 3: Convert detections
    det_list = []
    for i in range(len(player_detections.xyxy)):
        x1, y1, x2, y2 = player_detections.xyxy[i]
        w, h = x2 - x1, y2 - y1
        if w > 20 and h > 50:
            det_list.append(([x1, y1, w, h], player_detections.confidence[i],
                             player_detections.class_id[i]))

    # Step 4: Update StrongSORT tracker
    tracks = tracker.update_tracks(det_list, frame=frame)

    # Step 5: Collect track outputs
    for track in tracks:
        if track.is_confirmed() and track.time_since_update <= 10:
            tracked_boxes.append(track.to_ltrb())
            tracked_ids.append(track.track_id)
```

---

### Key Features Solving Re-identification

#### 1. **OSNet Appearance Embeddings**

* 256-dimensional vectors capturing rich visual features.
* Robust across clothing changes, occlusion, and lighting variation.
* Trained on MSMT17 (a large-scale person re-ID dataset).

#### 2. **Long-term Track Persistence**

* `max_age = 300` allows 12 seconds of memory during occlusion.
* Maintains ID continuity for players re-entering the scene.

#### 3. **Adaptive Data Association**

* Combines:

  * **IoU** for spatial overlap
  * **Cosine Distance** for visual similarity
* Prioritizes recently updated tracks.
* Prevents ID switches in dense scenes.

#### 4. **Flexible Tracker Design**

* Swap embedders (e.g., OSNet, Mobilenet, ResNet).
* Custom thresholds for various sports/cameras.
* GPU-accelerated for real-time deployment.

---

### Performance Analysis

#### 📊 STRONGSORT STABILITY ANALYSIS

```
✅ Stable players (30+ frames): 29
⚠️ Transient players (<30 frames): 1
🎯 Tracking efficiency: 96.7%
```

#### 🏆 MOST STABLE PLAYERS

```
Player #7: 337 frames (89.9%)
Player #19: 330 frames (88.1%)
Player #44: 299 frames (79.9%)
Player #12: 275 frames (73.4%)
Player #28: 256 frames (68.3%)
Player #31: 244 frames (65.0%)
Player #2: 229 frames (61.2%)
Player #25: 218 frames (58.2%)
Player #6: 206 frames (55.0%)
Player #38: 195 frames (52.1%)
```

#### ⚠️ TRANSIENT PLAYERS

```
Player #99: 5 frames (likely false positive)
```

---

### Performance Metrics

| Metric                        | Value                                  |
| ----------------------------- | -------------------------------------- |
| **Total Unique Players**      | 30                                     |
| **Stable Tracking Rate**      | 96.7% (29/30)                          |
| **Top Player Frame Coverage** | Up to 89.9%                            |
| **False Positive Rate**       | 3.3%                                   |
| **Re-ID Performance**         | Excellent (handles re-entry scenarios) |

---

### Strengths

* ✅ State-of-the-art re-ID with **OSNet**
* ✅ Long-term occlusion handling
* ✅ Excellent ID consistency
* ✅ Modular and extensible framework
* ✅ Effective in challenging visual conditions

### Limitations

* ⚠️ Higher computational demand (especially embedding step)
* ⚠️ Requires GPU for smooth performance
* ⚠️ Sensitive to parameter tuning for different scenes

---

### Optimization Achieved

#### Recommended Configuration

```python
YOLO_CONF_THRESHOLD = 0.85
max_age = 300
n_init = 3
max_cosine_distance = 0.2
embedder_model = 'osnet_x0_25'
```

#### Optimization Highlights

* 🎯 Use **OSNet** for discriminative features
* ⏱ `max_age = 300` for long occlusions
* 🔒 Tight cosine distance (0.2) for strict matching
* 🧹 Detection filtering to reduce noise

---

### DeepSORT vs StrongSORT vs ByteTrack Comparison

| Metric                  | ByteTrack    | DeepSORT         | StrongSORT        |
| ----------------------- | ------------ | ---------------- | ----------------- |
| **Approach**            | IoU + Kalman | IoU + Appearance | IoU + OSNet       |
| **Re-ID Capability**    | None         | Deep Features    | Strongest (OSNet) |
| **Tracking Efficiency** | 88.6%        | 100%             | 96.7%             |
| **Transient Players**   | 4            | 0                | 1                 |
| **Top Player Coverage** | \~80%        | \~87.7%          | \~89.9%           |
| **Complexity**          | Low          | Medium           | High              |
| **Speed**               | Fastest      | Slower           | Slowest           |

---

### Usage Instructions

#### ▶️ Run the Tracker

```bash
python strongsort_player_tracking.py
```

#### 📝 Output

* Annotated video with consistent player IDs
* Re-identification analysis report
* Visual tracking statistics and optimization suggestions

Great! Let’s add that to your **README.md** in the same style. You can copy-paste this Markdown section below:

---

## 🎥 Output Videos

This folder contains example output videos demonstrating the player and ball detection, tracking, and analysis:

* **Google Drive Folder Link**: [Output Videos](https://drive.google.com/drive/folders/1O1VvVHEYofreO2eZWwT_ksFGs7gkFJqH?usp=sharing)

> 📌 These videos showcase the results of the pipeline, including consistent player IDs and tracked ball trajectories.

---
## 🎥 Output result:

* bytetrack+iou method:
  
![image](https://github.com/user-attachments/assets/4be50443-52a2-4ef6-862d-ecbde170701d)

* yolo + deepsort method:

  ![image](https://github.com/user-attachments/assets/c093171b-c6d5-4915-87ef-cef51372c5c3)


* yolo + strong sort method:

  ![image](https://github.com/user-attachments/assets/f2a13ccc-c5d4-475d-83d4-d309fa1cb443)



---


