# Site Safety Vision

Site Safety Vision is an intelligent computer vision system designed for automated safety compliance monitoring in construction and industrial environments. The system detects Personal Protective Equipment (PPE), tracks workers across frames, applies temporal safety rules, and generates alerts and analytics in real time.

---

## Problem Statement

Construction sites are among the most hazardous workplaces, where failure to wear PPE such as helmets and safety vests leads to serious injuries and fatalities. Traditional monitoring relies on human supervision, which is:

- Inconsistent due to fatigue  
- Difficult to scale across large sites  
- Unable to track behavior over time  

This project addresses these limitations by introducing an automated, real-time safety monitoring system using deep learning and rule-based reasoning.

---

## System Architecture

```
YOLOv8 Detection
        ↓
ByteTrack Tracking
        ↓
PPE Matching (Worker ↔ PPE)
        ↓
Safety Rules Engine (Temporal Reasoning)
        ↓
Alert Generation
        ↓
Visualization and Analytics
        ↓
Incident Review
```

---

## Core Components

### 1. PPE Detection (YOLOv8)

- Model: YOLOv8m
- Image size: 640
- Framework: Ultralytics YOLOv8
- Hardware: NVIDIA A100 GPU

Detects:
- Hardhat
- Safety Vest
- Gloves
- Goggles
- Mask
- Person
- Safety violations (NO-Hardhat, NO-Vest, etc.)

---

### 2. Multi-Object Tracking (ByteTrack)

- Maintains consistent worker identities across frames
- Enables temporal reasoning
- Handles occlusions and movement

---

### 3. PPE Matching

Associates PPE with workers using:
- Bounding box overlap
- Spatial proximity thresholds

---

### 4. Safety Rules Engine

A stateful reasoning system that evaluates safety over time.

Key features:

- Temporal memory tracking
- Violation duration thresholds
- Negative evidence (NO-Hardhat, NO-Vest)
- Visibility-aware reasoning
- Conflict handling
- Multi-state classification

Worker states:
- Safe
- Uncertain
- Unsafe

---

### 5. Alert System

Generates structured alerts:

| Level     | Meaning        |
|----------|--------------|
| Info     | Safe          |
| Warning  | Needs review  |
| Critical | Unsafe        |

Each alert includes:
- Worker ID
- Violation reason
- Supporting notes

---

### 6. Analytics Engine

Computes:

- Average workers per frame  
- Helmet compliance rate  
- Vest compliance rate  
- Full PPE compliance rate  

---

## Model Performance

| Metric        | Value |
|--------------|------|
| Precision     | 0.713 |
| Recall        | 0.852 |
| mAP@50        | 0.792 |
| mAP@50-95     | 0.537 |

### Interpretation

- High recall ensures minimal missed safety violations  
- Moderate precision indicates some false positives  
- Lower mAP@50-95 suggests bounding box accuracy can be improved  

---

## Dataset

### Primary Dataset
- ~44,000 annotated images
- 14 classes (PPE + violations + context)

### Custom Dataset
- 125 images
- Focus: blue safety vests and edge cases

### Dataset Strengths
- Indoor and outdoor environments  
- Multiple lighting conditions  
- Occlusion scenarios  
- Multi-worker scenes  

---

## Setup Instructions

### 1. Clone Repository

```
git clone <your-repo-url>
cd site-safety-vision
```

---

### 2. Create Virtual Environment

#### Windows

```
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Set Python Path

#### Windows PowerShell

```
$env:PYTHONPATH="src"
```

#### Windows CMD

```
set PYTHONPATH=src
```

#### Linux/macOS

```
export PYTHONPATH=src
```

---

## Input Data Instructions

Place your input images or videos here:

```
G:\site-safety-vision\data\samples
```

Example:

```
G:\site-safety-vision\data\samples\sample_video.mp4
G:\site-safety-vision\data\samples\image1.jpg
```

---

## Model Setup

Place trained model inside:

```
models/
```

Example:

```
models/best_merged_ppe_yolov8m.pt
```

---

## Running the Project

### Run Inference

```
python scripts/run_inference.py --input data/samples/sample_video.mp4
```

---

### Run Analytics

```
python scripts/run_analytics.py --json-path outputs/predictions/sample_video_results.json
```

---

### Run Full Pipeline

```
python app.py
```

---

### Run Incident Review

```
python scripts/review_incidents.py
```

---

## Output Location

All outputs are saved in:

```
G:\site-safety-vision\outputs
```

Includes:

- Annotated videos  
- JSON predictions  
- Analytics reports  
- Alert logs  

---

## Safety Logic Highlights

This system improves over traditional approaches by:

- Using temporal reasoning instead of frame-based decisions  
- Applying violation time thresholds to reduce false positives  
- Handling uncertainty using multi-state classification  
- Leveraging negative evidence for stronger violation detection  

---

## Failure Cases

- Yellow/orange clothing misclassified as safety vests  
- Heavy occlusion reduces detection accuracy  
- Small/distant workers are harder to detect  
- Absence detection remains challenging  

---

## Limitations

- Class imbalance in dataset  
- Limited custom dataset size  
- Reduced performance under extreme conditions  

---

## Future Improvements

- Expand dataset diversity  
- Improve PPE detection using reflective features  
- Deploy on edge devices  
- Integrate multi-camera monitoring  

---

## Reproducibility

To reproduce results:

1. Create virtual environment  
2. Install dependencies  
3. Set PYTHONPATH=src  
4. Place model in models/  
5. Place inputs in data/samples/  
6. Run inference  

---

## Author

Ruchira Amarakone
AI/ML Engineer | Computer Vision