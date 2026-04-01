# PPE Detection Model

This document describes the trained object detection model used in the Site Safety Vision system for detecting Personal Protective Equipment (PPE) and safety violations in construction environments.

---

## Model Overview

The model is based on the YOLOv8 architecture and is trained to detect both PPE items and safety violations in real-world construction site scenarios.

### Model Details

- Architecture: YOLOv8m
- Framework: Ultralytics YOLOv8
- Image Size: 640
- Training Device: NVIDIA A100 GPU
- Epochs: 50
- Model File: best_merged_ppe_yolov8m.pt

---

## Classes

The model is trained on 14 classes grouped as follows:

### PPE Classes
- Hardhat
- Safety Vest
- Gloves
- Goggles
- Mask

### Violation Classes
- NO-Hardhat
- NO-Safety Vest
- NO-Gloves
- NO-Goggles
- NO-Mask

### Context Classes
- Person
- Safety Cone
- Ladder
- Fall-Detected

---

## Dataset

### Primary Dataset
- Approximately 44,000 annotated images
- YOLO format annotations
- Balanced mix of indoor and outdoor environments

### Custom Dataset
- 125 images
- Focus on edge cases such as blue safety vests
- Collected from royalty-free sources (Pexels, Unsplash)
- Manually annotated using Roboflow

### Dataset Characteristics

- Multiple workers per frame
- Occlusions and varying viewpoints
- Different lighting conditions (daylight, shadow, artificial)
- Real-world construction environments

---

## Performance Metrics

| Metric        | Value |
|--------------|------|
| Precision     | 0.713 |
| Recall        | 0.852 |
| mAP@50        | 0.792 |
| mAP@50-95     | 0.537 |

### Interpretation

- High recall ensures most safety violations are detected
- Moderate precision indicates some false positives
- Lower mAP@50-95 reflects bounding box precision limitations

---

## Training Configuration

- Batch Size: 16
- Epochs: 50
- Optimizer: Default YOLOv8 optimizer
- Loss Functions: Box loss, Class loss, DFL loss
- Augmentation: Standard YOLO augmentations (scaling, flipping, etc.)

---

## Training Process

The model was trained using a hybrid dataset strategy:

1. Initial training on large public PPE dataset  
2. Performance analysis to identify weaknesses  
3. Custom dataset creation targeting edge cases  
4. Dataset merging and retraining  
5. Validation and performance evaluation  

---

## Strengths

- Strong detection of helmets and safety vests  
- High recall reduces missed violations  
- Handles multi-worker scenes effectively  
- Robust under varying lighting conditions  
- Improved detection of non-standard PPE (blue vests)  

---

## Limitations

- Yellow/orange clothing may be misclassified as safety vests  
- Performance decreases under heavy occlusion  
- Small or distant objects reduce detection accuracy  
- Class imbalance affects some categories  
- Absence detection is more challenging than presence detection  

---

## Model Usage

Place the model file in the following directory:

```
models/
```

Example:

```
models/best_merged_ppe_yolov8m.pt
```

Ensure that the configuration file references the correct model path before running inference.

---

## Inference Example

```
python scripts/run_inference.py --input data/samples/sample_video.mp4
```

---

## Notes

- The model is optimized for real-time inference
- Works best when workers are clearly visible
- Performance depends on input resolution and scene complexity

---

## Author

Ruchira Amarakone