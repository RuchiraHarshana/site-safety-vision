# Model Summary

This folder contains the trained YOLOv8 model used for construction safety detection.

## Model

- **File:** `best.pt`
- **Architecture:** YOLOv8s
- **Task:** Object detection (workers and PPE)

## Classes

The model is trained to detect the following classes:

- person
- helmet
- vest
- gloves
- boots

Core compliance logic primarily relies on:
- person
- helmet
- vest

## Performance (Validation)

- **Precision:** ~0.80  
- **Recall:** ~0.94  
- **mAP@50:** ~0.82  
- **mAP@50–95:** ~0.69  

## Observations

- Strong performance for **helmet, vest, and boots**
- **Person detection** is moderate but acceptable
- **Gloves detection** is weaker due to small object size and visibility challenges

## Notes

- The model was trained on a cleaned and extended PPE dataset
- It is used for both image inference and video-based tracking pipelines
- Final safety decisions are made using additional rule-based logic on top of model predictions