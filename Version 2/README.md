# Robust Human Target Detection and Acquisition

This repository contains the implementation of a robust human target detection and acquisition system using YOLOv8 for activity detection and a custom algorithm for tracking and counting persons in video footage.

## Project Overview

The system consists of two main components:
1. **Activity Detection**: Utilizing YOLOv8 trained on a custom dataset for detecting human activities.
2. **Tracking and Counting**: A custom implementation for tracking and counting detected persons from video input.

Both components are structured into separate modules:
- **Activity Detection**: Responsible for detecting human targets using a YOLOv8 model trained on a custom dataset.
- **Tracking and Counting**: Responsible for tracking the detected targets and counting them over the duration of a video.

## File Structure

- Activity_Detection  
    - Yolo_v8_on_custom_dataset.ipynb   
    - exp.py                            
- Tracking_and_Counting/
    - track_count_persons.ipynb    


### Activity Detection

The `Yolo_v8_on_custom_dataset.ipynb` file contains the following:
- Details of the YOLOv8 model.
- Custom dataset preparation.
- Model training, validation, and evaluation.
- Visualization of detection results.

You can use `exp.py` to execute the trained YOLOv8 model for real-time activity detection.

### Tracking and Counting

The `track_count_persons.ipynb` file implements an algorithm for tracking and counting persons detected in a video stream. It uses the results from the activity detection model and applies tracking techniques to count human targets over time.

## How to Run

1. **Activity Detection**:
   - Train the YOLOv8 model using the `Yolo_v8_on_custom_dataset.ipynb`.
   - Then download the trained model and run the `exp.py` to execute the model on new inputs.

2. **Tracking and Counting**:
   - Open and run the `track_count_persons.ipynb` notebook to track and count detected persons in a given video.

## Dependencies

The project requires the following dependencies:
- Python 3.7+
- OpenCV
- YOLOv8 (Ultralytics)
- Numpy
- Matplotlib
- PyTorch
- Deep Sort
  
## Results
The YOLOv8 model is capable of detecting multiple human activities in real-time, while the tracking system efficiently counts the number of persons in video footage, offering robust performance in varied conditions.

## Future Improvements
- Enhance tracking accuracy by integrating advanced techniques like Kalman filtering or optical flow.
- Extend the model to detect and track multiple objects apart from human targets.
