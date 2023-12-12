# Robust Human Target Detection and Acquisition

This project leverages computer vision techniques and deep learning models to detect and acquire human targets within videos.

## Overview

The project is designed to perform the following tasks:
- **Recording Video:** Utilizes OpenCV to capture a video using the default camera.
- **Human Activity Detection:** Employs TensorFlow and OpenCV for human detection within the recorded video.
- **Video Playback:** Displays the processed video with identified human targets.

## Code Files

### 1. `my_video_capture.py`
- **Functionality:** Records a video using the default camera and saves it to a specified output path.
- **Technologies:** OpenCV for video capture and recording.

### 2. `main_file.py`
- **Functionality:** Implements human detection using TensorFlow and OpenCV.
- **Technologies:** TensorFlow for deep learning-based detection, OpenCV for image processing.

### 3. `play_my_video.py`
- **Functionality:** Plays the processed video, displaying identified human targets.
- **Technologies:** OpenCV for video playback and visualization.

### 4. `run_all.py`
- **Functionality:** Orchestrates the workflow, capturing a video, processing it for human detection, and playing the output video.
- **Technologies:** Integrates functionalities from other code files to execute the complete workflow.

## Technologies Used

### Libraries/Frameworks
- **OpenCV:** Used for video capture, image processing, and video playback.
- **TensorFlow:** Employed for deep learning-based human detection.
- **Keras:** Utilized for loading and running pre-trained deep learning models.
- **Matplotlib:** Used for displaying images.

## Deep Learning Models

The specific deep learning model used for human detection is loaded from TensorFlow Hub:
- **Module Handle:** `https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1`
- **Model Type:** Single Shot Multibox Detector (SSD) with MobileNet V2 backbone.
- **Purpose:** Object detection, particularly for detecting humans in images or video frames.

## Instructions for Cloning the Repository

**Clone the Repository:**
git clone https://github.com/AbhayPancholi/Activity_Detection.git

**Navigate to your folder:**
cd path/to/your/folder

**Install required packages:**
pip install -r requirements.txt

Once the above steps are done, you can open the folder containing the files in an IDE like VS Code and run the `run_all.py`.

## Workflow

1. `my_video_capture.py` captures a video, saving it to a specified output path.
2. `main_file.py` employs the loaded SSD MobileNet V2 model to detect human targets along with a image classifier that classifies the images within the recorded video frames.
3. `play_my_video.py` plays the processed video.
4. `run_all.py` orchestrates the workflow by integrating video recording, human detection, and video playback functionalities.


This project demonstrates the integration of computer vision techniques and deep learning models for human target detection and visualization within videos.
