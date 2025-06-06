# Smart Parking Spot Detector


Use Case

Problem: A lot of times, finding a free parking spot can be frustrating and time-consuming. This is a common issue among UCR students as well, especially when trying to find parking in the lots. Often, during rush hours, drivers have to drive in circles, increasing fuel usage, carbon emissions, and traffic congestion.

This problem is ideal for edge computing because we need real-time decision-making that is able to function offline and with intermittent connectivity. There also needs to be fast responses so drivers can be informed accordingly about parking spots. Privacy-sensitive video processing is also involved, and it is better to keep information locally on a device

Industry: This issue falls under the transportation and smart Infrastructure category. Currently, a lot of solutions depend on cloud connectivity or human intervention. This makes them unreliable or expensive. Our automated detection of available parking spots in real time will ensure minimal latency, low network load, and high reliability.

Challenges:

Some challenges that we might go through while developing the project include accurately detecting small vehicles or unusual parking angles, lighting such as night vs day, and syncing different devices.

Solution

Materials Jetson Nano Camera MicroSD Power Supply Ethernet Cable Keyboard/Mouse Monitor Laptop

Software Python 3 OpenCV PyTorch YOLOv5/ MobileNet

We propose SmartPark, a scalable and privacy-friendly smart parking system using Jetson Nano devices with cameras for real-time parking spot detection at the edge. The Jetson Nano will monitor a fixed number of parking spots via camera. Then, object detection modelsYOLOv5 or MobileNet, are used to determine if a spot is occupied or available. The videos are processed locally to protect user privacy and reduce network traffic.

Long-term Full-Scale To achieve a full-scale solution, we plan to deploy multiple edge devices across large parking spaces. Each edge device will process data locally and communicate with a central fog node. We can also add predictive analytics using AI to forecast spot availability based on trends. We can also possibly integrate payment or permit systems for billing and enforcement.

Demo For our demo, we will record a short video showcasing our edge-based parking spot detection system in action using a small, indoor mock parking lot. Demo Setup: One Jetson Nano device connected to a USB camera A physical mock parking lot setup using printed paper spots and toy cars A monitor connected to display parking status in real time

What We’ll Show in the Video: The Jetson Nano is capturing live video from the overhead camera The object detection model (YOLOv5 or MobileNet) running on the Jetson, identifying parked cars Real-time visualization of parking spot status (e.g., green = available, red = taken) on screen The entire system works locally without a cloud or internet connection

Why a Recorded Demo: Recording allows us to clearly explain and walk through each part of the system It captures the model working in real-time and makes it easier to share or present We can also show challenges like camera angle or lighting conditions during testing This video will serve as our proof of concept and demonstrate that SmartPark works as an edge-based parking detection system using a single Jetson Nano.


## Features

- Car detection using YOLOv8 (pre-trained or fine-tuned)
- Manual parking spot selection via a GUI tool
- Occupancy status display for each spot (red = occupied, green = vacant)
- Counter of occupied/vacant spots
- Support for both image and video files
- Automatic saving of processed results
- Fine-tuning pipeline for custom datasets

## Workflow

### 1. Select Parking Spots
First, define the parking spots on your image or a frame from your video:
```bash
python select_parking_spots.py path/to/image.jpg --output spots.json
```
This opens a GUI to draw rectangles for each parking spot. The coordinates are saved to `spots.json` (or `spots_video.json` for video).

### 2. (Optional) Extract Frames for Annotation
To fine-tune YOLOv8 for your specific video, extract frames:
```bash
python frame.py carPark.mp4 --output_dir frames --interval 30
```
This saves every 30th frame to the `frames/` directory.

### 3. Annotate Frames
Label all cars in the extracted frames using a tool like [LabelImg](https://github.com/tzutalin/labelImg) or [Roboflow](https://roboflow.com/). Export annotations in YOLOv8 txt format.

### 4. Organize Dataset
Structure your dataset as follows:
```
annotations/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  data.yaml
```

### 5. Fine-tune YOLOv8
Train a YOLOv8 model on your custom dataset:
```bash
python train_yolov8.py
```
- Adjust model size, epochs, and image size in `train_yolov8.py` as needed.
- The best weights will be saved in `runs/train/yolov8-parking-finetune/weights/best.pt`.

### 6. Run the Detector
#### For Images (hardcoded or detection-based):
```bash
python parking_detector.py path/to/image.jpg --spots spots.json
```
- Shows the image with each parking spot outlined and numbered.
- Spots are colored red (occupied) or green (vacant).
- The processed image is saved with a `_processed` suffix.

#### For Videos (detection-based, supports fine-tuned weights):
```bash
python parking_detector_video.py path/to/video.mp4 --spots spots_video.json --model runs/train/yolov8-parking-finetune/weights/best.pt --conf 0.01
```
- Shows the video with each parking spot outlined and numbered.
- Spots are colored red (occupied) or green (vacant) in real time as cars move.
- The processed video is saved with a `_processed` suffix.

## How It Works
- Parking spot locations are defined manually and saved to a JSON file.
- For images, you can use a fixed set of spot statuses for demonstration, or use detection.
- For videos, YOLOv8 detects cars in each frame; spots are marked occupied if a car overlaps the spot.
- Fine-tuning on your own frames greatly improves detection accuracy for your specific scene.

## Limitations & Next Steps
- **Pre-trained YOLOv8 may not work well on aerial parking lot views.**
- **Fine-tuning on your own frames is highly recommended for best results.**
- Future work: Add support for more robust detection, multi-class support, and automatic spot assignment.

## Notes
- Supported image formats: .jpg, .png, etc.
- Supported video formats: .mp4, .avi, .mov
- For demonstration, the occupancy status can be set based on a fixed list of spot indices (hardcoded) or dynamically via detection.
- See `train_yolov8.py` for training details and adjust as needed for your dataset.

## Setup

1. Install the required dependencies:
pip install -r requirements.txt
