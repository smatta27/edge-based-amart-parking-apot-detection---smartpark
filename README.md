# Smart Parking Spot Detector

A parking spot detection system using YOLOv8 for car detection and OpenCV for image/video processing. This project supports both static image and video parking lot analysis, and can be fine-tuned for custom aerial views.

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