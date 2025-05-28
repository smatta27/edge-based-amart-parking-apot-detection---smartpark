# Smart Parking Spot Detector

A parking spot detection system using YOLOv8 for car detection and OpenCV for image/video processing.

## Features

- Car detection using YOLOv8
- Parking spot selection via a GUI tool
- Occupancy status display for each spot (currently using fixed values for demonstration)
- Counter of occupied/vacant spots
- Support for both image and video files
- Automatic saving of processed results

## Workflow

### 1. Select Parking Spots
First, you need to define the parking spots on your image or video frame:
```bash
python select_parking_spots.py path/to/image.jpg --output spots.json
```
This will open a GUI where you can draw rectangles for each parking spot. The coordinates will be saved to `spots.json`.

TEAM: DONT WORRY ABOUT THIS, IVE ALREADY DONE IT (Check the spots.json files)

### 2. Run the Detector
#### For Images:
```bash
python parking_detector.py path/to/image.jpg --spots spots.json
```
- The script will display the image with each parking spot outlined and numbered.
- Spots are shown as red (occupied) or green (vacant) based on a fixed set of values for demonstration.
- The processed image is saved with a `_processed` suffix.

TEAM: RUN 'python3 parking_detector.py carPark.jpg --spots spots.json'

#### For Videos:
```bash
python parking_detector_video.py path/to/video.mp4 --spots spots.json
```
- The script will display the video with each parking spot outlined and numbered.
- Spots are shown as red (occupied) or green (vacant) based on a fixed set of values for demonstration.
- The processed video is saved with a `_processed` suffix.

TEAM: RUN 'python3 parking_detector_video.py carPark.mp4 --spots spots_video.json'

## How It Works
- Parking spot locations are defined manually using the GUI tool and saved to a JSON file.
- When running the detector, the script uses a predefined set of spot statuses to display which spots are occupied or vacant. This is useful for demonstration and testing.
- The spot numbers are shown on the image/video so you can easily identify and update which spots are free or taken.

## Limitations & Next Steps
- **Current Limitation:** The video detector currently uses fixed spot statuses, so it does not adapt if vehicles leave or enter spots during the video. This means the occupancy display will not update dynamically as cars move.
- **Future Work:** To make the video detector fully adaptive, the detection logic should be improved so that the script can automatically update spot statuses in real time as vehicles arrive or depart. This will require more robust object detection and overlap analysis.

## Notes
- Supported image formats: .jpg, .png, etc.
- Supported video formats: .mp4, .avi, .mov
- For demonstration, the occupancy status is set based on a fixed list of spot indices. You can update these indices in the code as needed.

## Setup

1. Install the required dependencies:
pip install -r requirements.txt
```