import cv2
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Extract frames from a video file.')
parser.add_argument('video_path', type=str, help='Path to the video file')
parser.add_argument('--output_dir', type=str, default='frames', help='Directory to save extracted frames')
parser.add_argument('--interval', type=int, default=30, help='Extract every Nth frame (default: 30)')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {args.video_path}")
    exit(1)

frame_count = 0
saved_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % args.interval == 0:
        frame_filename = os.path.join(args.output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"Extracted {saved_count} frames to {args.output_dir}/")