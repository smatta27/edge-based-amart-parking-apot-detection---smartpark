import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os
import json

class ParkingSpotDetector:
    def __init__(self, input_path=None, spots_path='spots.json', overlap_thresh=0.1, model_size='n', conf_thresh=0.05):
        # Initialize YOLO model
        model_path = 'runs/train/yolov8-parking-finetune/weights/best.pt'   #f'yolov8{model_size}.pt'
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        
        # Initialize input source
        self.input_path = input_path
        self.is_video = False
        self.overlap_thresh = overlap_thresh
        
        if input_path:
            # Check if input is video or image
            if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
                self.is_video = True
                self.cap = cv2.VideoCapture(input_path)
            else:
                self.is_video = False
                self.image = cv2.imread(input_path)
                if self.image is None:
                    raise ValueError(f"Could not read image from {input_path}")
        else:
            raise ValueError("Please provide an input path (image or video)")
        
        # Load parking spots from JSON
        self.spots_path = spots_path
        self.load_parking_spots()
        
        # Colors for visualization
        self.colors = {
            'occupied': (0, 0, 255),    # Red
            'vacant': (0, 255, 0),      # Green
            'text': (255, 255, 255),    # White
            'detection': (255, 0, 0)    # Blue for YOLO detections
        }
        
        # Initialize counters
        self.total_spots = len(self.parking_spots)
        self.occupied_spots = 0

    def load_parking_spots(self):
        if not os.path.exists(self.spots_path):
            raise FileNotFoundError(f"Parking spots file '{self.spots_path}' not found. Please run select_parking_spots.py first.")
        with open(self.spots_path, 'r') as f:
            self.parking_spots = json.load(f)
        print(f"Loaded {len(self.parking_spots)} parking spots from {self.spots_path}")

    def check_spot_occupancy(self, detections, spot):
        spot_x1, spot_y1, spot_x2, spot_y2 = spot
        spot_area = (spot_x2 - spot_x1) * (spot_y2 - spot_y1)
        print(f"\nChecking spot coordinates: ({spot_x1}, {spot_y1}) to ({spot_x2}, {spot_y2})")
        print(f"Spot area: {spot_area}")
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det)  # Ensure coordinates are integers
            print(f"Detection coordinates: ({x1}, {y1}) to ({x2}, {y2})")
            
            # Calculate intersection
            x1_intersect = max(spot_x1, x1)
            y1_intersect = max(spot_y1, y1)
            x2_intersect = min(spot_x2, x2)
            y2_intersect = min(spot_y2, y2)
            
            if x2_intersect > x1_intersect and y2_intersect > y1_intersect:
                intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
                overlap_ratio = intersection_area / spot_area
                print(f"Intersection area: {intersection_area}, Overlap ratio: {overlap_ratio:.2f}")
                if overlap_ratio > self.overlap_thresh:
                    print(f"Spot marked as occupied (overlap ratio {overlap_ratio:.2f} > threshold {self.overlap_thresh})")
                    return True
            else:
                print("No intersection found")
        return False

    def process_frame(self, frame):
        results = self.model(frame)
        all_detections = []
        for r in results:
            for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if conf > self.conf_thresh:
                    all_detections.append(box)
        # Draw all YOLO detections in blue for debugging
        for box in all_detections:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue for detections
        occupied_count = 0
        for idx, spot in enumerate(self.parking_spots):
            is_occupied = self.check_spot_occupancy(all_detections, spot)
            if is_occupied:
                occupied_count += 1
                color = self.colors['occupied']  # Red for occupied
            else:
                color = self.colors['vacant']    # Green for vacant
            cv2.rectangle(frame, (spot[0], spot[1]), (spot[2], spot[3]), color, 2)
            cv2.putText(frame, str(idx + 1), (spot[0], spot[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        self.occupied_spots = occupied_count
        available_count = self.total_spots - occupied_count
        counter_text = f"Taken: {occupied_count} / Available: {available_count}"
        cv2.putText(frame, counter_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
        return frame

    def process_image(self):
        processed_image = self.process_frame(self.image)
        output_path = os.path.splitext(self.input_path)[0] + '_processed.jpg'
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image saved to: {output_path}")
        cv2.imshow('Parking Spot Detection', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self):
        if not self.cap.isOpened():
            print("Error: Could not open video source")
            return
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        output_path = os.path.splitext(self.input_path)[0] + '_processed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame)
            out.write(processed_frame)
            cv2.imshow('Parking Spot Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to: {output_path}")

    def run(self):
        if self.is_video:
            self.process_video()
        else:
            self.process_image()

def main():
    parser = argparse.ArgumentParser(description='Parking Spot Detector (Video)')
    parser.add_argument('input_path', type=str, help='Path to input image or video file')
    parser.add_argument('--spots', type=str, default='spots.json', help='Path to parking spots JSON file')
    parser.add_argument('--overlap', type=float, default=0.1, help='Overlap threshold for occupancy (default: 0.1)')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l'], 
                       help='YOLO model size (n=nano, s=small, m=medium, l=large)')
    parser.add_argument('--conf', type=float, default=0.05, help='Confidence threshold for detections (default: 0.05)')
    args = parser.parse_args()
    try:
        detector = ParkingSpotDetector(args.input_path, args.spots, args.overlap, args.model, args.conf)
        detector.run()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 