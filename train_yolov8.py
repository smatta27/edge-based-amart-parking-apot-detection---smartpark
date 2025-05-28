import os
from ultralytics import YOLO

def main():
    # Path to the dataset YAML file
    data_yaml = os.path.join('annotations', 'data.yaml')
    # Choose model size: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'
    model_size = 'yolov8n.pt'  # Change to your preferred model size
    epochs = 50                # Number of training epochs
    imgsz = 640                # Image size

    # Train the model
    model = YOLO(model_size)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project='runs/train',
        name='yolov8-parking-finetune',
        exist_ok=True
    )

    print('Training complete! Best weights are saved in runs/train/yolov8-parking-finetune/weights/best.pt')

if __name__ == '__main__':
    main() 