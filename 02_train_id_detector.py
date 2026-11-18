from ultralytics import YOLO
import os

# 1. Load a pre-trained YOLOv8 "nano" model. 
# It is the fastest and lightest version, perfect for real-time CCTV processing.
print("Loading base YOLOv8 model...")
model = YOLO('yolov8n.pt') 

# 2. Define the path to your dataset's YAML file
# We use a relative path so it works perfectly from your pendrive on any computer
yaml_path = os.path.join(".", "dataset", "id_cards", "data.yaml")

# 3. Train the model
if __name__ == '__main__':
    print("Starting training process...")
    # epochs=50: The AI will review the dataset 50 times to learn.
    # imgsz=640: Standard YOLO resolution.
    # device=0: Tells the script to use your dedicated NVIDIA GPU.
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0, 
        project="database",
        name="id_card_model"
    )
    print("Training complete! The best model is saved in database/id_card_model/weights/best.pt")
