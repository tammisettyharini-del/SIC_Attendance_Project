
from ultralytics import YOLO
import torch

if __name__ == "__main__":

    print("Loading YOLOv8 Nano...")
    model = YOLO("yolov8n.pt")

    device = 0 if torch.cuda.is_available() else "cpu"
    batch_size = 16 if device == 0 else 4

    print(f"Training device: {device}")
    print("Starting ID-card detector training...")

    model.train(
        data="./dataset/id_cards/data.yaml",
        epochs=50,
        imgsz=640,
        batch=batch_size,
        device=device,
        workers=0,
        project="database",
        name="id_card_model",
        exist_ok=True
    )

    print("Training completed!")
    print("Best model: database/id_card_model/weights/best.pt")
