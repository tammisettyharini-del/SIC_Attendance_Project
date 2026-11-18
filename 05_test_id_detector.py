
from ultralytics import YOLO

model = YOLO("database/id_card_model/weights/best.pt")

results = model.predict(
    source="dataset/id_cards/test/images",
    conf=0.25,
    save=True
)

print("Testing completed!")
print("Number of images tested:", len(results))
print("Annotated images saved in runs/detect/predict")
