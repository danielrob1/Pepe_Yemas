from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=1024,
        batch=8,
        name="yemas_model",
        device="0"  # GPU
    )

if __name__ == "__main__":
    train()