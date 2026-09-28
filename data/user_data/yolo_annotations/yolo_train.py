from ultralytics import YOLO

def train_yolo_model(
    data_yaml_path: str = "./dataset.yaml",
    model_weights: str = "yolo26n.pt",  # e.g., 'yolo11n.pt', 'yolov8n.pt', 'yolo26n.pt'
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0",  # '0' for CUDA GPU, 'mps' for Apple Silicon, or 'cpu'
    project_name: str = "testing",
    run_name: str = "rectroi_detector"
):
    """
    Trains a YOLO object detection model on custom data using the Ultralytics API.
    """
    # 1. Load a pretrained model baseline (recommended for faster convergence)
    model = YOLO(model_weights)

    # 2. Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project_name,
        name=run_name,
        pretrained=True,
        plots=True,          # Generates loss plots, PR curves, and confusion matrices
        save=True,           # Automatically saves best.pt and last.pt checkpoints
        workers=4            # Dataloader worker subprocesses
    )

    print("Training complete.")
    print(f"Results and weights saved to: {results.save_dir}")

if __name__ == "__main__":
    # Guard required for PyTorch multi-worker execution
    train_yolo_model(
        data_yaml_path="./dataset.yaml",
        model_weights="yolo26n.pt",
        epochs=50,
        batch_size=16,
        device="0"  # Change to 'cpu' if no CUDA GPU is available
    )