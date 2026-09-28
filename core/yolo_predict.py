import cv2
from pathlib import Path
from ultralytics import YOLO
import json
import pandas as pd


def predict_bounding_boxes(
    model_path: str,
    image_path: str,
    conf_threshold: float = 0.25,
    save_annotated_image: bool = True,
    output_image_path: str = "prediction_output.jpg",
    check_iou: bool = True,
    selected_classes: list = None
    ):
    """
    Runs YOLO inference on an image and returns detections in the format:
    (x1, y1, dx, dy, label, confidence)
    """
    image_file = Path(image_path).name

    # 1. Load trained YOLO model
    model = YOLO(model_path)
    classes = model.names  #   Get class names from the model

    # 2. Perform inference
    results = model.predict(source=image_path, conf=conf_threshold, iou=0.3, classes = selected_classes)
    result = results[0]  # Get results for the first image

    formatted_predictions = []

    # 3. Parse bounding box coordinates and metadata
    if result.boxes is not None and len(result.boxes) > 0:
        # Get bounding boxes in [x1, y1, x2, y2] pixel coordinates
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        box_id = 0

        for box, conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
            x1, y1, x2, y2 = box
            dx = x2 - x1
            dy = y2 - y1
            label = model.names[cls_id]
            objectType = "RectROI"
            object_name = f"object_{box_id}"
            confidence = float(conf)

            box_id += 1

            record = {
                "objectName": object_name,
                "type": objectType,
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "dx": round(float(dx), 2),
                "dy": round(float(dy), 2),
                "file": image_file,
                "label": label,
                "confidence": round(confidence, 4)
            }
            formatted_predictions.append(record)

    # 4. Save visualization of predictions
    if save_annotated_image:
        annotated_frame = result.plot()  # Draws bounding boxes with class names and confidence
        cv2.imwrite(output_image_path, annotated_frame)
        print(f"Annotated image saved to '{output_image_path}'.")

            # Remove duplicate detections: for boxes with IoU >= 0.5, keep the
        # detection with the greater confidence score.

    prediction_results = pd.DataFrame(formatted_predictions)

    if isinstance(prediction_results, pd.DataFrame) and not prediction_results.empty:
        confidence_column = next(
            (name for name in ("confidence", "conf", "score")
                if name in prediction_results.columns), None
        )
        box_columns = {"x1", "y1", "dx", "dy"}
        if confidence_column and box_columns.issubset(prediction_results.columns):
            kept_indices = []
            for index, row in prediction_results.sort_values(
                confidence_column, ascending=False, kind="stable"
            ).iterrows():
                x1, y1 = float(row["x1"]), float(row["y1"])
                x2, y2 = x1 + float(row["dx"]), y1 + float(row["dy"])
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                discard = False
                for kept_index in kept_indices:
                    kept = prediction_results.loc[kept_index]
                    kx1, ky1 = float(kept["x1"]), float(kept["y1"])
                    kx2, ky2 = kx1 + float(kept["dx"]), ky1 + float(kept["dy"])
                    intersection = max(0.0, min(x2, kx2) - max(x1, kx1)) * max(
                        0.0, min(y2, ky2) - max(y1, ky1)
                    )
                    kept_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
                    union = area + kept_area - intersection
                    if union > 0 and intersection / union >= 0.5:
                        discard = True
                        break
                if not discard:
                    kept_indices.append(index)
            prediction_results = prediction_results.loc[kept_indices].copy()

    return prediction_results, classes


if __name__ == "__main__":
    # Path to trained best model weights
    MODEL_WEIGHTS = "runs/detect/rectroi_detector/weights/best.pt"
    TEST_IMAGE = "path/to/test_image.jpg"

    predictions = predict_bounding_boxes(
        model_path=MODEL_WEIGHTS,
        image_path=TEST_IMAGE,
        conf_threshold=0.3,
        save_annotated_image=True
    )

    print("\nExtracted Predictions (x1, y1, dx, dy, label, confidence):")
    for pred in predictions:
        print(pred)