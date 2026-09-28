
from pathlib import Path


if __name__ == "__main__":
    augment_yolo_dataset(
        images_dir="yolo_dataset/images/train",
        labels_dir="yolo_dataset/labels/train",
        output_images_dir="augmented_dataset/images/train",
        output_labels_dir="augmented_dataset/labels/train",
        num_aug_per_image=3  # Multiplies training set by 4x (1 original + 3 augmented copies)
    )