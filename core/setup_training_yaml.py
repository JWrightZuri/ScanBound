import os
import shutil
import random
from pathlib import Path




if __name__ == "__main__":
    # Pass the same label-to-id mapping used in the conversion step
    classes = {"label1": 0, "label2": 1}

    split_yolo_dataset(
        images_dir="./images",
        labels_dir="./labels",
        output_dir="./yolo_dataset",
        val_ratio=0.2,
        class_mapping=classes
    )