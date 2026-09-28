import os
import pandas as pd
from pathlib import Path
from PIL import Image
import shutil
import random
import cv2
import albumentations as A


def get_classes(csv_path: str,):
    """
    Extracts unique class labels from a CSV file and returns a mapping of label to integer class ID.
    """
    cols = {
            'objectName': 'objectName',
            'type': 'type',
            'x1': 'x1',
            'y1': 'y1',
            'dx': 'dx',
            'dy': 'dy',
            'file': 'file',
            'label': 'label',
        }

    df = pd.read_csv(csv_path)

    # 1. Filter for RectROI entries
    if cols['type'] in df.columns:
        roi_df = df[df[cols['type']].astype(str).str.strip() == 'RectROI'].copy()
    else:
        roi_df = df.copy()

    if roi_df.empty:
        print("No 'RectROI' records found in the provided CSV file.")
        return

    if cols['label'] in roi_df.columns:
        roi_counts = roi_df[cols['label']].value_counts()
        class_mapping = pd.DataFrame({
            'label': roi_counts.index,
            'roi_count': roi_counts.values,
            'type' : 'RectROI'
        })
        # class_mapping.drop(class_mapping[class_mapping['label'] == 'Empty Label'].index, inplace=True)
        
    else:
        return None
    return class_mapping



def convert_rectroi_csv_to_yolo(
    input_dir: str,
    csv_path: str,
    output_dir: str,
    column_mapping: dict = None,
    class_mapping: dict = None
    ):
    """
    Converts CSV bounding box rows of type 'RectROI' into YOLO annotation format.
    
    YOLO line format: <class_id> <x_center> <y_center> <width> <height> (0.0 to 1.0)
    """
    # Standard default CSV column names (adjust in function call if your CSV differs)
    cols = {
        'objectName': 'objectName',
        'type': 'type',
        'x1': 'x1',
        'y1': 'y1',
        'dx': 'dx',
        'dy': 'dy',
        'file': 'file',
        'label': 'label',
    }
    if column_mapping:
        cols.update(column_mapping)

    df = pd.read_csv(csv_path)

    # 1. Filter for RectROI entries
    if cols['type'] in df.columns:
        roi_df = df[df[cols['type']].astype(str).str.strip() == 'RectROI'].copy()
    else:
        roi_df = df.copy()

    if roi_df.empty:
        print("No 'RectROI' records found in the provided CSV file.")
        return

    # 2. Build class label mapping (text label -> integer class_id)
    if class_mapping is None:
        if cols['label'] in roi_df.columns:
            unique_labels = sorted(roi_df[cols['label']].dropna().unique())
            unique_labels = [label for label in unique_labels if label != 'Empty Label']
            class_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            class_mapping = {'default': 0}

    label_outdir = os.path.join(output_dir, "labels")
    images_outdir = os.path.join(output_dir, "images")


    # 3. Process annotations grouped by image
    grouped = roi_df.groupby(cols['file'])

    for img_name, group in grouped:
        annotations = []
        
        # Get image dimensions from CSV or read directly from the image file
        img_w, img_h = None, None
        # if cols['img_w'] in group.columns and cols['img_h'] in group.columns:
        #     img_w = float(group[cols['img_w']].iloc[0])
        #     img_h = float(group[cols['img_h']].iloc[0])
        if images_outdir:
            try:
                # CSV values may already include the image directory. Use the
                # filename component to avoid constructing a duplicated path.
                img_file = input_dir + "/" + img_name
                with Image.open(img_file) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"Error reading image {img_name}. Skipping: {e}")
                continue

        for _, row in group.iterrows():
            x = float(row[cols['x1']])
            y = float(row[cols['y1']])
            w = float(row[cols['dx']])
            h = float(row[cols['dy']])

            # Extract class ID
            label = row.get(cols['label'], 'default')
            class_id = int(class_mapping.get(label, 0))
            # print(f"Processing {img_name}: Label '{label}' mapped to class ID {class_id}. With coordinates x={x}, y={y}, w={w}, h={h} and image size {img_w}x{img_h}")
            # Convert corner coordinates to YOLO normalized center format
            x_center = max(0.0, min(1.0, (x + w / 2.0) / img_w))
            y_center = max(0.0, min(1.0, (y + h / 2.0) / img_h))
            norm_w = max(0.0, min(1.0, w / img_w))
            norm_h = max(0.0, min(1.0, h / img_h))
            # print(f"Image: {img_name}, Class: {label} ({class_id}), YOLO: {x_center:.6f}, {y_center:.6f}, {norm_w:.6f}, {norm_h:.6f}")
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        # Save annotation file using image stem name (e.g., image1.jpg -> image1.txt)
        txt_filename = Path(str(img_name)).stem + ".txt"
        with open(label_outdir + "\\" + txt_filename, "w") as f:
            f.write("\n".join(annotations))

        shutil.copy2(os.path.join(input_dir, img_name), os.path.join(images_outdir, img_name))  # Copy image to output images dir

    # print(f"Successfully processed {len(grouped)} images. Annotations written to '{label_outdir}'.")


def split_yolo_dataset(
    data_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42
    ):
    """
    Splits image and label files into train/val subsets.
    """
    random.seed(seed)
    
    images_path = data_dir + "\\images"
    labels_path = data_dir + "\\labels"

    # Output directory structure
    train_img_dir = data_dir + "\\images\\train"
    val_img_dir = data_dir + "\\images\\val"
    train_lbl_dir = data_dir + "\\labels\\train"
    val_lbl_dir = data_dir + "\\labels\\val"

    for p in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(p, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Match image files with generated label files
    matched_pairs = []

    for img_file in os.listdir(images_path):
        if os.path.splitext(img_file)[1].lower() in image_extensions:
            lbl_file = f"{os.path.splitext(img_file)[0]}.txt"
            try:
                matched_pairs.append((os.path.join(images_path, img_file), os.path.join(labels_path, lbl_file)))
            except Exception as e:
                print(f"Warning: No matching label found for {img_file}, skipping.")

    if not matched_pairs:
        print("No matching image-label pairs found!")
        return

    # Random split
    random.shuffle(matched_pairs)
    val_size = int(len(matched_pairs) * val_ratio)
    val_pairs = matched_pairs[:val_size]
    train_pairs = matched_pairs[val_size:]

    def copy_files(pairs, img_dest, lbl_dest):
        for img_f, lbl_f in pairs:
            shutil.copy2(img_f, img_dest)
            shutil.copy2(lbl_f, lbl_dest)

    copy_files(train_pairs, train_img_dir, train_lbl_dir)
    copy_files(val_pairs, val_img_dir, val_lbl_dir)
    print(f"Dataset split complete: {len(train_pairs)} train, {len(val_pairs)} val.")



def augment_yolo_dataset(
    images_dir: str,
    labels_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    num_aug_per_image: int = 2
):
    """
    Applies spatial and color transformations to images and updates YOLO bounding boxes.
    """
    # Define augmentation pipeline (bbox_params ensures coordinates update correctly)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.2, 0.2),
            rotate=(-15, 15),
            shear=(-10, 10),
            p=0.5            
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.3, clip=True))


    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [f for f in Path(images_dir).iterdir() if f.suffix.lower() in image_extensions]

    for img_file in img_files:

        lbl_file = labels_dir + "\\" + f"{img_file.stem}.txt"

        image = cv2.imread(str(img_file))
        if image is None:
            continue

        # Parse existing YOLO annotations
        bboxes, category_ids = [], []
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    bbox = [float(x) for x in parts[1:]]  # [x_center, y_center, w, h]
                    # Albumentations expects YOLO boxes as center/width/height,
                    # not [x_min, y_min, x_max, y_max].  Reject malformed or
                    # zero-area boxes before it converts them to corner format.
                    if (
                        0.0 <= bbox[0] <= 1.0 and 0.0 <= bbox[1] <= 1.0
                        and 0.0 < bbox[2] <= 1.0 and 0.0 < bbox[3] <= 1.0
                    ):
                        category_ids.append(int(float(parts[0])))
                        bboxes.append(bbox)

        # Copy original image & label to output folder
        # cv2.imwrite(str(output_images_dir + "\\" + img_file.name), image)
        # with open(lbl_file, 'w') as f:
        #     f.write(lbl_file)

        # Generate augmented variations
        for i in range(num_aug_per_image):
            try:
                transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
                aug_image = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_cls = transformed['category_ids']

                aug_img_name = f"{img_file.stem}_aug{i}{img_file.suffix}"
                aug_lbl_name = f"{img_file.stem}_aug{i}.txt"

                # Save transformed image
                cv2.imwrite(str(output_images_dir + "\\" + aug_img_name), aug_image)

                # Save updated YOLO labels
                lines = [
                    f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
                    for c, b in zip(aug_cls, aug_bboxes)
                ]
                with open(output_labels_dir + "\\" + aug_lbl_name, 'w') as f:
                    f.write("\n".join(lines))

            except ValueError as e:
                print(f"Skipped augment iteration {i} for {img_file.name}: {e}")

    print(f"Augmentation complete. Saved to '{output_images_dir}' and '{output_labels_dir}'.")

if __name__ == "__main__":
    # Example usage
    classes = {"label1": 0, "label2": 1}

    split_yolo_dataset(
        data_dir="./data/user_data/yolo_annotations")