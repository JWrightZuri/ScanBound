import cv2
import numpy as np
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import opening, closing, square, remove_small_objects
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def segment_gnrs(image_path: str, min_size: int = 100):
    """
    Performs instance segmentation on an STM scan to isolate and mask each GNR.
    """
    # 1. Load image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # 2. Normalize and smooth image to suppress background wave patterns
    normalized = (img - img.min()) / (img.max() - img.min())
    smoothed = gaussian(normalized, sigma=1.5)

    # 3. Thresholding to isolate bright GNR structures
    thresh = threshold_otsu(smoothed)
    binary = smoothed > (thresh * 1.15)  # Slightly raise threshold to avoid background artifacts

    # 4. Morphological cleaning to remove noise and bridge small gaps
    cleaned = opening(binary, square(3))
    cleaned = closing(cleaned, square(3))
    
    # Remove small speckles/noise below pixel threshold
    filtered_mask = remove_small_objects(cleaned, min_size=min_size)

    # 5. Connected Component Analysis for Instance Segmentation
    labeled_instances = label(filtered_mask)
    num_instances = labeled_instances.max()

    print(f"Detected {num_instances} distinct GNR instances.")

    # 6. Extract individual masks and bounding properties
    instances = []
    for region in regionprops(labeled_instances):
        # Create an individual binary mask for each GNR instance
        instance_mask = (labeled_instances == region.label).astype(np.uint8) * 255
        instances.append({
            'id': region.label,
            'bbox': region.bbox,  # (min_row, min_col, max_row, max_col)
            'area': region.area,
            'mask': instance_mask
        })

    # 7. Visualize instances
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.imshow(labeled_instances, cmap='nipy_spectral', alpha=0.5)
    plt.title(f"GNR Instance Segmentation ({num_instances} instances found)")
    plt.axis('off')
    plt.show()

    return labeled_instances, instances

if __name__ == "__main__":
    # Process the provided STM scan image
    labeled_mask, gnr_instances = segment_gnrs(r"C:\Users\wrja\Desktop\STM_Data\GNR_files\20260716_AuMica_694a_44974_44980_44981\saved_output\0081_Z.jpg", min_size=150)