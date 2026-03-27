import numpy as np
from skimage import filters
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk
from skimage.morphology import opening

class second_pass_segment():
    def __init__(self, image):
        self.image = image

    def run(self):
        """
        Run the second pass segmentation on the provided image.
        """
        chirality, vectors = find_blobs(self.image)
        return chirality, vectors

def find_best_sigma(img):
    img_width = img.shape[1]
    print(f"Image width (pixels): {img_width}")
    sigma1 = img_width / 18.0
    sigma2 = sigma1 * 1.6
    return sigma1, sigma2

def produce_watershed(gauss_im):
    # Threshold to get binary image
    thresh_val = filters.threshold_otsu(gauss_im)
    binary = gauss_im > thresh_val

    # Perform binary opening to clean up the binary mask
    # binary = opening(binary, disk(5))
    
    # Estimate lobe width using distance transform on binary mask
    # We'll use the median of the maximum Feret diameter (major axis length) of each region as lobe width
    props = regionprops(label(binary))
    widths = [p.major_axis_length for p in props if p.area > 10]
    # print(np.median(widths))
    med_width=np.median(widths)

    # Step 2: Distance transform
    distance = ndi.distance_transform_edt(binary)
    # Step 3: Find local maxima as markers
    local_max = peak_local_max(distance, min_distance=20, num_peaks=3)
    intensities = gauss_im[tuple(local_max.T)]
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(local_max.T)] = True
    markers, _ = ndi.label(mask)

    # Step 4: Apply watershed
    labels_ws = watershed(-distance, markers, mask=binary)
    return labels_ws, local_max, intensities, binary, distance

def find_blobs(image):

    sigma1, sigma2 = find_best_sigma(image)
    best_sigma = 5.0
    gauss = filters.difference_of_gaussians(image, low_sigma=sigma1, high_sigma=sigma2)
    
    # Step 4: Apply watershed
    labels_ws, local_max, intensities, binary, distance = produce_watershed(gauss, )

    # Step 5: Get centroids
    props = regionprops(labels_ws)
    centroids_local = np.array([p.centroid for p in props if p.area > 10])

    # Get area of each watershed region
    areas = np.array([p.area for p in props])
    print("Areas of each watershed region:", areas)

    # Rank watershed regions by area (descending)
    sorted_indices = np.argsort(areas)[::-1]
    print("Watershed regions ranked by area (largest to smallest):", areas[sorted_indices])
    
    # Sort centroids by watershed area (descending)
    # if len(centroids_local) == 3:
    #     sorted_idx = np.argsort(areas)[::-1]
    #     centroids_sorted = centroids_local[sorted_idx]
    #     # Assign: 0 - smallest, 1 - middle, 2 - largest
    #     # Vectors: 2->1, 1->0, 0->2 (small->mid, mid->large, large->small)
    #     pts = centroids_sorted  # [large, mid, small]
    #     vectors = [pts[0] - pts[1], pts[1] - pts[2], pts[2] - pts[0]]
    #    # Compute chirality: signed area of triangle
    #     a, b, c = pts
    #     signed_area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) -
    #                          (b[1] - a[1]) * (c[0] - a[0]))
    #     chirality = "CCW (S)" if signed_area > 0 else "CW (R)"
    #     # print(f"Vector-based chirality: {chirality}")
    # else:
    #     print("Chirality vector drawing requires exactly 3 centroids.")

    # Convert to global coordinates
    # coords_global = centroids_local + np.array([y0, x0])
    fig, axes = plt.subplots(ncols=4, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(binary, cmap='gray')
    ax[1].set_title('Binary Mask')
    ax[2].imshow(-distance, cmap='gray')
    ax[2].set_title('Distance Transform')
    ax[3].imshow(labels_ws, cmap='nipy_spectral')
    ax[3].set_title('Watershed Segmentation')
    # Draw vectors
    colors = ['cyan', 'magenta', 'lime']

    # Add area labels to watershed plot
    for i, prop in enumerate(props):
        if prop.area > 10:
            y, x = prop.centroid
            ax[3].text(x, y, f"{prop.area:.0f}", color='white', fontsize=8, ha='center', va='center')
    # Add intensity labels to watershed plot (at centroid positions)
    for i, (centroid, intensity) in enumerate(zip(centroids_local, intensities)):
        y, x = centroid
        ax[3].text(x, y + 10, f"{intensity:.4f}", color='yellow', fontsize=8, ha='center', va='center')

    print(f"Detected {len(centroids_local)} lobes using watershed in box.")
    # classify_chirality(centroids_local)
    if len(centroids_local) == 3:
        # sorted_indices: [largest, mid, smallest]
        mid_idx = sorted_indices[1]
        small_idx = sorted_indices[2]
        vector_mid_to_small = centroids_local[small_idx] - centroids_local[mid_idx]
        # print(f"Vector from middle to smallest watershed: {vector_mid_to_small}")
        # Plot the vector from middle to smallest centroid
        ax[3].arrow(
            centroids_local[mid_idx][1], centroids_local[mid_idx][0],
            centroids_local[small_idx][1] - centroids_local[mid_idx][1],
            centroids_local[small_idx][0] - centroids_local[mid_idx][0],
            head_width=5, head_length=8, fc='orange', ec='orange', linewidth=2, length_includes_head=True
        )
        # ax[3].set_title(f'Watershed Segmentation\nChirality: {chirality}\nMid→Small vector shown')

    # Find which watershed region has the most interface with the smallest region
    if len(centroids_local) == 3:
        # Find label of smallest region
        smallest_idx = sorted_indices[2]
        smallest_label = props[smallest_idx].label
        # Find boundaries of all regions
        boundaries = find_boundaries(labels_ws, mode='thick')
        # Get mask for smallest region
        smallest_mask = labels_ws == smallest_label
        # Find neighbors: pixels at the boundary of smallest region
        dilated = dilation(smallest_mask, disk(1))
        border = dilated & ~smallest_mask
        neighbor_labels = labels_ws[border]
        # Count occurrences of each neighbor label (excluding background and itself)
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        neighbor_counts = {u: c for u, c in zip(unique, counts) if u != 0 and u != smallest_label}
        if neighbor_counts:
            # Find neighbor with most interface
            most_interface_label = max(neighbor_counts, key=neighbor_counts.get)
            # Find centroid of that region
            neighbor_centroid = [p.centroid for p in props if p.label == most_interface_label][0]
            # Draw a line between smallest centroid and neighbor centroid
            y0, x0 = props[smallest_idx].centroid
            y1, x1 = neighbor_centroid
            vector_interface = np.array([x0, y0, x1 - x0, y1 - y0]) # x0, y0, dx, dy

            ax[3].plot([x0, x1], [y0, y1], color='red', linewidth=2, linestyle='--')
            ax[3].text((x0 + x1) / 2, (y0 + y1) / 2, "Max interface", color='red', fontsize=8, ha='center')
            # print(f"Region with most interface to smallest: label {most_interface_label}")
            chirality = "S"
            return chirality, vector_interface
        else:
            print("No neighboring regions found for smallest watershed.")
            return None, None

    else:
        print("Not enough regions to find neighbors.")
        return None, None

def run_second_pass(region):
    """    Run the second pass segmentation on a specific image.   """
    # Load image
    # img = imread(filename, as_gray=True)
    # img = (img - img.min()) / (img.max() - img.min())  # normalize
    chirality, vectors = find_blobs(region)
    # plt.show()

    return chirality, vectors