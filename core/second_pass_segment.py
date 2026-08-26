import numpy as np
from skimage import filters
from skimage import exposure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk
from skimage.morphology import opening
from skimage.transform import rescale

def run_second_pass_segment(image, plot_results=False):
    """Run the second pass segmentation on the provided image.
    Args:
        image (ndarray): Input image for segmentation.
    Returns:
        chirality (str): Chirality of the detected lobes.
        vectors (list): List of vectors representing the relationships between lobes.
    """
    # Upscale image by a factor of 10 using anti-aliasing
    try:
        image = rescale(image, scale=10.0, anti_aliasing=True, channel_axis=None)
    except Exception as e:
        # print(f"Error during image rescaling: {e}")
        image = np.nan_to_num(image, nan=0.0)
        # print("Nan found")
        # print(np.isnan(image))
        # image = rescale(image, scale=10.0, anti_aliasing=True, channel_axis=None)
    chirality = None
    vectors = None
    # max_attempts = 10
    # attempts = 0

    # while chirality is None or vectors is None:
    chirality, vectors = find_blobs(image, plot_results=plot_results)
        # bias += 1.0
        # attempts += 1
        # if attempts >= max_attempts:
        #     print("Max attempts reached")
        #     break
    # plt.show()
    return chirality, vectors


def produce_watershed(gauss_img):
    # Threshold to get binary image
    markers_count = 0
    attempts = 0
    otsu_bias = 0.1
    end_loop = False
    while end_loop is False:
        if attempts >= 1:
            gauss_img = exposure.equalize_adapthist(gauss_img, clip_limit=0.01+attempts*0.05)
        try:
            thresh_val = filters.threshold_otsu(gauss_img) * otsu_bias
            # print(f"Otsu threshold value (with bias {otsu_bias}): {thresh_val}")
        except:
            # print("Otsu thresholding failed, using fixed threshold of 0.1")
            thresh_val = 0.1
        binary = gauss_img > thresh_val


        # Perform binary opening to clean up the binary mask
        # binary = opening(binary, disk(5))
        
        # Estimate lobe width using distance transform on binary mask
        # We'll use the median of the maximum Feret diameter (major axis length) of each region as lobe width
        props = regionprops(label(binary))
        widths = [p.major_axis_length for p in props if p.area > 10]

        # Step 2: Distance transform
        distance = ndi.distance_transform_edt(binary)
        # Step 3: Find local maxima as markers
        local_max = peak_local_max(distance, min_distance=20, num_peaks=3)
        intensities = gauss_img[tuple(local_max.T)]
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(local_max.T)] = True
        markers, _ = ndi.label(mask)

        markers_count = len(np.unique(markers))
        if markers_count == 4:
            end_loop = True
        else:
            pass
        # Step 4: Apply watershed
        labels_ws = watershed(-distance, markers=markers, mask=binary)

        otsu_bias += 0.8
        attempts += 1
        if attempts >= 5:
            print("Max attempts reached")
            end_loop = True
    return labels_ws, local_max, intensities, binary, distance

def find_blobs(image, plot_results=False):
    img_width = image.shape[0]
    img_height = image.shape[1]
    sigma1x = img_width / 18.0 
    sigma2x = sigma1x * 1.6

    sigma1y = img_height / 18.0 
    sigma2y = sigma1y * 1.6

    # Apply Laplace filter to enhance edges
    # laplace_filtered = filters.laplace(image, ksize=5)
    # gauss = np.clip(laplace_filtered, 0, None)
    gauss = filters.difference_of_gaussians(image, low_sigma=(sigma1x, sigma1y), high_sigma=(sigma2x, sigma2y))
    gauss = np.clip(gauss, 0, None)

    # gauss = exposure.adjust_log(gauss, gain=1)
    # gauss = filters.
    # gauss = exposure.rescale_intensity(gauss, in_range=(0, .99))
    if plot_results:
        plt.figure(figsize=(6, 4))
        plt.imshow(gauss, cmap='gray')
        plt.title('Difference of Gaussians')
        plt.axis('off')
    # Step 4: Apply watershed
    try:
        labels_ws, local_max, intensities, binary, distance = produce_watershed(gauss)
    except Exception as e:
        print(f"Error in watershed segmentation: {e}")
        return None, None
    # Step 5: Get centroids
    props = regionprops(labels_ws)
    centroids_local = np.array([p.centroid for p in props if p.area > 10])

    # Get area of each watershed region
    areas = np.array([p.area for p in props])
    # print("Areas of each watershed region:", areas)

    # Rank watershed regions by area (descending)
    sorted_by_area = np.argsort(areas)[::-1]
    # print("Watershed regions ranked by area (largest to smallest):", areas[sorted_by_area])
    
    if plot_results:
        print("Plotting results...")
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
            mid_idx = sorted_by_area[1]
            small_idx = sorted_by_area[2]
            vector_mid_to_small = centroids_local[small_idx] - centroids_local[mid_idx]
            # print(f"Vector from middle to smallest watershed: {vector_mid_to_small}")
            # Plot the vector from middle to smallest centroid
            # ax[3].arrow(
            #     centroids_local[mid_idx][1], centroids_local[mid_idx][0],
            #     centroids_local[small_idx][1] - centroids_local[mid_idx][1],
            #     centroids_local[small_idx][0] - centroids_local[mid_idx][0],
            #     head_width=5, head_length=8, fc='orange', ec='orange', linewidth=2, length_includes_head=True
            # )
            # ax[3].set_title(f'Watershed Segmentation\nChirality: {chirality}\nMid→Small vector shown')

    # Find which watershed region has the most interface with the smallest region
    if len(centroids_local) == 3:
        # Find label of smallest region
        smallest_idx = sorted_by_area[2]
        smallest_label = props[smallest_idx].label

        # Get mask for smallest region
        smallest_mask = labels_ws == smallest_label
        # Find neighbors: pixels at the boundary of smallest region
        dilated = dilation(smallest_mask, disk(1))
        border = dilated & ~smallest_mask
        neighbor_labels = labels_ws[border]
        # Count occurrences of each neighbor label (excluding background and itself)
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        neighbor_counts = {u: c for u, c in zip(unique, counts) if u != 0 and u != smallest_label}
        # print("Neighbor counts:", neighbor_counts)
        if neighbor_counts:
            # Find neighbor with most interface
            most_interface_label = max(neighbor_counts, key=neighbor_counts.get)
            most_interface_idx = most_interface_label - 1
            # Find centroid of that region
            try:
                neighbor_centroid = [p.centroid for p in props if p.label == most_interface_label][0]
            except IndexError:
                # print(f"Error: No centroid found for label {most_interface_label}")
                return None, None
            # Draw a line between smallest centroid and neighbor centroid
            y0, x0 = neighbor_centroid
            y1, x1 = props[smallest_idx].centroid
            vector_interface = np.array([x0, y0, x1, y1]) # Largest to smallest lobe
            vector_interface = vector_interface / 10
            if plot_results:
                ax[3].arrow(x0, y0, x1 - x0, y1 - y0, color='red', linewidth=2)
                ax[3].text((x0 + x1) / 2, (y0 + y1) / 2, "Max interface", color='red', fontsize=8, ha='center')
                plt.show()
            # print(f"Region with most interface to smallest: label {most_interface_label}")
            
            if most_interface_idx is not None:
                # Find the index in props that is not smallest_idx or most_interface_idx
                final_idx = next(i for i in range(len(props)) if i != smallest_idx and i != most_interface_idx)
                sorted_array = np.array([final_idx, most_interface_idx, smallest_idx])
            else:
                sorted_array = sorted_by_area
            centroids_sorted = centroids_local[sorted_array]
            # Assign: a - smallest, b - middle, c - largest
            # Compute chirality: signed area of triangle
            a, b, c = centroids_sorted
            signed_area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) -
                                (b[1] - a[1]) * (c[0] - a[0]))
            chirality = "S" if signed_area > 0 else "R"
            return chirality, vector_interface
        else:
            # print("No neighboring regions found for smallest watershed.")
            return None, None

    else:
        # print("Not enough regions to find neighbors.")
        return None, None

if __name__ == "__main__":
    # Load a sample image
    image = imread(r"C:\Users\wrja\Desktop\ScanBound\data\user_data\L240427_171746_region10001_9AP.png", as_gray=True)
    chirality, vectors = run_second_pass_segment(image, plot_results=True)
    # print(f"Chirality: {chirality}, Vectors: {vectors}")