import numpy as np
from skimage import filters, measure, morphology, exposure
from skimage.io import imread
# from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.morphology import disk
import pandas as pd
import os
# from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

from scipy.ndimage import gaussian_filter, median_filter, gaussian_laplace

def run_first_pass_segment(image, vacancy_detection=False, feature_properties=None, num_bins=None, thresholds=None, gamma=None, threshold=None, sigma=None, plot_results=False):
    img = (image - image.min()) / (image.max() - image.min())
    
    if sigma is None:
        init_sigma = img.shape[0] / 150.0
    else:
        init_sigma = sigma

    img = gaussian_laplace(img, sigma=init_sigma)
    img = 1 - img

    img = exposure.adjust_gamma(img, gamma=gamma if gamma is not None else 20)
    # img = exposure.adjust_log(img, gain=20)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # thresh = filters.threshold_otsu(img)+.1
    thresh = filters.threshold_multiotsu(img)[1]
    print(f"Multi Otsu thresholds: {thresh}")
    if threshold is not None:
        thresh = threshold
    img_binary = img > thresh 
    # background_black_percentage = np.sum(img_binary == 0) / img_binary.size * 100
    # print(f"Percentage of background that is black (0): {background_black_percentage:.2f}%")
    maxima = peak_local_max(img,
                            min_distance=10, 
                            threshold_abs=thresh*.01)

    vacancies = []
    if vacancy_detection is True:
        try:
            # 1. Detect local maxima
            image_inv = 1.0 - image
            vacancies = peak_local_max(image_inv, min_distance=thresholds['min_maxima_distance'], threshold_abs=0.90)
            # print(f"Found {len(vacancies)} local maxima in inverted image.")
        except Exception as e:
            print(f"Error detecting vacancies: {e}")

    # 5. Label connected components
    labeled = measure.label(img_binary, connectivity=2)
    props = measure.regionprops(labeled)

    # Keep only regions whose bounding box contains at least one local maxima
    # and whose area is larger than min_feature_size
    # min_feature_size = min([feature['area'] for feature in feature_properties.values()])
    min_feature_size = thresholds['min_feature_size'] if thresholds and 'min_feature_size' in thresholds else 10
    # print(f"Using min_feature_size: {min_feature_size}")
    filtered_props = []
    areas = []
    aspect_ratios = []
    for region in props:
        minr, minc, maxr, maxc = region.bbox
        region_area = (maxr - minr) * (maxc - minc)
        if region_area <= min_feature_size:
            continue
        # Check if any maxima falls within the bounding box
        # for peak in maxima:
        #     if minr <= peak[0] < maxr and minc <= peak[1] < maxc:
        filtered_props.append(region)
        areas.append(region_area)
        aspect_ratios.append((maxr - minr) / (maxc - minc) if (maxc - minc) > 0 else 0)
        # break  # Only need one maxima per region
    props = filtered_props

    # print(areas)
    # print(aspect_ratios)

    # Dynamically create centroid and index lists for each label
    bin_centroids = {label: [] for label in list(feature_properties.keys()) + ['Empty Label']}
    bin_indices = {label: [] for label in list(feature_properties.keys()) + ['Empty Label']}


    # print(bin_centroids)
    # print(bin_indices)

    feature_labels = list(feature_properties.keys())
    feature_areas = np.array([feature_properties[l]['area_px'] for l in feature_labels])
    feature_aspects = np.array([feature_properties[l]['aspect_ratio'] for l in feature_labels])

    for idx, (area, aspect) in enumerate(zip(areas, aspect_ratios)):
        # Compute normalized distance in area and aspect ratio
        area_diffs = np.abs(feature_areas - area) / (feature_areas + 1e-6)
        aspect_diffs = np.abs(feature_aspects - aspect) / (feature_aspects + 1e-6) #* 2
        # Weighted sum or Euclidean distance
        dists = area_diffs + aspect_diffs
        # print(f"Region {idx}: Area={area}, Aspect Ratio={aspect}, Dists={dists}")
        best_idx = np.argmin(dists)
        # print(best_idx)
        if dists[best_idx] < 1.0:
            best_label = feature_labels[best_idx]
            if  best_label == '9AP (S)' or best_label == '9AP (R)' or best_label == '9AP (None)':
                best_label = '9AP'
            bin_centroids[best_label].append(props[idx].centroid)
            bin_indices[best_label].append(idx)
        else:
            bin_centroids['Empty Label'].append(props[idx].centroid)
            bin_indices['Empty Label'].append(idx)
    # print(bin_centroids)
    # print(bin_indices)

    # print(f"Detected {len(bin_centroids['CO'])} CO molecules and {len(bin_centroids['9AP'])} 9AP molecules.")
    # 6. Extract centroids
    if plot_results is True:
        centroids = np.array([p.centroid for p in props])
        # 7. Plot result
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax = axes.ravel()

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        ax[1].imshow(img, cmap='gray')
        ax[1].set_title("Laplacian of Gaussian")
        ax[1].axis('off')


        ax[2].imshow(img_binary, cmap='nipy_spectral')
        if len(maxima) > 0:
            ax[2].scatter(maxima[:, 1], maxima[:, 0], color='red', s=15, marker='x', label='Local Maxima')
            if vacancy_detection is True:
                ax[2].scatter(vacancies[:, 1], vacancies[:, 0], color='blue', s=15, marker='x', label='Local Maxima (Inverted)')
        ax[2].set_title("Local Maxima on Cleaned Binary")
        ax[2].axis('off')
        ax[2].legend(loc='upper right', fontsize='small')

        ax[3].imshow(image, cmap='gray')
        # Plot each centroid with its corresponding label and color
        colors = {
            'CO': 'cyan',
            '9AP': 'magenta',
            'Adsorbate': 'yellow',
            'Other': 'white'
        }
        for label, centroids in bin_centroids.items():
            if len(centroids) > 0:
                centroids_arr = np.array(centroids)
                ax[3].scatter(centroids_arr[:, 1], centroids_arr[:, 0], color=colors.get(label, 'white'), s=20, label=label)
        ax[3].set_title("Centroids by Label")
        ax[3].axis('off')
        ax[3].legend(loc='upper right', fontsize='small')

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111)
        # for region in props:
        #     # Draw a rectangle around segmented region
        #     region_index = props.index(region)
        #     if region_index in bin_indices_1:
        #         edgecolor = 'blue'
        #         bin1_centroids.append(region.centroid)
        #     elif region_index in bin_indices_3:
        #         edgecolor = 'red'
        #         bin3_centroids.append(region.centroid)
        #         # Crop the image for the current region
        #         print(region.bbox)
        #         minr, minc, maxr, maxc = region.bbox
        #         pad = 20
        #         minr_p = max(minr - pad, 0)
        #         minc_p = max(minc - pad, 0)
        #         maxr_p = min(maxr + pad, image.shape[0])
        #         maxc_p = min(maxc + pad, image.shape[1])
        #     else:
        #         edgecolor = 'green'
        #         bin2_centroids.append(region.centroid)

        #     minr, minc, maxr, maxc = region.bbox
        #     rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
        #                          fill=False, edgecolor='lime', linewidth=2)
        #     ax.add_patch(rect)

        # ax.set_title("Labeled Regions")
        # ax.axis('off')
        # ax.imshow(image, cmap='gray')

        plt.tight_layout()
        plt.show()

    return bin_indices, props, vacancies, img 

if __name__ == "__main__":
    # Example usage
    # image_path = r"C:\Users\wrja\Desktop\STM_Data\Orito_Reaction\QPlus\20250803_PdGaA111_279Prod_2min_35C_98K\jpg_conversion\0002_Z.jpg"
    # image_path = r"C:\Users\wrja\Desktop\STM_Data\Orito_Reaction\PdGaA(111)_Pd1_CX3-2\20250526_PdGaA111_279a_1min_flash_0.6A_200K_298K_433K_500K\saved_output\area10026_Z.jpg"
    # image_path = r"C:\Users\wrja\Desktop\STM_Data\Orito_Reaction\PdGaA(111)_Pd1_CX3-2\20250526_PdGaA111_279a_1min_flash_0.6A_200K_298K_433K_500K\saved_output\area10009_Z.jpg"
    image_path = r"C:\Users\wrja\Desktop\STM_Data\Orito_Reaction\PdGaA(111)_Pd1_CX3-2\20250526_PdGaA111_279a_1min_flash_0.6A_200K_298K_433K_500K\saved_output\area10016_Z.jpg"

    image = imread(image_path, as_gray=True)
    feature_properties = {
        'CO': {'area': 200, 'aspect_ratio': 1.0},
        '9AP': {'area': 10000, 'aspect_ratio': 1.2},
        'Adsorbate': {'area': 500, 'aspect_ratio': 1.5}
    }
    num_bins = None
    thresholds = {
        'background_thresh': 0.5,
        'min_maxima_distance': 30
    }
    
    run_first_pass_segment(image, True, feature_properties, num_bins, thresholds, plot_results=True)

