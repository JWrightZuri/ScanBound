import numpy as np
from skimage import filters, measure, morphology, exposure
from skimage.io import imread
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.filters import median
from skimage.morphology import disk
from core.second_pass_segment import second_pass_segment
import pandas as pd


# Variable abstraction
active_directory = r'C:\Users\wrja\Desktop\STM_Data\napari-stm-annotator\my_napari_testing\\'  # Directory where images are stored
image_name = r'segment1_test_(active)20250521_PdGaA111_279a_1min_flash_0.6A_200K_area10010_z.png'

class first_pass_segment():
    def __init__(self):
        self.feature_properties, self.num_bins, self.thresholds = self.load_feature_parameters()

    def load_feature_parameters(self):
        """
        Load feature parameters from a CSV file.
        """
        csv_path = active_directory + 'object_parameters.csv'

        print(f"Searching for parameter file...")
        if not pd.io.common.file_exists(csv_path):
            print(f"CSV file not found at {csv_path}. Please ensure the file exists.")
            return None

        df = pd.read_csv(csv_path)

        labels = df['label'].tolist()
        areas = df['area'].tolist()
        aspect_ratios = df['aspect_ratio'].tolist()
        chirality = df['isChiral'].tolist()
        gnr = df['isGNR'].tolist()

        print("Labels:", labels)
        print("Areas:", areas)
        print("Aspect Ratios:", aspect_ratios)

        feature_properties = {
            label: {
                'area': area,
                'aspect_ratio': aspect_ratio,
                'isChiral': chirality,
                'isGNR': gnr
            }
            for label, area, aspect_ratio, chirality, gnr in zip(labels, areas, aspect_ratios, chirality, gnr)
        }

        num_bins = len(labels)  # Number of bins for histogram
        min_feature_size = min(areas) * 0.1  # Minimum size for removing small objects
        background_thresh = 0.7  # Threshold for background removal
        min_maxima_distance = 20  # Minimum distance between local maxima

        thresholds = {
            'min_feature_size': min_feature_size,
            'background_thresh': background_thresh,
            'min_maxima_distance': min_maxima_distance
        }

        return feature_properties, num_bins, thresholds

# def load_feature_parameters():
#     """
#     Load feature parameters from a CSV file.
#     """
#     stm_data_directory = r'C:\Users\wrja\Desktop\STM_Data\\'  # Directory where images are stored
#     csv_path = stm_data_directory + 'stm_labeled_annotations.csv'


#     print(f"Seartching for parameter file...")
#     if not pd.io.common.file_exists(csv_path):
#         print(f"CSV file not found at {csv_path}. Please ensure the file exists.")
#         return None

#     df = pd.read_csv(csv_path)

#     labels = df['label'].tolist()
#     areas = df['area'].tolist()
#     aspect_ratios = df['aspect_ratio'].tolist()
#     chirality = df['isChiral'].tolist()
#     gnr = df['isGNR'].tolist()

#     print("Labels:", labels)
#     print("Areas:", areas)
#     print("Aspect Ratios:", aspect_ratios)

#     feature_properties = {
#         label: {
#             'area': area,
#             'aspect_ratio': aspect_ratio,
#             'isChiral': chirality,
#             'isGNR': gnr
#         }
#         for label, area, aspect_ratio, chirality, gnr in zip(labels, areas, aspect_ratios, chirality, gnr)
#     }
#     # print(f"Feature properties: {feature_properties}")
#     # print(f"Feature names: {feature_properties.keys()}")
#     # print(f"Feature sizes: {[feature_properties[name]['area'] for name in feature_names]}")
#     num_bins = len(labels)  # Number of bins for histogram
#     min_feature_size = min(areas) * 0.1  # Minimum size for removing small objects
#     background_thresh = 0.7  # Threshold for background removal
#     min_maxima_distance = 20  # Minimum distance between local maxima
    
#     thresholds = {
#         'min_feature_size': min_feature_size,
#         'background_thresh': background_thresh,
#         'min_maxima_distance': min_maxima_distance
#     }

#     return feature_properties, num_bins, thresholds


# feature_properties, num_bins, thresholds = load_feature_parameters()

    # Load your STM image
    image = imread(active_directory + image_name, as_gray=True)

    img = (image - image.min()) / (image.max() - image.min())
    smoothed = gaussian_filter(img, sigma=3)

    median = median(smoothed, disk(2))

    maxima = peak_local_max(median, min_distance=thresholds['min_maxima_distance'], threshold_abs=thresholds['background_thresh'])

    # 2. (Optional) Enhance contrast5
    contrast = exposure.equalize_adapthist(median, clip_limit=0.5)

    # Remove background: keep only pixels above a brightness threshold
    foreground_mask = contrast > thresholds['background_thresh']
    contrast = contrast * foreground_mask

    # 3. Threshold to segment bright regions (molecules)
    threshold = filters.threshold_otsu(contrast)
    binary = contrast > threshold

    # 4. Remove tiny objects (noise)
    binary_opened = morphology.binary_opening(binary, disk(5))
    binary_cleaned = morphology.remove_small_objects(binary_opened, min_size=thresholds['min_feature_size'])

    # 5. Label connected components
    labeled = measure.label(binary_cleaned, connectivity=2)
    props = measure.regionprops(labeled)

    # Keep only regions whose bounding box contains at least one local maxima
    filtered_props = []
    areas = []
    aspect_ratios = []
    for region in props:
        minr, minc, maxr, maxc = region.bbox
        # Check if any maxima falls within the bounding box
        for peak in maxima:
            if minr <= peak[0] < maxr and minc <= peak[1] < maxc:
                filtered_props.append(region)
                areas.append((region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1]))
                # areas.append(region.area)
                aspect_ratios.append((region.bbox[2] - region.bbox[0]) / (region.bbox[3] - region.bbox[1]) if (region.bbox[3] - region.bbox[1]) > 0 else 0)
                break  # Only need one maxima per region
    props = filtered_props

    # print(areas)
    # print(aspect_ratios)

    # Dynamically create centroid and index lists for each label
    bin_centroids = {label: [] for label in list(feature_properties.keys()) + ['Other']}
    bin_indices = {label: [] for label in list(feature_properties.keys()) + ['Other']}


    # print(bin_centroids)
    # print(bin_indices)

    feature_labels = list(feature_properties.keys())
    feature_areas = np.array([feature_properties[l]['area'] for l in feature_labels])
    feature_aspects = np.array([feature_properties[l]['aspect_ratio'] for l in feature_labels])

    for idx, (area, aspect) in enumerate(zip(areas, aspect_ratios)):
        # Compute normalized distance in area and aspect ratio
        area_diffs = np.abs(feature_areas - area) / (feature_areas + 1e-6)
        aspect_diffs = np.abs(feature_aspects - aspect) / (feature_aspects + 1e-6) * 2
        # Weighted sum or Euclidean distance
        dists = area_diffs + aspect_diffs
        print(f"Region {idx}: Area={area}, Aspect Ratio={aspect}, Dists={dists}")
        best_idx = np.argmin(dists)
        print(best_idx)
        if dists[best_idx] < 1.0:
            best_label = feature_labels[best_idx]
            bin_centroids[best_label].append(props[idx].centroid)
            bin_indices[best_label].append(idx)
        else:
            bin_centroids['Other'].append(props[idx].centroid)
            bin_indices['Other'].append(idx)
    print(bin_centroids)
    print(bin_indices)
    chiralites = []
    vectors = []


    print(f"Detected {len(bin_centroids['CO'])} CO molecules and {len(bin_centroids['9AP'])} 9AP molecules and {len(bin_centroids['Other'])} Other molecules.")
    # 6. Extract centroids
    centroids = np.array([p.centroid for p in props])


    # 7. Plot result
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax = axes.ravel()

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(contrast, cmap='gray')
    ax[1].set_title("Contrast Enhanced")
    ax[1].axis('off')

    ax[2].imshow(binary_cleaned, cmap='gray')
    ax[2].set_title("Cleaned Binary")
    ax[2].axis('off')

    ax[3].imshow(binary_cleaned, cmap='nipy_spectral')
    if len(maxima) > 0:
        ax[3].scatter(maxima[:, 1], maxima[:, 0], color='red', s=15, marker='x', label='Local Maxima')
    ax[3].set_title("Local Maxima on Cleaned Binary")
    ax[3].axis('off')
    ax[3].legend(loc='upper right', fontsize='small')

    ax[4].imshow(image, cmap='gray')
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
            ax[4].scatter(centroids_arr[:, 1], centroids_arr[:, 0], color=colors.get(label, 'white'), s=20, label=label)
    ax[4].set_title("Centroids by Label")
    ax[4].axis('off')
    ax[4].legend(loc='upper right', fontsize='small')

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
    #         # Pass in the crop of the binary image to second_pass_segment
    #         cropped_binary = contrast[minr_p:maxr_p, minc_p:maxc_p]
    #         # cropped_image = image[minr_p:maxr_p, minc_p:maxc_p]
    #         # Optionally, save or process cropped_image as needed
    #         chirality, vector = second_pass_segment.run_second_pass(cropped_binary)
    #         if vector is not None:
    #             vector[0] += minc_p  # Adjust x coordinate
    #             vector[1] += minr_p  # Adjust y coordinate
    #         chiralites.append(chirality)
    #         vectors.append(vector)
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

    # Plot chirality and vector for each 9AP centroid
    # if len(bin3_centroids) > 0:
    #     for i, centroid in enumerate(bin3_centroids):
    #         # y, x = centroid
    #         # Draw vector if available
    #         if i < len(vectors) and vectors[i] is not None:
    #             x, y, vx, vy = vectors[i]
    #             ax[4].arrow(x, y, vx, vy, color='red', head_width=5, head_length=8, length_includes_head=True)
    #         # Annotate chirality if available
    #         if i < len(chiralites) and chiralites[i] is not None:
    #             ax[4].text(x, y, str(chiralites[i]), color='white', fontsize=10, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    plt.tight_layout()
    plt.show()



