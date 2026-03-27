import numpy as np
from skimage.io import imread
from skimage.filters import threshold_local, gaussian, sobel
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from skimage.morphology import erosion, disk, dilation
from skimage import exposure
from scipy import ndimage

def terrace_safe_flatten(img):
    # 1. Row-wise median subtraction to align terrace heights
    img_aligned = img - np.median(img, axis=1, keepdims=True)
    
    # 2. Apply a large-scale Gaussian high-pass filter
    # This removes long-range gradients (tilts) while keeping local GNR height
    background = ndimage.gaussian_filter(img_aligned, sigma=50) # Large sigma for gradient
    img_flattened = img_aligned - background
    
    # 3. Re-normalize to 0-1 range
    img_flattened = (img_flattened - img_flattened.min()) / (img_flattened.max() - img_flattened.min())
    return img_flattened

def mask_step_edges(img, binary_mask):
    # Detect sharp gradients
    edges = sobel(img)
    
    # Threshold the edges to find just the terrace steps
    # Steps are usually the highest gradient features in the image
    step_mask = edges > (np.percentile(edges, 85)) 
    
    # Dilate the step mask slightly to ensure overlap
    step_mask_dilated = dilation(step_mask, disk(2))
    
    # Remove these areas from your GNR detection
    clean_gnr_mask = binary_mask & ~step_mask_dilated
    return clean_gnr_mask



def run_GNR_segment(image, px_scale=None, sigma=None, block_size= None, num_bins=None, thresholds=None, plot_results=False):
    img = (image - image.min()) / (image.max() - image.min())

    img = terrace_safe_flatten(img) # Required when multiple terraces with gradients are present.

    # Apply Gaussian smoothing
    if sigma is None:
        sigma = img.shape[0] / 300.0
    img_smooth = gaussian(img, sigma=sigma)

    # Remove the dark background with contrast stretching 
    # img_smooth = exposure.rescale_intensity(img_smooth, in_range=(0.45, 1.0), out_range=(0, 1)) # Required when Au111 herringbones are present

    # Adaptive thresholding
    if block_size is None:
        block_size = int(img.shape[0] / 20) | 1  # Ensure block_size is odd

    local_thresh = threshold_local(img_smooth, block_size, offset=0, method='gaussian')

    img_binary = img_smooth > local_thresh

    # Remove small objects
    # min_size = thresholds['min_feature_size'] if thresholds and 'min_feature_size' in thresholds else 20
    min_size = 100 # temp
    binary_clean = remove_small_objects(img_binary, min_size=min_size)
    
    # Perform morphological erosion to remove thin connections / smooth edges
    selem = disk(1.5)  # adjust radius (1-3) as needed
    binary_clean = erosion(binary_clean.astype(bool), selem)
    
    # binary_clean = mask_step_edges(img_smooth, binary_clean)

    # Skeletonization
    img_skeleton = skeletonize(binary_clean)

    # Distance transform
    distance_map = distance_transform_edt(binary_clean)

    # Build adjacency graph of skeleton pixels
    skeleton_bool = img_skeleton.astype(bool)
    coords = set(map(tuple, np.column_stack(np.nonzero(skeleton_bool))))
    neighbors8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    adj = {}
    for y, x in coords:
        nbrs = []
        for dy, dx in neighbors8:
            ny, nx = y + dy, x + dx
            if (ny, nx) in coords:
                nbrs.append((ny, nx))
        adj[(y, x)] = nbrs

    # Identify nodes (endpoints and branchpoints)
    nodes = {p for p, nbrs in adj.items() if len(nbrs) != 2}

    # Walk edges to extract segments (avoid duplicating edges)
    visited_edges = set()
    segments = []

    def edge_id(a, b):
        return frozenset((a, b))

    # Walk from each node along each outgoing edge
    for node in nodes:
        for nbr in adj[node]:
            eid = edge_id(node, nbr)
            if eid in visited_edges:
                continue
            path = [node]
            prev = node
            cur = nbr
            while True:
                path.append(cur)
                visited_edges.add(edge_id(prev, cur))
                if cur in nodes:
                    break
                # move forward: choose neighbor that's not the previous pixel
                nexts = [n for n in adj[cur] if n != prev]
                if not nexts:
                    break
                prev, cur = cur, nexts[0]
            if len(path) > 3:
                segments.append(path)

    # Handle pure cycles (no nodes) by finding any remaining unvisited edge
    for p in list(adj.keys()):
        for nbr in adj[p]:
            eid = edge_id(p, nbr)
            if eid in visited_edges:
                continue
            # start a cycle traversal
            start = p
            prev = p
            cur = nbr
            path = [start]
            while True:
                path.append(cur)
                visited_edges.add(edge_id(prev, cur))
                if cur == start:
                    break
                nexts = [n for n in adj[cur] if n != prev]
                if not nexts:
                    break
                prev, cur = cur, nexts[0]
            if len(path) > 3:
                segments.append(path)

    # Create a colored overlay image where each segment gets a distinct color
    h, w = skeleton_bool.shape
    seg_img = np.zeros((h, w, 3), dtype=float)
    rng = np.random.RandomState(0)
    colors = rng.rand(max(1, len(segments)), 3)
    for i, seg in enumerate(segments):
        col = colors[i % len(colors)]
        for (y, x) in seg:
            seg_img[y, x] = col

    # Overlay segments on grayscale image for plotting
    bg = np.stack((img, img, img), axis=2)
    overlay = bg.copy()
    mask = (seg_img.sum(axis=2) > 0)
    alpha = 0.7
    overlay[mask] = (1 - alpha) * bg[mask] + alpha * seg_img[mask]

    # Compute Euclidean length of each segment (sum of distances between consecutive skeleton pixels)
    seg_lengths = []
    for seg in segments:
        if len(seg) < 30:
            # seg_lengths.append(0.0)
            continue
        pts = np.array(seg)  # (y, x)
        diffs = np.diff(pts, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        seg_lengths.append(dists.sum())

    if px_scale is None:
        px_scale = 0.1  # nm per pixel
    seg_lengths_nm = [l * px_scale for l in seg_lengths]
    avg_length = float(np.mean(seg_lengths_nm)) if seg_lengths_nm else 0.0

    print(f"Average segment length: {avg_length:.2f} nm")


    if plot_results:
        # Plot contours on the original image
        fig, axs = plt.subplots(1, 4, figsize=(16, 8))


        # # Optionally, visualize pinches
        axs[0].imshow(img_smooth, cmap='gray')
        axs[0].set_title("Original Image")
        axs[0].axis('off')


        axs[1].imshow(distance_map, cmap='gray')
        axs[1].set_title("Distance Map")
        axs[1].axis('off')


        # Masked image
        axs[2].imshow(binary_clean, cmap='gray')
        axs[2].set_title("Masked Image")
        axs[2].axis('off')

        axs[3].imshow(img_skeleton, cmap='Reds')
        axs[3].set_title("Skeletonized Mask")
        axs[3].axis('off')

        # Replace skeleton panel with colored segments overlay
        axs[3].clear()
        axs[3].imshow(overlay)
        axs[3].set_title(f"Skeleton segments ({len(segments)} segments)")
        axs[3].axis('off')
        
        plt.show()

    return binary_clean, img_skeleton, distance_map, len(segments)


if __name__ == "__main__":
    # Example usage
    # image_path = r"C:\Users\wrja\Desktop\STM_Data\GNR_files\2025-05-14\saved_output\20250506_AuMica_125j_393b_38594_38612_38613_20250514_area1_0013_Z.jpg"  # Replace with your image path
    # image_path = r"C:\Users\wrja\Desktop\STM_Data\GNR_files\2025-05-14\saved_output\20250506_AuMica_125j_393b_38594_38612_38613_20250514_area1_0001_Z.jpg"
    image_path = r"C:\Users\wrja\Desktop\STM_Data\GNR_files\S5_350.7depo_524a_39370_39372_0001_crop.png"
    # image_path = r"C:\Users\wrja\Desktop\STM_Data\GNR_files\S6_350.10depo_524a_39379_39380_0007.png"
    
    image = imread(image_path, as_gray=True)
    binary_mask, skeleton, distance_map, num_segments = run_GNR_segment(image, px_scale=0.1, plot_results=True)