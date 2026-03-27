import numpy as np
from skimage.io import imread
from skimage.filters import threshold_local, gaussian, threshold_otsu, threshold_sauvola, threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from skimage.morphology import erosion, disk
from skimage.filters import try_all_threshold

# Load image
image = imread(r'C:\Users\wrja\Desktop\STM_Data\GNR_files\2025-05-14\saved_output\20250506_AuMica_125j_393b_38594_38612_38613_20250514_area1_0001_Z.jpg', as_gray=True)
img = (image - image.min()) / (image.max() - image.min())

# Preprocessing
img_blur = gaussian(img, sigma=2)

# Adaptive thresholding
block_size = 11
thresh = threshold_local(img_blur, block_size=block_size)
# thresh = threshold_otsu(img_blur)
# thresh = threshold_sauvola(img_blur, window_size=block_size, k=0.05, r=0.5)
# thresh = threshold_triangle(img_blur, nbins=128)
# try_all_threshold(img_blur, figsize=(10, 8))
# plt.show()
binary = img_blur > thresh

# Remove small junk
binary_clean = remove_small_objects(binary, min_size=100)

# Perform morphological erosion to remove thin connections / smooth edges
selem = disk(2.5)  # adjust radius (1-3) as needed
binary_clean = erosion(binary_clean.astype(bool), selem)


# # Label and analyze
# labeled = label(binary_clean)
# regions = regionprops(labeled)



# Detect "pinches" (narrow constrictions) in the mask
# We'll use skeletonization and look for points with high curvature or small width

# Compute distance transform (distance to background)
dist = distance_transform_edt(binary_clean)

# # Filter regions to select long rectangular ribbons
# ribbon_mask = np.zeros_like(binary_clean, dtype=bool)
# ribbon_props = []
# Skeletonize the ribbon mask
skeleton = skeletonize(binary_clean)

# Build adjacency graph of skeleton pixels
skeleton_bool = skeleton.astype(bool)
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

pixel_scale = 0.0992063492063491  # nm per pixel
seg_lengths_nm = [l * pixel_scale for l in seg_lengths]
avg_length = float(np.mean(seg_lengths_nm)) if seg_lengths_nm else 0.0

print(f"Average segment length: {avg_length:.2f} nm")

# --- Compute segment orientations and overall misorientation ---

orientations_deg = []  # orientation in degrees in [0, 180)
for seg in segments:
    if len(seg) < 2:
        orientations_deg.append(np.nan)
        continue
    pts = np.array(seg)  # (y, x)
    # convert to (x, y) for standard Cartesian orientation
    pts_xy = pts[:, ::-1].astype(float)
    # center and run PCA (SVD) to get principal direction
    centered = pts_xy - pts_xy.mean(axis=0)
    if centered.shape[0] < 2 or np.allclose(centered, 0):
        orientations_deg.append(np.nan)
        continue
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]  # principal axis (x, y)
    angle_rad = np.arctan2(direction[1], direction[0])  # [-pi, pi]
    angle_deg = np.degrees(angle_rad) % 180.0  # map to [0,180)
    orientations_deg.append(angle_deg)

angles = np.array(orientations_deg)
valid = np.isfinite(angles)
n_segments = int(valid.sum())

if n_segments == 0:
    print("No valid segments for orientation analysis.")
else:
    # Circular statistics with 180-degree symmetry: double angles, compute mean, then halve
    doubled_rad = np.deg2rad(angles[valid] * 2.0)
    mean_sin = np.mean(np.sin(doubled_rad))
    mean_cos = np.mean(np.cos(doubled_rad))
    mean_angle2 = np.arctan2(mean_sin, mean_cos)  # mean of doubled angles
    mean_orientation_deg = (np.degrees(mean_angle2) / 2.0) % 180.0

    # Resultant length R (for doubled angles)
    R = np.hypot(mean_cos, mean_sin)
    # circular std for doubled angles: sqrt(-2 ln R); convert back by dividing by 2
    if R > 0:
        circ_std_deg = (np.degrees(np.sqrt(max(0.0, -2.0 * np.log(R)))) / 2.0)
    else:
        circ_std_deg = np.nan

    # Compute absolute angular deviations (minimal angle considering 180deg symmetry)
    # Map differences into [-90, 90] then take absolute
    diffs = (angles[valid] - mean_orientation_deg + 90.0) % 180.0 - 90.0
    abs_devs = np.abs(diffs)
    mean_abs_dev = float(np.mean(abs_devs))
    max_abs_dev = float(np.max(abs_devs))

    print(f"Segments used for orientation: {n_segments}")
    print(f"Mean orientation: {mean_orientation_deg:.2f}° (0-180°)")
    print(f"Circular std (deg, 180° symmetry): {circ_std_deg:.2f}°")
    print(f"Mean absolute misorientation: {mean_abs_dev:.2f}°; Max: {max_abs_dev:.2f}°")

    # Compute linearity metrics for each segment:
    # - path_linearity = end-to-end distance / skeleton path length (1.0 = perfectly straight)
    # - pca_linearity  = fraction of variance explained by first principal component (0.5-1.0; 1.0 = perfectly straight)

    path_linearity = []
    pca_linearity = []

    for seg, path_len in zip(segments, seg_lengths):
        if len(seg) < 2 or path_len <= 0:
            path_linearity.append(np.nan)
            pca_linearity.append(np.nan)
            continue

        pts = np.array(seg)[:, ::-1].astype(float)  # convert (y,x) -> (x,y)
        end_to_end = np.hypot(*(pts[-1] - pts[0]))
        path_linearity.append(end_to_end / path_len)

        centered = pts - pts.mean(axis=0)
        if centered.shape[0] < 2 or np.allclose(centered, 0):
            pca_linearity.append(np.nan)
        else:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            var1 = s[0] ** 2
            var2 = s[1] ** 2 if s.size > 1 else 0.0
            pca_linearity.append(var1 / (var1 + var2) if (var1 + var2) > 0 else np.nan)

    # Summary statistics
    path_linearity = np.array(path_linearity)
    pca_linearity = np.array(pca_linearity)
    valid_path = np.isfinite(path_linearity)
    valid_pca = np.isfinite(pca_linearity)

    if valid_path.any():
        print(f"Average path linearity (end-to-end / path): {np.nanmean(path_linearity[valid_path]):.3f} ± {np.nanstd(path_linearity[valid_path]):.3f}")
    else:
        print("No valid segments for path linearity.")

    if valid_pca.any():
        print(f"Average PCA linearity (var explained by PC1): {np.nanmean(pca_linearity[valid_pca]):.3f} ± {np.nanstd(pca_linearity[valid_pca]):.3f}")
    else:
        print("No valid segments for PCA linearity.")


# Plot contours on the original image
fig, axs = plt.subplots(1, 3, figsize=(16, 8))


# # Optionally, visualize pinches
axs[0].imshow(img)
axs[0].set_title("Original Image")
axs[0].axis('off')


# Masked image
axs[1].imshow(binary_clean, cmap='gray')
axs[1].set_title("Masked Image")
axs[1].axis('off')

axs[2].imshow(skeleton, cmap='Reds')
axs[2].set_title("Skeletonized Mask")
axs[2].axis('off')

# Replace skeleton panel with colored segments overlay
axs[2].clear()
axs[2].imshow(overlay)
axs[2].set_title(f"Skeleton segments ({len(segments)} segments)")
axs[2].axis('off')



plt.tight_layout()
plt.show()