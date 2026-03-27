import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import data, filters, exposure, color, measure
from skimage.io import imread
from skimage.morphology import disk, opening

def flatten_terraces(img):
    # Rough segmentation (Otsu or manual threshold)
    thresh = filters.threshold_otsu(img)
    labels = measure.label(img > thresh)

    flat_img = np.zeros_like(img)

    for region in measure.regionprops(labels):
        coords = region.coords
        rr, cc = coords[:,0], coords[:,1]

        # Fit plane only to this terrace
        A = np.c_[cc, rr, np.ones(len(rr))]
        C, _, _, _ = np.linalg.lstsq(A, img[rr, cc], rcond=None)

        # Subtract local plane
        plane = C[0]*cc + C[1]*rr + C[2]
        flat_img[rr, cc] = img[rr, cc] - plane

    return flat_img
# Load sample image
# image = data.camera()
# image = imread(r"C:\Users\wrja\Desktop\stm_analysis_project\data\user_data\L240427_174509_region10001.jpg", as_gray=True)
image = imread(r"C:\Users\wrja\Desktop\STM_Data\Orito_Reaction\QPlus\20250803_PdGaA111_279Prod_2min_35C_98K\jpg_conversion\0002_Z.jpg", as_gray=True)

# image = imread(r"C:\Users\wrja\Desktop\stm_analysis_project\data\user_data\L240427_171746_region10001.jpg", as_gray=True)
# Initial parameters
init_sigma = 2.0
init_contrast = 1.0
init_thresh = 50

# Processing functions
def process_image(img, sigma, contrast, thresh):
    # Normalize image
    img = (img - img.min()) / (img.max() - img.min())
    # img = flatten_terraces(img)  # Flatten terraces if needed
    # Difference of Gaussians filter to enhance features
    img_gauss1 = filters.gaussian(img, sigma=5, preserve_range=True)
    img_gauss2 = filters.gaussian(img, sigma=5 * 1.6, preserve_range=True)
    img_gauss = img_gauss1 - img_gauss2
    img_gauss = np.clip(img_gauss, 0, None)
    # Remove terrace edges using morphological opening (removes long/linear features)
    selem = disk(10)  # Adjust radius as needed for your terrace width
    img_no_edges = opening(img_gauss, selem)
    # Contrast adjustment
    p2, p98 = np.percentile(img_no_edges, (2, 98))
    img_rescale = exposure.rescale_intensity(img_no_edges, in_range=(p2, p98))
    img_contrast = exposure.adjust_gamma(img_rescale, gamma=1/contrast)
    # Binary conversion
    img_binary = img_contrast > (thresh / 255.0)
    # img_sav = filters.threshold_sauvola(img, window_size=101)
    # img_binary_sav = img_contrast > img_sav
    return img_gauss, img_contrast, img_binary

# Initial processing
img_gauss, img_contrast, img_binary = process_image(image, init_sigma, init_contrast, init_thresh)

# Plot setup
fig, axes = plt.subplots(1, 2)
plt.subplots_adjust(bottom=0.25)

titles = ['Original', 'Binary']
images = [
    axes[0].imshow(image, cmap='gray'),
    axes[1].imshow(img_binary, cmap='gray'),
    # axes[2].imshow(img_binary_sav, cmap='gray')
]
for ax, title in zip(axes, titles):
    ax.set_title(title)
    ax.axis('off')
images[0].set_data(image)
images[1].set_data(img_binary)


plt.show()

