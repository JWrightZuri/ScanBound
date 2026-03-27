import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import simpledialog
from skimage import filters, measure, morphology, exposure

# Load and normalize image
active_directory = r'C:\Users\wrja\Desktop\STM_Data\napari-stm-annotator\my_napari_testing\\'  # Directory where images are stored
image_name = r'segment1_test_(active)20250521_PdGaA111_279a_1min_flash_0.6A_200K_area10010_z.png'

# Setup Tkinter input prompt
root = tk.Tk()
root.withdraw()

# Load image and normalize
img = cv2.imread(active_directory + image_name, cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32)
img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX)

# Store annotations and rectangles
annotations = []
rect_patches = []

# Drawing canvas
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
ax.set_title("Draw boxes to label features\nPlease be precise!\nPress 's' to save, 'u' to undo, 'q' to quit")

# --- Rectangle selection callback ---
def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    roi = img[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        print("Invalid ROI.")
        return
    
    # 2. (Optional) Enhance contrast5
    contrast = exposure.equalize_adapthist(roi, clip_limit=0.5)

    # Remove background: keep only pixels above a brightness threshold
    # foreground_mask = contrast > 00.7
    # contrast = contrast * foreground_mask

    # 3. Threshold to segment bright regions (molecules)
    threshold = filters.threshold_otsu(contrast)
    binary = contrast > threshold

    # 5. Label connected components
    labeled = measure.label(binary, connectivity=2)
    props = measure.regionprops(labeled)


    height, width = roi.shape
    # area = width * height
    area = [region.area for region in props]
    aspect_ratio = width / height if height > 0 else 0
    mean_intensity = roi.mean()

    # Checkbox for isChiral
    isChiral = False
    isGNR = False
    chiral_dialog = tk.Toplevel()
    chiral_dialog.title("Label Feature")

    label = None
    # Label entry
    label_var = tk.StringVar(value=label)
    tk.Label(chiral_dialog, text="Enter label (e.g. molecule name, adsorbate, vacancy):").pack(padx=10, pady=(10, 0))
    entry = tk.Entry(chiral_dialog, textvariable=label_var)
    entry.pack(padx=10, pady=(0, 10))

    # Chirality checkbox
    chiral_var = tk.BooleanVar()
    chiral_chk = tk.Checkbutton(chiral_dialog, text="Is chiral?", variable=chiral_var)
    chiral_chk.pack(padx=10, pady=(0, 10))

   # GNR checkbox
    gnr_var = tk.BooleanVar()
    gnr_chk = tk.Checkbutton(chiral_dialog, text="Is GNR?", variable=gnr_var)
    gnr_chk.pack(padx=10, pady=(0, 10))

    def on_ok():
        nonlocal isChiral, isGNR, label
        isChiral = chiral_var.get()
        isGNR = gnr_var.get()
        label = label_var.get()
        if label is None or label.strip() == "":
            label = "unlabeled"
        chiral_dialog.destroy()

    btn = tk.Button(chiral_dialog, text="OK", command=on_ok)
    btn.pack(pady=(0, 10))

    chiral_dialog.grab_set()
    chiral_dialog.wait_window()

    if label is None or label.strip() == "":
        label = "unlabeled"

    annotation = {
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max,
        'area': area,
        'aspect_ratio': aspect_ratio,
        'mean_intensity': mean_intensity,
        'isChiral' : isChiral,
        'isGNR': isGNR,
        'label': label.strip()
    }

    annotations.append(annotation)

    # Draw rectangle for visual feedback
    rect = plt.Rectangle((x_min, y_min), width, height, edgecolor='lime', facecolor='none', linewidth=1.0)
    ax.add_patch(rect)
    rect_patches.append(rect)
    fig.canvas.draw_idle()

    print(f"Added: {annotation}")

# --- Keyboard events ---
def on_key(event):
    if event.key == 's':
        df = pd.DataFrame(annotations)
        df.to_csv('stm_labeled_annotations.csv', index=False)
        print("✅ Saved annotations to 'stm_labeled_annotations.csv'")
    elif event.key == 'u':
        if annotations:
            removed = annotations.pop()
            rect = rect_patches.pop()
            rect.remove()
            fig.canvas.draw_idle()
            print(f"↩️ Undid: {removed}")
        else:
            print("Nothing to undo.")
    elif event.key == 'q':
        print("👋 Exiting.")
        plt.close()

# --- Activate selector ---
toggle_selector = RectangleSelector(ax, onselect,
                                    useblit=True,
                                    button=[1],
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

# Should also save images of each feature