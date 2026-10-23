"""
Image adaptation steps: turn an image into a clean binary mask of the features.

Gray steps take and return a float32 image in [0, 1]; the threshold step turns it into a bool
mask; binary steps (morphology, size filter, ...) take and return a bool mask.  Keys and
parameter names are those of the saved ``image_adapt_params.json`` files, so pipelines saved by
the dialog load unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

from .._image import norm01, to_u8, to_u16
from .registry import Param, StepRegistry, StepSpec

THRESHOLD_METHODS = ("Otsu", "Triangle", "Manual", "Adaptive (Gaussian)", "Adaptive (Mean)")


# =============================================================================
# Gray steps
# =============================================================================

def f_line_align(img, method):
    ref = np.median(img, axis=1, keepdims=True) if method == "Median" else img.mean(axis=1, keepdims=True)
    return norm01(img - ref)


def smooth_background(img, sigma):
    """Gaussian blur; large sigmas are done on a downsampled copy (same result, far faster)."""
    sigma = float(sigma)
    factor = int(sigma // 6)
    h, w = img.shape
    if factor < 2 or min(h, w) // factor < 16:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def f_background(img, sigma):
    return norm01(img - smooth_background(img, sigma))


def f_contrast(img, low, high):
    lo, hi = np.percentile(img, [low, high])
    if hi <= lo:
        return img
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def f_rescale(img, low, high):
    if high <= low:
        return img
    return np.clip((img - low) / (high - low), 0, 1).astype(np.float32)


def f_gamma(img, gamma):
    return np.power(np.clip(img, 0, 1), gamma).astype(np.float32)


def f_gaussian(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), float(sigma))


def f_median(img, size):
    if size <= 1:
        return img
    return cv2.medianBlur(to_u8(img), int(size)).astype(np.float32) / 255.0


def f_unsharp(img, sigma, amount):
    if sigma <= 0 or amount == 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), float(sigma))
    return np.clip(img + amount * (img - blur), 0, 1).astype(np.float32)


def f_clahe(img, clip, tiles):
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    return clahe.apply(to_u16(img)).astype(np.float32) / 65535.0


def f_invert(img):
    return (1.0 - img).astype(np.float32)


# =============================================================================
# Gray -> binary
# =============================================================================

def f_threshold(img, method, features, value, block, offset):
    from skimage.filters import threshold_otsu, threshold_triangle

    # keep the window valid for small images
    block = int(min(block, (min(img.shape) // 2) * 2 - 1))
    block = max(block, 3)

    try:
        if method == "Otsu":
            t = threshold_otsu(img)
        elif method == "Triangle":
            t = threshold_triangle(img)
        elif method == "Manual":
            t = value
        elif method == "Adaptive (Gaussian)":
            t = cv2.GaussianBlur(img, (block, block), 0)
        else:  # "Adaptive (Mean)"
            t = cv2.blur(img, (block, block))
    except ValueError:  # e.g. constant image
        t = 0.5

    off = 0.0 if method == "Manual" else offset
    # positive offset = stricter in both modes
    if features == "Bright":
        return img > (t + off)
    return img < (t - off)


# =============================================================================
# Binary steps
# =============================================================================

def f_morph(mask, op, size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size), int(size)))
    code = {"Open": cv2.MORPH_OPEN, "Close": cv2.MORPH_CLOSE,
            "Erode": cv2.MORPH_ERODE, "Dilate": cv2.MORPH_DILATE}[op]
    out = cv2.morphologyEx(mask.astype(np.uint8), code, kernel, iterations=int(iterations))
    return out > 0


def f_size_filter(mask, min_size, max_size):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= min_size
    if max_size > 0:
        keep &= areas <= max_size
    keep[0] = False  # background
    return keep[labels]


def f_fill_holes(mask, max_area):
    h, w = mask.shape
    inv = (~mask).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    touches_border = (x == 0) | (y == 0) | (x + bw == w) | (y + bh == h)
    fill = (~touches_border) & (area <= max_area)
    fill[0] = False
    return mask | fill[labels]


def f_skeleton(mask):
    from skimage.morphology import skeletonize
    return skeletonize(mask)


# =============================================================================
# Registry
# =============================================================================

ADAPT_STEPS = StepRegistry("adapt", [
    # ---- gray image steps ---------------------------------------------------
    StepSpec("line_align", "Line align", "gray", f_line_align, (
        Param("method", "Method", "choice", "Median", choices=("Median", "Mean")),
    ), "Subtract each scan row's median/mean (removes STM scan-line offsets)."),
    StepSpec("background", "Background subtraction", "gray", f_background, (
        Param("sigma", "Sigma (px)", "float", 30.0, 1, 200, 1, 0),
    ), "Subtract a heavily blurred copy to flatten slow background variations."),
    StepSpec("contrast", "Contrast stretch", "gray", f_contrast, (
        Param("low", "Low %", "float", 1.0, 0, 49, 0.5, 1),
        Param("high", "High %", "float", 99.0, 51, 100, 0.5, 1),
    ), "Clip to percentiles and rescale to the full range."),
    StepSpec("rescale", "Intensity window", "gray", f_rescale, (
        Param("low", "Low", "float", 0.0, 0, 1, 0.01, 2),
        Param("high", "High", "float", 1.0, 0, 1, 0.01, 2),
    ), "Map the fixed range [low, high] to [0, 1] (e.g. drop a dark substrate)."),
    StepSpec("gamma", "Gamma", "gray", f_gamma, (
        Param("gamma", "Gamma", "float", 1.0, 0.1, 5.0, 0.05, 2),
    ), "Power-law intensity curve (<1 brightens, >1 darkens)."),
    StepSpec("gaussian", "Gaussian blur", "gray", f_gaussian, (
        Param("sigma", "Sigma (px)", "float", 1.0, 0, 10, 0.1, 1),
    ), "Smooth noise."),
    StepSpec("median", "Median blur", "gray", f_median, (
        Param("size", "Kernel size", "odd", 3, 1, 31, 2, 0),
    ), "Edge-preserving noise removal."),
    StepSpec("unsharp", "Unsharp mask", "gray", f_unsharp, (
        Param("sigma", "Sigma (px)", "float", 2.0, 0.1, 20, 0.1, 1),
        Param("amount", "Amount", "float", 1.0, 0, 5, 0.1, 1),
    ), "Sharpen edges."),
    StepSpec("clahe", "CLAHE", "gray", f_clahe, (
        Param("clip", "Clip limit", "float", 2.0, 1, 40, 0.5, 1),
        Param("tiles", "Tile grid", "int", 8, 2, 32, 1, 0),
    ), "Local adaptive histogram equalisation."),
    StepSpec("invert", "Invert", "gray", f_invert, (), "Flip bright and dark."),

    # ---- gray -> binary -----------------------------------------------------
    StepSpec("threshold", "Threshold", "gray", f_threshold, (
        Param("method", "Method", "choice", "Otsu", choices=THRESHOLD_METHODS),
        Param("features", "Features are", "choice", "Bright", choices=("Bright", "Dark")),
        Param("value", "Value", "float", 0.5, 0, 1, 0.01, 2,
              visible_if=lambda p: p["method"] == "Manual"),
        Param("block", "Block size", "odd", 31, 3, 301, 2, 0,
              visible_if=lambda p: p["method"].startswith("Adaptive")),
        Param("offset", "Offset", "float", 0.0, -0.5, 0.5, 0.005, 3,
              visible_if=lambda p: p["method"] != "Manual",
              tooltip="Positive = stricter (fewer pixels pass)."),
    ), "Turns the image into a binary mask."),

    # ---- binary steps -------------------------------------------------------
    StepSpec("morph", "Morphology", "binary", f_morph, (
        Param("op", "Operation", "choice", "Open", choices=("Open", "Close", "Erode", "Dilate")),
        Param("size", "Kernel size", "int", 3, 1, 25, 1, 0),
        Param("iterations", "Iterations", "int", 1, 1, 10, 1, 0),
    ), "Open removes specks, close fills gaps, erode/dilate shrink/grow."),
    StepSpec("size_filter", "Object size filter", "binary", f_size_filter, (
        Param("min_size", "Min area (px)", "int", 30, 0, 5000, 1, 0),
        Param("max_size", "Max area (px, 0=off)", "int", 0, 0, 100000, 10, 0),
    ), "Drop connected objects that are too small / too large."),
    StepSpec("fill_holes", "Fill holes", "binary", f_fill_holes, (
        Param("max_area", "Max hole area (px)", "int", 50, 0, 10000, 1, 0),
    ), "Fill enclosed background holes up to a given size."),
    StepSpec("skeleton", "Skeletonize", "binary", f_skeleton, (),
             "Thin every object to a 1 px wide centre line."),
])

# Pipeline loaded on startup / "Reset": (step key, enabled, param overrides)
DEFAULT_ADAPT_PIPELINE: list[tuple[str, bool, dict]] = [
    ("gaussian", True, {"sigma": 1.0}),
    ("background", True, {"sigma": 40.0}),
    ("threshold", True, {"method": "Otsu"}),
    ("morph", False, {}),
    ("size_filter", True, {"min_size": 30}),
]

# Graphene-nanoribbon preset, after core/GNR_segment.create_image_mask: row alignment and a
# wide high-pass for terraces, light smoothing, a fixed window that drops the Au(111)
# herringbone background, local threshold, small-object removal and a thin erosion.
NANORIBBON_PIPELINE: list[tuple[str, bool, dict]] = [
    ("line_align", True, {"method": "Median"}),
    ("background", True, {"sigma": 50.0}),
    ("gaussian", True, {"sigma": 1.5}),
    ("rescale", True, {"low": 0.40, "high": 1.0}),
    ("threshold", True, {"method": "Adaptive (Gaussian)", "features": "Bright", "block": 51, "offset": 0.0}),
    ("size_filter", True, {"min_size": 100}),
    ("morph", True, {"op": "Erode", "size": 3, "iterations": 1}),
]
