"""Small array helpers shared by the steps, the renderer and the detector."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatImage = npt.NDArray[np.float32]
Mask = npt.NDArray[np.bool_]


def norm01(img: npt.ArrayLike) -> FloatImage:
    """Linearly rescale to [0, 1] (float32).  NaNs become 0, constant images become zeros."""
    arr = np.nan_to_num(np.asarray(img, dtype=np.float32))
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def to_u8(img: np.ndarray) -> npt.NDArray[np.uint8]:
    """[0, 1] float -> uint8."""
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_u16(img: np.ndarray) -> npt.NDArray[np.uint16]:
    """[0, 1] float -> uint16."""
    return np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16)


def to_float_gray(image: npt.ArrayLike) -> FloatImage:
    """Any image (gray / RGB / RGBA, any dtype) -> float32 gray in [0, 1]."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = arr[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D image (or HxWxC), got shape {np.shape(image)}")
    return norm01(arr)


def display_u8(img: np.ndarray) -> npt.NDArray[np.uint8]:
    """Bool -> black / white, float in [0, 1] -> 0..255 gray."""
    if img.dtype == np.bool_:
        return img.astype(np.uint8) * 255
    return to_u8(img)
