"""
Shared fixtures.  The equivalence tests compare against the legacy ``core`` package and use the
training images in ``data/user_data/myModel/images``; both are looked up next to this package
(the ScanBound project folder) or via SCANBOUND_LEGACY_ROOT, and the tests skip when absent.
"""

from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY_ROOT = os.environ.get("SCANBOUND_LEGACY_ROOT", os.path.normpath(os.path.join(HERE, "..", "..")))


def _legacy_images() -> list[str]:
    return sorted(glob.glob(os.path.join(LEGACY_ROOT, "data", "user_data", "myModel", "images", "*.jpg")))


@pytest.fixture(scope="session")
def sample_paths() -> list[str]:
    paths = _legacy_images()
    if not paths:
        pytest.skip("training images not found (set SCANBOUND_LEGACY_ROOT)")
    return paths


@pytest.fixture(scope="session")
def saved_params_path() -> str:
    p = os.path.join(LEGACY_ROOT, "data", "user_data", "image_adapt_params.json")
    if not os.path.isfile(p):
        pytest.skip("data/user_data/image_adapt_params.json not found")
    return p


@pytest.fixture(scope="session")
def legacy():
    """The old ``core`` modules (needs PyQt6, which is imported by the dialog module)."""
    if not os.path.isdir(os.path.join(LEGACY_ROOT, "core")):
        pytest.skip("legacy core/ not found")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    pytest.importorskip("matplotlib")
    if LEGACY_ROOT not in sys.path:
        sys.path.insert(0, LEGACY_ROOT)
    import importlib
    mods = {name: importlib.import_module(f"core.{name}")
            for name in ("image_adapt_dialog", "roi_extraction", "object_labeling", "spm_processing")}
    return mods


@pytest.fixture
def blobs() -> np.ndarray:
    """Gray image with 6 bright discs of different sizes on a tilted, noisy background."""
    rng = np.random.default_rng(0)
    img = np.zeros((200, 260), np.float32)
    for (x, y, r) in [(30, 30, 6), (90, 40, 9), (150, 60, 4), (210, 150, 12), (60, 150, 7), (130, 160, 5)]:
        cv2.circle(img, (x, y), r, 1.0, -1)
    yy, xx = np.mgrid[0:200, 0:260]
    return img + 0.002 * xx + rng.normal(0, 0.03, img.shape).astype(np.float32)


@pytest.fixture
def ribbon_mask() -> np.ndarray:
    """A straight 5 px wide ribbon (100 px long) and a T made of two ribbons."""
    m = np.zeros((200, 200), np.uint8)
    cv2.line(m, (20, 30), (120, 30), 1, 5)
    cv2.line(m, (40, 100), (160, 100), 1, 5)       # T bar
    cv2.line(m, (100, 100), (100, 180), 1, 5)      # T stem
    return m.astype(bool)
