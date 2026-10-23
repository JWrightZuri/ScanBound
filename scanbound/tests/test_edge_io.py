"""Readers and the image model on real and deliberately broken files."""

from __future__ import annotations

import math
import os
import shutil

import numpy as np
import pytest
from PIL import Image as PILImage

from scanbound.io import open_data
from scanbound.io.data import DataFormatError, list_data_files, load_stack
from scanbound.model import Image
from scanbound.model.axes import Axis

DATA = os.path.join(os.path.dirname(__file__), "data")
ANISO = os.path.join(DATA, "aniso_0020.sxm")
DAT = os.path.join(DATA, "bias_spec.dat")


def codes(img: Image) -> set[str]:
    return {q.code for q in img.quality()}


# --------------------------------------------------------------------------- real files
def test_anisotropic_sxm_keeps_both_pixel_sizes():
    f = open_data(ANISO)
    img = f.image()
    assert img.shape == (50, 144)
    dx, dy = img.pixel_size
    assert dx != pytest.approx(dy, rel=1e-3)
    assert not img.isotropic
    assert img.x.unit == img.y.unit == "m"
    assert "anisotropic" in codes(img)
    assert img.frame is not None and len(f.channels) >= 2


def test_bias_spectrum_axis_and_position():
    f = open_data(DAT)
    assert f.kind == "spectrum"
    sp = f.spectrum()
    assert len(sp.x) == 400
    assert sp.x[0] == pytest.approx(-2.0) and sp.x[-1] == pytest.approx(2.0)
    assert sp.position is not None and sp.position[0] == pytest.approx(-54.5676e-9)
    assert sp.signal("Current") is not None


def test_truncated_sxm_pads_with_nan(tmp_path):
    full = os.path.getsize(ANISO)
    cut = tmp_path / "cut.sxm"
    with open(ANISO, "rb") as fh:
        cut.write_bytes(fh.read(full - 9000))
    f = open_data(str(cut))
    images = f.images()
    assert all(im.shape == (50, 144) for im in images)
    assert np.isfinite(images[0].data).all()             # channels are stored one after the other:
    last = images[-1]                                     # the cut removes the end of the last one
    assert np.isnan(last.data).any() and np.isfinite(last.data).any()
    assert "nan" in codes(last)
    img = last
    assert f.warnings                                  # the user is told the file is incomplete
    assert math.isfinite(img.stats.minimum)


@pytest.mark.parametrize("payload", [b"", b"\x00" * 10, b"garbage that is not a scan\n" * 50])
def test_malformed_files_raise_clear_errors(tmp_path, payload):
    for ext in (".sxm", ".dat", ".png"):
        p = tmp_path / f"bad{ext}"
        p.write_bytes(payload)
        with pytest.raises(Exception) as exc:
            open_data(str(p))
        assert str(exc.value)                          # a message, not a bare crash
        assert not isinstance(exc.value, (MemoryError, RecursionError))


def test_list_data_files_skips_project_and_hidden(tmp_path):
    shutil.copy(ANISO, tmp_path / "a.sxm")
    shutil.copy(DAT, tmp_path / "b.dat")
    (tmp_path / "scanbound_project.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("x")
    names = sorted(os.path.basename(p) for p in list_data_files(str(tmp_path)))
    assert names == ["a.sxm", "b.dat"]


# --------------------------------------------------------------------------- plain images, bit depths
@pytest.mark.parametrize("mode,dtype,maxval", [("L", np.uint8, 255), ("I;16", np.uint16, 65535), ("F", np.float32, None),
                                               ("1", bool, 1)])
def test_bit_depths_keep_raw_values(tmp_path, mode, dtype, maxval):
    rng = np.random.default_rng(0)
    if dtype is bool:
        arr = rng.random((40, 60)) > 0.5
        pil = PILImage.fromarray(arr.astype(np.uint8) * 255).convert("1")
    elif dtype is np.float32:
        arr = rng.normal(size=(40, 60)).astype(np.float32)
        pil = PILImage.fromarray(arr, mode="F")
    else:
        arr = rng.integers(0, maxval, size=(40, 60)).astype(dtype)
        pil = PILImage.fromarray(arr)
    path = tmp_path / ("img.tif" if mode in ("F", "I;16") else "img.png")
    pil.save(path)
    img = open_data(str(path)).image()
    assert img.shape == (40, 60)
    if dtype is bool:
        assert set(np.unique(img.data)) <= {0.0, 1.0, 255.0}
    else:
        np.testing.assert_allclose(img.data, arr.astype(float), rtol=1e-6)
    assert not img.calibrated                          # no metadata: pixel units, said so
    assert "uncalibrated" in codes(img)


def test_rgb_image_and_metadata_sidecar(tmp_path):
    rgb = np.zeros((20, 30, 3), np.uint8)
    rgb[..., 0] = 200
    PILImage.fromarray(rgb).save(tmp_path / "pic_Z.jpg")
    (tmp_path / "metadata.csv").write_text("Filename,Pixel Scale X (nm),Pixel Scale Y (nm)\npic_Z.jpg,0.5,0.25\n")
    img = open_data(str(tmp_path / "pic_Z.jpg")).image()
    assert img.calibrated
    assert img.pixel_size[0] == pytest.approx(0.5e-9) and img.pixel_size[1] == pytest.approx(0.25e-9)
    assert img.x.calibration.source == "sidecar"


def test_multipage_tiff_is_a_stack(tmp_path):
    frames = [PILImage.fromarray(np.full((16, 16), k * 10, np.uint8)) for k in range(4)]
    path = tmp_path / "stack.tif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    f = open_data(str(path))
    assert len(f.images()) == 4
    stack = load_stack(str(path))
    assert len(stack.images) == 4
    assert [float(im.data.mean()) for im in stack.images] == [0.0, 10.0, 20.0, 30.0]


# --------------------------------------------------------------------------- the image model
def test_image_is_immutable_and_nan_aware():
    data = np.arange(12, dtype=float).reshape(3, 4)
    data[1, 2] = np.nan
    img = Image(data)
    with pytest.raises(ValueError):
        img.data[0, 0] = 5
    data[0, 0] = 99                                    # the caller's array is copied
    assert img.data[0, 0] == 0
    assert img.stats.n == 11 and img.stats.n_invalid == 1
    filled = img.filled("nearest")
    assert np.isfinite(filled).all() and filled.flags.writeable


def test_all_nan_image_is_an_error_not_a_crash():
    img = Image(np.full((8, 8), np.nan))
    q = img.quality()
    assert q and q[0].level == "error"
    assert img.filled("nearest").shape == (8, 8)


def test_saturated_pixels_flagged():
    data = np.random.default_rng(1).normal(size=(64, 64))
    data[:5, :] = data.max() + 1                       # a clipped band
    img = Image(data)
    assert "saturated" in codes(img)
    assert img.saturated.sum() >= 5 * 64


def test_huge_image_statistics_are_fast():
    import time
    data = np.zeros((8192, 8192), np.float32)
    data[::7, ::5] = 1.0
    data[100, :50] = np.nan
    t = time.perf_counter()
    img = Image(data)
    st = img.stats
    q = img.quality()
    assert time.perf_counter() - t < 20
    assert st.n_invalid == 50 and st.maximum == 1.0
    assert any(i.code == "nan" for i in q)


def test_axis_coordinates_pixel_edges():
    ax = Axis.spatial("x", 10, 2e-10, offset_m=1e-9)
    assert ax.to_physical(0) == pytest.approx(1e-9)
    assert ax.to_pixel(ax.to_physical(3.25)) == pytest.approx(3.25)
