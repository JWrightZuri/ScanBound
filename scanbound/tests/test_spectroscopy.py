"""Spectra: reading, derivatives against an analytic curve, NaN gaps, provenance."""

from __future__ import annotations

import os

import numpy as np
import pytest

from scanbound.io.nanonis_dat import load_nanonis_dat
from scanbound.measurements.spectroscopy import derivative, normalized_conductance, smooth
from scanbound.model import Axis, Signal, Spectrum

DAT = os.path.join(os.path.dirname(__file__), "data", "bias_spec.dat")


def synthetic(v, i, unit="A"):
    return Spectrum(Axis.from_values("bias", np.asarray(v, float), "V", label="Bias"),
                    [Signal("Current", unit, np.asarray(i, float))])


def test_real_spectrum_reads_with_units_and_position():
    sp = load_nanonis_dat(DAT)
    assert sp.axis.unit == "V" and len(sp.x) > 10
    cur = sp.signal("Current")
    assert cur is not None and cur.unit == "A"
    d = derivative(sp, "Current")
    assert d.names()[-1].startswith("dCurrent/d") and d.signal(d.names()[-1]).unit == "A/V"


def test_derivative_of_a_cubic_is_exact_and_records_the_step():
    v = np.linspace(-1, 1, 201)
    sp = derivative(synthetic(v, v ** 3), "Current", window=9, order=3)
    g = sp.signal(sp.names()[-1]).data
    assert np.allclose(g[5:-5], 3 * v[5:-5] ** 2, atol=1e-6)
    assert sp.history[-1].op == "spectrum.derivative"


def test_nan_gaps_stay_nan_and_do_not_spread():
    v = np.linspace(-1, 1, 101)
    i = v.copy(); i[40:45] = np.nan
    sp = smooth(synthetic(v, i), "Current")
    out = sp.signal(sp.names()[-1]).data if sp.signal(sp.names()[-1]) is not None else sp.signal("Current").data
    assert np.isnan(out[40:45]).all() and np.isfinite(out[:40]).all() and np.isfinite(out[45:]).all()


def test_too_short_or_all_nan_does_not_crash():
    for i in ([1.0, 2.0, 3.0], [np.nan] * 20):
        v = np.linspace(0, 1, len(i))
        sp = derivative(synthetic(v, i), "Current")
        assert np.isnan(sp.signal(sp.names()[-1]).data).all()


def test_normalized_conductance_is_finite_through_zero_bias():
    v = np.linspace(-1, 1, 201)                       # includes V = 0
    sp = derivative(synthetic(v, v ** 3 + 0.1 * v), "Current")
    n = normalized_conductance(sp, sp.names()[-1])
    out = n.signal(n.names()[-1]).data
    assert np.isfinite(out[10:-10]).all()


def test_missing_signal_names_the_available_ones():
    with pytest.raises(KeyError, match="Current"):
        derivative(synthetic([0, 1], [0, 1]), "LIX")
