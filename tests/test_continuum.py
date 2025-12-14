import numpy as np
import pytest

import pydoppler


def test_auto_continuum_band_uses_outer_five_percent():
    dop = pydoppler.spruit(auto_install=False, interactive=False)
    dop.wave = [np.linspace(6500.0, 6600.0, 100)]

    band = dop._auto_continuum_band()

    wave = dop.wave[0]
    window = max(3, int(round(0.05 * wave.size)))
    left = wave[:window]
    right = wave[-window:]
    expected = (float(left.min()), float(left.max()), float(right.min()), float(right.max()))

    assert band == expected


def test_auto_continuum_band_rejects_too_few_samples():
    dop = pydoppler.spruit(auto_install=False, interactive=False)
    dop.wave = [np.linspace(6500.0, 6600.0, 7)]

    with pytest.raises(ValueError):
        dop._auto_continuum_band()
