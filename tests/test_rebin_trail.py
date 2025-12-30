import numpy as np

import pydoppler


def test_rebin_trail_accepts_array_delp():
    waver = np.linspace(0.0, 1.0, 5)
    flux = np.vstack(
        [
            np.linspace(1.0, 2.0, waver.size),
            np.linspace(2.0, 3.0, waver.size),
        ]
    )
    phases = np.array([0.1, 0.6])
    delp = np.array([0.05, 0.1])

    trail, phase = pydoppler.rebin_trail(waver, flux, phases, nbins=4, delp=delp)

    assert trail.shape == (waver.size, 2 * 4 + 2)
    assert phase.size == 2 * 4 + 2
    assert np.any(np.isfinite(trail))
