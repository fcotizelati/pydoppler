from pathlib import Path

import numpy as np
import pytest

import pydoppler
from pydoppler import pydoppler as pydoppler_module


def test_foldspec_interpolates_descending_wavelength_inputs(tmp_path: Path):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()

    wave = np.linspace(6500.0, 6510.0, 6)
    flux1 = np.linspace(1.0, 2.0, wave.size)
    flux2 = np.linspace(2.0, 3.0, wave.size)

    np.savetxt(data_dir / "spec1.txt", np.c_[wave, flux1])
    np.savetxt(data_dir / "spec2.txt", np.c_[wave[::-1], flux2[::-1]])
    (data_dir / "phases.txt").write_text("spec1.txt 0.1\nspec2.txt 0.6\n", encoding="utf-8")

    dop = pydoppler.spruit(auto_install=False, interactive=False, workdir=tmp_path / "workdir")
    dop.base_dir = str(data_dir)
    dop.list = "phases.txt"
    dop.Foldspec()

    assert np.allclose(dop.flux[0], flux1)
    assert np.allclose(dop.flux[1], flux2)


def test_rebin_trail_wraps_phase_near_one():
    waver = np.linspace(0.0, 1.0, 5)
    flux = np.array([np.linspace(1.0, 2.0, waver.size)])

    trail, phase = pydoppler.rebin_trail(waver, flux, np.array([0.99]), nbins=10, delp=0.2)

    near_wrap = (phase >= -0.05) & (phase <= 0.15)
    wrap_cols = trail[:, near_wrap]
    finite_cols = np.isfinite(wrap_cols[0])

    assert np.any(finite_cols)
    assert np.allclose(wrap_cols[:, finite_cols], flux[0][:, None])


def test_dopin_rejects_empty_velocity_window(tmp_path: Path):
    dop = pydoppler.spruit(auto_install=False, interactive=False, workdir=tmp_path)
    wave = np.linspace(6500.0, 6600.0, 64)
    flux = [np.ones_like(wave) for _ in range(3)]
    phases = np.array([0.1, 0.4, 0.8])

    dop.wave = [wave]
    dop.flux = flux
    dop.pha = phases
    dop.input_phase = phases
    dop.lam0 = 6550.0
    dop.delw = 1e-6

    with pytest.raises(ValueError, match="No spectral samples fall inside"):
        dop.Dopin(
            plot=False,
            poly_degree=1,
            continuum_band=[6500.0, 6520.0, 6580.0, 6600.0],
        )


def test_stream_calculate_rejects_non_positive_mass_ratio():
    with pytest.raises(ValueError, match="must be > 0"):
        pydoppler_module.stream_calculate(0.0)


def test_surface_initializes_azimuth_grid():
    q = 0.7
    rs = pydoppler_module.rlq1(q)
    ni, nj = 12, 12
    r = np.zeros((ni, nj))
    ch = np.zeros(ni)
    ps = np.zeros(nj)

    _, _, ps_out = pydoppler_module.surface(q, rs, ni, nj, r, ch, ps)
    unique_ps = np.unique(np.round(ps_out, decimals=10))

    assert unique_ps.size >= nj - 1
