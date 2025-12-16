from pathlib import Path

import numpy as np

import pydoppler


def test_foldspec_reads_optional_error_column(tmp_path: Path):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()

    wave = np.linspace(6500.0, 6510.0, 5)
    flux1 = np.linspace(1.0, 2.0, wave.size)
    err1 = np.full_like(wave, 0.1)
    flux2 = np.linspace(1.2, 2.2, wave.size)
    err2 = np.full_like(wave, 0.2)

    (data_dir / "spec1.txt").write_text(
        "\n".join(f"{w:.3f} {f:.6f} {e:.6f}" for w, f, e in zip(wave, flux1, err1)),
        encoding="utf-8",
    )
    (data_dir / "spec2.txt").write_text(
        "\n".join(f"{w:.3f} {f:.6f} {e:.6f}" for w, f, e in zip(wave, flux2, err2)),
        encoding="utf-8",
    )
    (data_dir / "phases.txt").write_text(
        "spec1.txt 0.1\nspec2.txt 0.6\n",
        encoding="utf-8",
    )

    dop = pydoppler.spruit(auto_install=False, workdir=tmp_path / "workdir")
    dop.base_dir = str(data_dir)
    dop.list = "phases.txt"
    dop.Foldspec()

    assert dop.flux_err is not None
    assert len(dop.flux_err) == 2
    assert all(err is not None for err in dop.flux_err)
    assert np.asarray(dop.flux_err[0]).shape == wave.shape
