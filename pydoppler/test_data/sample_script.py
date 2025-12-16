import os
from pathlib import Path

import pydoppler
import numpy as np

# Set to True to save figures and run without GUI windows.
SAVE_PNGS = True
# Set to True to display figures interactively at the end.
SHOW_PLOTS = False

# Fortran execution + outputs are isolated in this directory.
workdir = Path.cwd() / "pydoppler-workdir"
if SAVE_PNGS and not SHOW_PLOTS:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(workdir / ".mplconfig"))

import matplotlib.pyplot as plt
# Import sample data
# <<< COMMENT OUT IF YOU DONT NEED THE TEST DATASET >>>
pydoppler.test_data()

pydoppler.copy_fortran_code(workdir, overwrite=True)

# Load base object for tomography
dop = pydoppler.spruit(workdir=workdir)

# Basic data for the tomography to work
dop.object = 'U Gem'
dop.base_dir = 'ugem99' # Base directory for input spectra
dop.list = 'ugem0all.fas'		# Name of the input file
dop.lam0 = 6562.8 # Wavelength zero in units of the original spectra
dop.delta_phase = 0.003
dop.delw = 35	# size of Doppler map in wavelenght
dop.overs = 0.3 # between 0-1, Undersampling of the spectra. 1= Full resolution
dop.gama = 36.0  # km /s
dop.nbins = 28

# Read in the individual spectra and orbital phase information
dop.Foldspec()

# Normalise the spectra
dop.Dopin(continuum_band=[6500,6537,6591,6620],
        plot_median=False,poly_degree=2, show=False)

# Perform tomography
dop.Syncdop()

# This routine will display the outcome of the Doppler tomography.
# You can overplot contours and streams.
cb,data = dop.Dopmap(limits=[0.05,0.99],colorbar=False,cmaps=plt.cm.magma_r,
                     smooth=False,remove_mean=False,show=False)

# Overplot the donor contours, keplerian and ballistic streams
qm=0.35
k1 = 107
inc=70
m1=1.2
porb=0.1769061911

pydoppler.stream(qm,k1,porb,m1,inc)

# Always check that reconstructed spectra looks like the original one. A good
# rule of thumb "If a feature on the Doppler tomogram isn not in the trail,
# most likely its not real!"
cb2,cb3,dmr,dm = dop.Reco(colorbar=False,limits=[.05,0.95],cmaps=plt.cm.magma_r, show=False)

dop.Residuals(dm=dm, dmr=dmr, show=False)

if SAVE_PNGS:
    outdir = workdir / "output_images"
    outdir.mkdir(parents=True, exist_ok=True)
    figures = {
        "Average Spec": "Average_Spec.png",
        "Trail": "Trail.png",
        "Doppler Map": "Doppler_Map.png",
        "Reconstruction": "Reconstruction.png",
        "Residuals": "Residuals.png",
    }
    existing = set(plt.get_figlabels())
    for label, filename in figures.items():
        if label not in existing:
            continue
        fig = plt.figure(label)
        fig.savefig(outdir / filename, dpi=200, bbox_inches="tight")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
