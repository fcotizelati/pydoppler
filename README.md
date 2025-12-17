# PyDoppler

  This is the repository for a python wrapper for Henk Spruit's doppler tomography software.
  This code can produce a trail spectra of a dataset, and perform
  Doppler tomography maps. It is intended to be a light-weight code for
  single emission line datasets.
  The code will be able to:
  - [x] Load and normalise spectra
  - [x] Plot a trail spectra at initial phase resolution and binned
  - [x] Run Doppler tomography and plot reconstructed spectra
  - [ ] Perform initial spectra binning into user-defined phase bins
  - [ ] User-friendly functions
  - [ ] Auto-save figures.

  The original code and IDL interface can be found at:
   *  https://wwwmpa.mpa-garching.mpg.de/~henk/

  At the moment, there are many features that haven't been implemented. However, the code will be updated
  continuously. If you have any queries/comments/bug reports please send an e-mail to:
   * jvhs1 (at) st-andrews.ac.uk

  ## Acknowledgment
  
  If you make use of this software, please acknowledge the original code and this repository with the following citations:
   * Spruit 1998, arXiv, astro-ph/9806141 (https://ui.adsabs.harvard.edu/abs/1998astro.ph..6141S/abstract)
   * Hernandez Santisteban, 2021, ASCL, 6003 (https://ui.adsabs.harvard.edu/abs/2021ascl.soft06003H/abstract)

  ## Requirements & Installation

  The doppler tomography code is written in fortran. Please ensure that you have a fortran compiler installed.
  At the moment, only gfortran is supported.


  Python >=3.8 is required.

  You can download and install PyDoppler via pip. In a terminal command line, just type:

  ```
  pip install pydoppler
  ```

  If you need to upgrade the package to the latest version, you can do this with
  ```
  pip install --upgrade pydoppler
  ```

  ##  Section 1:  Usage

  You can use the ``sample_script.py`` file to run all the relevant commands
  _shown in sections 2 and 3_ from a terminal command line as:
  ```
  python sample_script.py
  ```
  or in a python console:
  ```python
  run sample_script.py
  ```
  This will read all the files, normalise the spectra, perform Doppler
  tomography and output the results. In the following sections, I will briefly
  explain each main routine.

  ### Quick start for automated workflows

  The package bundles helper utilities that simplify scripting and CI usage. For example:

  ```python
  from pathlib import Path

  import matplotlib.pyplot as plt
  import pydoppler

  # Keep Fortran build products + outputs isolated
  workdir = Path.cwd() / "pydoppler-workdir"
  pydoppler.copy_fortran_code(workdir)
  pydoppler.copy_test_data(workdir)

  dop = pydoppler.spruit(workdir=workdir, interactive=False)
  dop.delw = 35
  dop.overs = 0.3
  dop.base_dir = workdir / "ugem99"
  dop.list = "ugem0all.fas"
  dop.Foldspec()
  dop.Dopin(plot=False)  # continuum bands are estimated automatically
  dop.Syncdop()
  _, dopmap = dop.Dopmap(plot=False)
  ```

  When ``interactive`` is disabled, the continuum bands are determined automatically so the
  normalisation step can run in headless or automated environments. Informational output is
  routed through Python's :mod:`logging` module; configure it to surface diagnostics that were
  previously printed to stdout.

  ##  Section 2: How to load data

  ###  Section 2.1: Test case - the accreting white dwarf - U Gem

  You can start to test PyDoppler with a test dataset kindly provided by J. Echevarria and published
  in Echevarria et al. 2007, AJ, 134, 262 (https://ui.adsabs.harvard.edu/abs/2007AJ....134..262E/abstract).
  To copy the data to your working directory, open a (i)python console and run the
  following commands:
  ```python
  import pydoppler

  pydoppler.test_data()

  ```
  This will create a subdirectory (called ugem99) in your current working directory
  which will contain text files for each spectra (txhugem40*). The format of
  each spectrum file is two columns: _Wavelength_ and _Flux_ (an optional third
  column can provide 1σ flux uncertainties).
  * Wavelength is assumed to be in Angstrom.
  * Don't use headers in the files or us a _#_ at the start of the line, so it
  will be ignored.

  In addition, a phase file (ugem0all.fas) will be added inside the ugem99
  directory which contains the name
  of each spectrum file and its corresponding orbital phase.
  This is a two- or three-column file with the following format:
```
  txhugem4004 0.7150
  txhugem4005 0.7794
         .
         .
         .
```

  ###  Section 2.2: Load your data
  I recommend to stick to the previous file format (as in the test dataset):

  * Individual spectra. Two- or three-column files, _space separated_: Wavelength  Flux  [Flux_Error]
  * Phase file. Two- or three-column file, _space separated_: Spectrum_name  Orbital_Phase  [Delta_Phase]

  and the following directory tree in order for PyDoppler to work properly:

  ```
  wrk_dir
  ├── data_dir (your target)
  │   │
  │   ├── individual_spectra (N spectra)
  │   │
  │   └── phases_file
  │
  └── fortran_code_files
  ```

  ##  Section 3:  Doppler tomography tutorial
  Before running any routines, verify that you have added all the relevant
  parameters into the PyDoppler object.

  * _NOTE:_ PyDoppler runs the Fortran code inside ``dop.workdir`` (defaults to
  ``./pydoppler-workdir``) and writes ``dopin``, ``dop.in``, ``dop.out`` and ``dop.log``
  there. Copy the bundled Fortran sources into that directory with
  ``pydoppler.copy_fortran_code(dop.workdir)`` (or set ``auto_install=True`` when
  constructing the object). Use ``dop.make_run_dir()`` to create a fresh run
  directory when running multiple maps.

  ```python
  from pathlib import Path

  import pydoppler

  # Load base object for tomography
  dop = pydoppler.spruit(workdir=Path.cwd() / "pydoppler-workdir")
  pydoppler.copy_fortran_code(dop.workdir)

  # Basic data for the tomography to work
  dop.object = 'U Gem'
  dop.base_dir = 'ugem99' # Base directory for input spectra
  dop.list = 'ugem0all.fas'		# Name of the input file
  dop.lam0 = 6562.8 # Wavelength zero in units of the original spectra
  dop.delta_phase = 0.003  # Exposure time in terms of orbital phase
  dop.delw = 35	# size of Doppler map in wavelength centred at lam0
  dop.overs = 0.3 # between 0-1, Undersampling of the spectra. 1= Full resolution
  dop.gama = 36.0  # Systemic velocity in km /s
  dop.nbins = 28  # phase bins for diagnostic trail plots (tomography uses all spectra)
  ```

  ### Section 3.1: Read the data
  This routine reads in the raw data and prepares the files for further
  processing.
  ```python
  # Read in the individual spectra and orbital phase information
  dop.Foldspec()
  ```
  ```
  001 txhugem4004  0.715 2048
  002 txhugem4005  0.7794 2048
  003 txhugem4006  0.8348 2048
  004 txhugem4007  0.8942 2048
  005 txhugem4008  0.9518 2048
  006 txhugem4009  0.0072 2048
  007 txhugem4010  0.0632 2048
  008 txhugem4011  0.1186 2048
  009 txhugem4012  0.1745 2048
  010 txhugem4013  0.2344 2048
  011 txhugem4014  0.2904 2048
  012 txhugem4015  0.3724 2048
  013 txhugem4016  0.4283 2048
  014 txhugem4017  0.4866 2048
  015 txhugem4018  0.5425 2048
  016 txhugem4019  0.5979 2048
  017 txhugem4020  0.6544 2048
  018 txhugem4021  0.7098 2048
  019 txhugem4022  0.7652 2048
  020 txhugem4023  0.8195 2048
  021 txhugem4024  0.8772 2048
  022 txhugem4025  0.9269 2048
  023 txhugem4026  0.9614 2048
  024 txhugem4027  0.9959 2048
  025 txhugem4028  0.0304 2048
  026 txhugem4029  0.0648 2048
  027 txhugem4030  0.1027 2048
  028 txhugem4031  0.1372 2048
```

  ### Section 3.2: Normalise the data and set Doppler files
  You will need to define a continuum band (one at each side of the emission line)
  to fit and later subtract the continuum. The normalised spectra are written to
  ``dopin`` inside ``dop.workdir``.

  If your phase file includes a third column with per-spectrum exposure widths
  in orbital phase, pass ``use_list_dpha=True`` to :meth:`Dopin` to write those
  values into ``dopin``.
  ```python  
  # Normalise the spectra
  dop.Dopin(continuum_band=[6500,6537,6591,6620],
            plot_median=False, poly_degree=2)
  ```


  <p align="middle">
     <img src="pydoppler/test_data/output_images/Average_Spec.png" width="350" height="450" />
     <img src="pydoppler/test_data/output_images/Trail.png" width="350" height="450" />
  </p>

  ### Section 3.3: Run the Fortran code
  Now, let's run the tomography software!
  ```python
  # Perform tomography
  dop.Syncdop()
  ```

  The main outputs are written into ``dop.workdir``:
  ``dop.log`` (iteration log) and ``dop.out`` (tomogram + reconstructions).
  ### Section 3.4: Plot the tomography map
  This routine will display the outcome of the Doppler tomography. You can overplot
  contours and streams.
  ```python
  # Read and plot map
  cb,data = dop.Dopmap(limits=[0.05,0.99],colorbar=True,cmaps=plt.cm.magma_r,
  					smooth=False,remove_mean=False)
  # Overplot the donor contours, keplerian and ballistic streams
  qm=0.35   # mass ratio M_2 / M_1
  k1 = 107  # Semi amplitude of the primary in km/s
  inc=70    # inclination angle, in degrees
  m1=1.2    # Mass of the primary in solar masses
  porb=0.1769061911  # orbital period in days


  pydoppler.stream(qm,k1,porb,m1,inc)
  ```
  <p align="middle">
     <img src="pydoppler/test_data/output_images/Doppler_Map.png" width="520" height="450" />
  </p>

  ### Section 3.5: Spectra reconstruction
  Always check that reconstructed spectra looks like the original one. A good
  rule of thumb "If a feature on the Doppler tomogram is not in the trail
  spectrum, most likely its not real!"

  ```python
  # plot trail spectra
  cb2,cb3,dmr,dm = dop.Reco(colorbar=True,limits=[.05,0.95],cmaps=cm.magma_r)
  ```
  where the output variables cb2 and cb3 hold the colorbar objects (if selected); dmr and dm hold the data cubes for the reconstructed trail spectra and the binned data, respectively.
  <p align="middle">
     <img src="pydoppler/test_data/output_images/Reconstruction.png" width="520" height="450" />
  </p>

  ### Section 3.6: Residual diagnostics
  A lightweight residual diagnostic is available to verify that the reconstructed
  trail matches the input data.

  ```python
  dop.Residuals()
  ```

  ## Section 4: Extra commands
  There are many other commands that will interact with the Doppler tomogram. As
  usual, you can update them inside the pydoppler.spruit object as we did in
  section 3. The configurable parameters are:
  ```python3
  dop.ih = 0            # 0=log-likelihood, 1=rms/least-squares (chi^2-like)
  dop.iw = 0            # 1=read error bars from `dopin` (if provided)
  dop.pb0 = 0.95        # range of phases to be ignored, between 0 and 1
  dop.pb1 = 1.05        # range of phases to be ignored, between 1 and 2
  dop.ns = 7            # smearing width in default map
  dop.ac = 8e-4         # accuracy of convergence
  dop.nim = 150         # max no of iterations
  dop.al0 = 0.002       # starting value of alfa
  dop.alf = 1.7         # factor
  dop.nal = 0           # max number of alfas
  dop.clim = 1.6        # 'C-aim'
  dop.ipri = 2          # printout control for standard output channel (ipr=2 for full)
  dop.norm = 1          # norm=1 for normalization to flat light curve
  dop.wid = 10e5        # width of central absorption fudge
  dop.af = 0.0          # amplitude of central absorption fudge
  ```

  ## Section 5: Troubleshoot
  This is an early version of the wrapper. Things will go wrong. If you find a
  bug or need a feature, I will try my best to work it out. If you think you can
  add to this wrapper, I encourage you push new changes and contribute to it.
