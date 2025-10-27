#! Python PyDoppler

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib.cm as cm
import matplotlib.collections as mcoll
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from matplotlib.colors import Normalize

try:  # Python 3.9+
    from importlib.resources import files as resource_files
except ImportError:  # pragma: no cover - Python <3.9 fallback
    from importlib_resources import files as resource_files  # type: ignore


plt.rcParams.update({'font.size': 12})


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def _copy_tree(source, destination: Path, overwrite: bool = False) -> List[Path]:
    """Copy the files contained in *source* into *destination*.

    Parameters
    ----------
    source:
        An importlib ``Traversable`` pointing to a directory.
    destination:
        Filesystem directory that will receive the copied files.
    overwrite:
        When ``True`` existing files will be replaced, otherwise they are kept
        untouched.
    """

    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []
    for item in source.rglob("*"):
        if item.is_dir():
            continue

        target = dest_path / item.relative_to(source)
        if target.exists() and not overwrite:
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        with item.open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        copied.append(target)

    return copied


def get_fortran_code_path():
    """Return a Traversable pointing to the bundled Fortran sources."""

    return resource_files(__package__).joinpath("fortran_code")


def get_test_data_path():
    """Return a Traversable pointing to the bundled test data."""

    return resource_files(__package__).joinpath("test_data")


def copy_fortran_code(
    destination: Union[Path, str], overwrite: bool = False
) -> List[Path]:
    """Copy the bundled Fortran assets into *destination*."""

    return _copy_tree(get_fortran_code_path(), Path(destination), overwrite)


def copy_test_data(
    destination: Union[Path, str], overwrite: bool = False
) -> List[Path]:
    """Copy the bundled sample data into *destination*."""

    return _copy_tree(get_test_data_path(), Path(destination), overwrite)


def install_sample_script(
    destination: Union[Path, str],
    overwrite: bool = False,
) -> Path:
    """Install the example ``sample_script.py`` in *destination*.

    When ``overwrite`` is ``False`` the script will not clobber an existing
    file. Instead it will fall back to the ``sample_script-<n>.py`` naming
    convention previously used by the project.
    """

    dest_dir = Path(destination)
    dest_dir.mkdir(parents=True, exist_ok=True)

    target = dest_dir / "sample_script.py"
    if target.exists() and not overwrite:
        index = 1
        while target.exists():
            target = dest_dir / f"sample_script-{index}.py"
            index += 1

    source = get_test_data_path().joinpath("sample_script.py")
    with source.open("rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)

    return target


class spruit:
    """
        A class to store and process data for Doppler tomography code
        by Henk Spruit.

    ...

    Methods
    -------
    foldspec()
        Reads the data and stores in spruit object
    sort(column, order='ascending')
        Sort by `column`
    """
    def __init__(
        self,
        force_install: bool = False,
        auto_install: bool = True,
        interactive: bool = False,
    ):
        self.object = 'disc'
        self.wave = 0.0
        self.flux = 0.0
        self.pha = 0.0
        self.input_files = 0.0
        self.input_phase = 0.0
        self.trsp = 0.0
        self.nbins = 20
        self.normalised_flux = 0.0
        self.normalised_wave = 0.0
        self.base_dir = '.'
        self.lam0 = 6562.83
        self.delw = 80
        self.list = 'phases.txt'
        self.gama = 0.0
        self.delta_phase = 0.001

        self.verbose = True
        self.interactive = interactive
        self.logger = logging.getLogger(f"{__name__}.spruit")
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

        ###### Plotting parameters
        self.psname='j0644'              # Name of output plot file
        self.output='pdf'                    # Can choose between: pdf, eps or png
        self.data=False                      # If True then exit data will put in file *.txt
        self.plot=True                       # Plot in Python window
        self.plotlim=1.3                     # Plot limits. 1 = close fit.
        self.overs=0.4

        ####### Dop.in parameters
        self.ih = 0
        self.iw = 0
        self.pb0 = 0.95
        self.pb1 = 1.05
        self.ns = 7
        self.ac = 8e-4
        self.nim = 150
        self.al0 = 0.002
        self.alf = 1.7
        self.nal = 0
        self.clim = 1.6
        self.ipri = 2
        self.norm = 1
        self.wid = 10e5
        self.af = 0.0


        # %%%%%%%%%%%%%%%%%%  Doppler Options   %%%%%%%%%%%%%%%%%%

        self.lobcol='white'                 # Stream color

        if auto_install:
            self.install_assets(Path.cwd(), force_install)

    # ------------------------------------------------------------------
    # helper methods
    def _log(self, level: int, message: str) -> None:
        """Emit *message* through :mod:`logging` and optionally echo it."""

        self.logger.log(level, message)
        if self.verbose and level >= logging.INFO:
            print(message)

    def install_assets(
        self, destination: Union[Path, str], force: bool = False
    ) -> None:
        """Install bundled assets into *destination*.

        Parameters
        ----------
        destination:
            Directory where the Fortran sources and helper script should be
            installed.
        force:
            When ``True`` existing files are overwritten.
        """

        dest_dir = Path(destination)

        copied_fortran = copy_fortran_code(dest_dir, overwrite=force)
        script_path: Optional[Path] = None
        if force or copied_fortran:
            script_path = install_sample_script(dest_dir, overwrite=force)

        if copied_fortran:
            self._log(logging.INFO, f"Installed Fortran assets into {dest_dir}.")
        else:
            self._log(logging.DEBUG, "Fortran assets already present; skipping copy.")

        if script_path:
            self._log(logging.INFO, f"Sample script available at {script_path}.")
        else:
            self._log(logging.DEBUG, "Sample script already present; not copied.")


    def Foldspec(self):
        """Foldspec. Prepares the spectra to be read by dopin.
        *** Remember to prepare the keywords before running ***

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        list_path = Path(self.base_dir) / self.list
        if not list_path.exists():
            raise FileNotFoundError(
                f"Phase file '{list_path}' is not accessible. Check 'base_dir' and 'list'."
            )

        list_dir = list_path.parent

        inputs = np.genfromtxt(
            list_path,
            dtype=[("files", "U256"), ("phase", float)],
            usecols=(0, 1),
            comments="#",
            autostrip=True,
        )

        if inputs.shape == ():
            inputs = inputs.reshape(1,)

        if inputs.size < 1:
            raise ValueError("No entries found in the list file.")

        first_file = inputs["files"][0]
        first_path = list_dir / first_file

        try:
            w1st, f1st = np.loadtxt(first_path, usecols=(0, 1), unpack=True)
        except Exception as exc:  # pragma: no cover - numpy handles the heavy lifting
            raise RuntimeError(f"Failed to read first spectrum '{first_path}': {exc}") from exc

        wo = w1st
        wave, flux = [], []

        if getattr(self, "nbins", None) is None:
            phases_sorted = np.sort(np.asarray(inputs["phase"], dtype=float))
            dphi = np.diff(phases_sorted)
            dphi = dphi[dphi > 0]
            if dphi.size == 0:
                self.nbins = 20
            else:
                self.nbins = int(max(4, round(1.5 / float(np.median(dphi)))))

        try:
            rep_dphi = float(np.median(np.diff(np.sort(inputs["phase"]))))
        except Exception:  # pragma: no cover - extremely small data set
            rep_dphi = float("nan")

        self._log(
            logging.INFO,
            f"Number of bins: {self.nbins} (median phase spacing {rep_dphi:.5f})",
        )

        for z, (fname, ph) in enumerate(zip(inputs["files"], inputs["phase"])):
            sp_path = list_dir / fname
            try:
                w, f = np.loadtxt(sp_path, usecols=(0, 1), unpack=True)
            except Exception as exc:
                raise RuntimeError(f"Failed to read spectrum '{sp_path}': {exc}") from exc

            if z == 0:
                wave.append(wo)
                flux.append(f1st)
            else:
                wave.append(wo)
                flux.append(np.interp(wo, w, f))

            self._log(
                logging.INFO,
                f"{str(z + 1).zfill(3)} {fname}  {ph} {w.size}",
            )

        self.wave = wave
        self.flux = flux
        self.pha = np.asarray(inputs["phase"], float)
        self.input_files = np.asarray(inputs["files"], dtype=str)
        self.input_phase = np.asarray(inputs["phase"], dtype=float)
        self.trsp = np.vstack(flux)


    def _auto_continuum_band(self) -> Sequence[float]:
        """Derive continuum windows for non-interactive use.

        The routine samples the outer 5% of the wavelength coverage on both
        sides of the line and returns two intervals suitable for the continuum
        polynomial fit used in :meth:`Dopin`.
        """

        if not self.wave or not self.wave[0].size:
            raise RuntimeError(
                "Wavelength grid is empty. Run 'Foldspec' before calling 'Dopin'."
            )

        wave = np.asarray(self.wave[0], dtype=float)
        if wave.size < 8:
            raise ValueError("Cannot determine continuum bands from fewer than eight samples.")

        window = max(3, int(round(0.05 * wave.size)))
        left = wave[:window]
        right = wave[-window:]

        return (
            float(left.min()),
            float(left.max()),
            float(right.min()),
            float(right.max()),
        )


    def Dopin(self,poly_degree=2, continnum_band=False,
              rebin=True,plot_median = False, rebin_wave= 0.,
              xlim=None,two_orbits=True,vel_space=True,
              verbose=False):
        """Normalises each spectrum to a user-defined continnum.
            Optional, it plots a trail spectra. When ``self.interactive`` is
            ``False`` and no continuum bands are supplied the routine selects
            suitable regions automatically so it can run headlessly.

        Parameters
        ----------
        poly_degree : int, Optional
            polynomial degree to fit the continuum. Default, 2.

        continnum_band : array-like,
            Define two wavelength bands (x1,x2) and (x3,x4)
            to fit the continuum.
            contiunnum_band = [x1,x2,x3,x4].
            If False, an interactive plot will allow to select this four numbers
            - Default, False

        plot_median : bool,
            Plots above teh trail spectra a median of the dataset.
            - Defautl, False

        rebin_wave : float,
            TBD

        xlim : float,
            TBD

        two_orbits : float,
            TBD

        vel_space : float,
            TBD

        verbose : float,
            TBD

        Returns
        -------
        None.

        """
        if not self.wave:
            raise RuntimeError("No spectra loaded. Run 'Foldspec' before normalising.")

        lam=self.lam0
        cl=2.997e5

        cmaps = plt.cm.binary_r #cm.winter_r cm.Blues#cm.gist_stern

        wmin = float(np.min(self.wave[0]))
        wmax = float(np.max(self.wave[0]))
        if lam < wmin or lam > wmax:
            raise ValueError(
                f"Input wavelength {lam} Å out of bounds. Must be between {wmin} and {wmax}."
            )
        ss=0
        for i in np.arange(len(self.wave[0])-1):
            if lam >= self.wave[0][i]   and lam <= self.wave[0][i+1]:
                ss=i


        fig=plt.figure(num="Average Spec",figsize=(6.57,8.57))
        plt.clf()
        ax=fig.add_subplot(211)
        avgspec=np.sum(self.flux,axis=0)
        plt.plot(self.wave[0],avgspec/len(self.pha))

        if not continnum_band:
            if self.interactive:
                self._log(logging.INFO, "Choose 4 points to define the continuum.")
                xor=[]
                for _ in np.arange(4):
                    selection=plt.ginput(1,timeout=-1)
                    if not selection:
                        raise RuntimeError("Continuum selection aborted by user.")
                    xor.append(float(selection[0][0]))
                    if self.plot:
                        plt.axvline(x=xor[-1],linestyle='--',color='k')
                        plt.draw()
            else:
                xor=list(self._auto_continuum_band())
                self._log(
                    logging.INFO,
                    "Using automatic continuum bands at "
                    f"[{xor[0]:.2f}, {xor[1]:.2f}] and [{xor[2]:.2f}, {xor[3]:.2f}] Å.",
                )
                if self.plot:
                    for idx,val in enumerate(xor):
                        label='Cont Bands' if idx == 0 else ''
                        plt.axvline(x=val,linestyle='--',color='k',label=label)
        else:
            xor=[float(v) for v in continnum_band]
            if len(xor) != 4:
                raise ValueError("continnum_band must contain four wavelength limits.")
            if xor[0] >= xor[1] or xor[2] >= xor[3]:
                raise ValueError("Each continuum interval must be strictly increasing.")
            if self.plot:
                for idx,val in enumerate(xor):
                    label='Cont Bands' if idx == 0 else ''
                    plt.axvline(x=val,linestyle='--',color='k',label=label)
        lop = ((self.wave[0]>xor[0]) * (self.wave[0]<xor[1])) + ((self.wave[0]>xor[2]) * (self.wave[0]<xor[3]))
        yor=avgspec[lop]/len(self.pha)
        plt.ylim(avgspec[lop].min()/len(self.pha)*0.8,avgspec.max()/len(self.pha)*1.1)
        z = np.polyfit(self.wave[0][lop], yor, poly_degree)
        pz = np.poly1d(z)
        linfit = pz(self.wave[0])
        plt.plot(self.wave[0],linfit,'r',label='Cont Fit')
        lg = plt.legend(fontsize=14)
        plt.xlim(xor[0]-10,xor[3]+10)
        plt.xlabel(r'Wavelength / $\AA$')
        plt.ylabel('Input flux')


        ax=fig.add_subplot(212)
        vell=((self.wave[0]/self.lam0)**2-1)*cl/(1+(self.wave[0]/self.lam0)**2)

        plt.plot(vell,avgspec/len(self.pha)-linfit,'k')
        plt.axhline(y=0,linestyle='--',color='k')
        plt.axvline(x=-self.delw/self.lam0*cl,linestyle='-',color='DarkOrange')
        plt.axvline(x= self.delw/self.lam0*cl,linestyle='-',
                    color='DarkOrange',label='DopMap limits')
        lg = plt.legend(fontsize=14)
        plt.xlim(-self.delw/self.lam0*cl*1.5,self.delw/self.lam0*cl*1.5)
        qq = (np.abs(vell) < self.delw/self.lam0*cl*1.5)
        plt.ylim(-0.05*np.max(avgspec[qq]/len(self.pha)-linfit[qq] -1.0),
                np.max(avgspec[qq]/len(self.pha)-linfit[qq] -1.0)*1.1)
        plt.xlabel('Velocity km/s')
        plt.ylabel('Bkg subtracted Flux')
        if self.plot:
            plt.draw()
        plt.tight_layout()

        ######## Do individual fit on the blaze
        for ct,flu in enumerate(self.flux):
            #print(lop.sum)
            if ct == 0 :
                nufac=(1.0+self.gama/2.998e5) * np.sqrt(1.0-(self.gama/2.998e5)**2)
                lop = (self.wave[0]/nufac > self.lam0 - self.delw) * \
                      (self.wave[0]/nufac < self.lam0 + self.delw)
                self.normalised_wave = np.array(self.wave[0][lop]/nufac)
                # Interpolate in velocity space
                vell_temp=((self.normalised_wave/self.lam0)**2-1.0)*cl/(1.0 + \
                         (self.normalised_wave/self.lam0)**2)
                self.vell = np.linspace(vell_temp[0],vell_temp[-1],vell_temp.size)
                self.normalised_flux = np.zeros((len(self.flux),lop.sum()))

            polmask = ((self.wave[0]/nufac>xor[0]) * (self.wave[0]/nufac<xor[1])) +\
                  ((self.wave[0]/nufac>xor[2]) * (self.wave[0]/nufac<xor[3]))
            z = np.polyfit(self.wave[0][polmask]/nufac,flu[polmask], 3)
            pz = np.poly1d(z)
            linfit = pz(self.normalised_wave)

            self.normalised_flux[ct] = np.array(flu[lop]) - np.array(linfit)
            self.normalised_flux[ct] = np.interp(self.vell,vell_temp,self.normalised_flux[ct])

        self._log(
            logging.INFO,
            f">> Max/Min velocities in map: {self.vell.min()} / {self.vell.max()}",
        )


        ##  JVHS 2019 August 6
        ## Add binning
        phase = np.linspace(0,2,self.nbins*2+1,endpoint=True) - 1./(self.nbins)/2.
        phase = np.concatenate((phase,[2.0+1./(self.nbins)/2.]))
        phase_dec = phase - np.floor(phase)
        #print(phase_dec)
        #rebin_trail(waver, flux, input_phase, nbins, delp, rebin_wave=None):
        trail,temp_phase = rebin_trail(self.vell, self.normalised_flux,
                            self.input_phase, self.nbins, self.delta_phase,
                            rebin_wave=None)

        self.pha = self.input_phase
        self.trsp = self.normalised_flux
        #print(">> SHAPES = ",self.pha.shape,self.trsp.shape)
        ## Phases of individual spectra
        #print("LAM_SIZE= {}, VELL_SIZE={}".format(self.normalised_wave.size,self.vell.size))
        f=open('dopin','w')
        f.write("{:8.0f}{:8.0f}{:13.2f}\n".format(self.pha.size,
                                        self.vell.size,
                                        self.lam0))
#        f.write(str(len(self.flux))+" "+str(self.nbins)+" "+str(self.lam0)+'\n')
        f.write("{:13.5f}{:8.0f}{:8.0f}    {:}\n".format(self.gama*1e5,0,0,
                                                self.base_dir+'/'+self.list))
        ctr = 0
        for pp in self.pha:
                if ctr <5:
                    f.write("{:13.6f}".format(pp))
                    ctr +=1
                else:
                    f.write("{:13.6f}\n".format(pp))
                    ctr=0
        f.write("\n{:8.0f}\n".format(1))
        ctr = 0

        for pp in np.ones(self.pha.size)*self.delta_phase:
                if ctr <5:
                    f.write("{:13.6f}".format(pp))
                    ctr +=1
                else:
                    f.write("{:13.6f}\n".format(pp))
                    ctr=0
        if ctr != 0: f.write("\n")
        ##
        ctr = 0
        for pp in self.vell:
                #print('velo size:',len(vell))
                if ctr <5:
                    f.write("{:13.5e}".format(pp*1e5))
                    ctr +=1
                else:
                    f.write("{:13.5e}\n".format(pp*1e5))
                    ctr=0
        if ctr != 0: f.write("\n")
        ctr = 0

        # Where we write the normalised flux
        for pp in np.array(self.trsp.T).flatten():
                if ctr <5:
                    f.write("{:13.5f}".format(pp))
                    ctr +=1
                else:
                    f.write("{:13.5f}\n".format(pp))
                    ctr=0
        if ctr != 0: f.write("\n")
        f.close()



        if xlim == None:
            rr = np.ones(self.normalised_wave.size,dtype='bool')
        else:
            rr = (self.normalised_wave > xlim[0]) & (self.normalised_wave < xlim[1])

        if rebin_wave == 0:
            waver = self.normalised_wave[rr]
        else:
            dw = (self.normalised_wave[rr][1] - self.normalised_wave[rr][0]) *\
                                                 rebin_wave
            self._log(logging.DEBUG, f"Rebin step {dw} ({dw/rebin_wave})")
            waver = np.arange(self.normalised_wave[rr][0],
                              self.normalised_wave[rr][-1],dw )
        """
        trail = np.zeros((waver.size,phase.size))

        tots = trail.copy()
        #print(phases.size)
        for i in range(self.input_phase.size):
            #print("spec phase = ",grid['phase'][i])
            dist = phase_dec - (self.input_phase[i]+self.delta_phase/2.)
            #print(dist)
            dist[np.abs(dist)>1./self.nbins] = 0.
            #print(dist/delpha)
            dist[dist>0] = 0.0
            #print(dist)
            weights = np.abs(dist)/(1./self.nbins)
            #print(weights)
            #print('---------------')
            dist = phase_dec - (self.input_phase[i]-self.delta_phase/2.)
            #print(dist)
            dist[np.abs(dist)>1./self.nbins] = 0.0
            #print(dist)
            dist[dist>0] = 0.0
            #print(dist/delpha)
            dist[np.abs(dist)>0] = 1.0 - (np.abs(dist[np.abs(dist)>0]))/(1./self.nbins)
            weights += dist
            #print(weights)
            temp = trail.copy().T

            for j in range(phase.size):
                if rebin_wave == 0:
                    temp[j] =  self.normalised_flux[i][rr] * weights[j]
                    temp[j] =  self.normalised_flux[i][rr] * weights[j]
                else:
                    temp[j] = np.interp(waver,wave[rr],
                              self.normalised_flux[i][rr]) * weights[j]
                    temp[j] = np.interp(waver,wave[rr],
                              self.normalised_flux[i][rr]) * weights[j]
            trail+=temp.T
            tots += weights
        trail /= tots
        """

        if plot_median:
            si = 0
            lo = 2
        else:
            si = 2
            lo = 0

        plt.figure('Trail',figsize=(6.57,8.57))
        plt.clf()
        if plot_median:
            ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=2)
            ax1.minorticks_on()
            if rebin_wave ==0:
                plt.plot(waver,np.nanmedian(self.normalised_flux,axis=0)[rr],
                    label='Median',color='#8e44ad')
            else:
                self._log(logging.DEBUG, str(dw))
                new_med = np.interp(waver,wave[rr],
                                    np.nanmedian(self.normalised_flux,axis=0)[rr])
                plt.plot(waver,np.nanmedian(self.normalised_flux,axis=0)[rr],
                    label='Median',color='k',alpha=1)
                plt.plot(waver,new_med,
                    label='Median',color='#8e44ad',alpha=1)
            plt.axhline(y=0,ls='--',color='r',alpha=0.7)
            ax1.set_xticklabels([])

            #plt.xlim(self.lam0 - self.delw, self.lam0 + self.delw)
            plt.ylim(-0.05,np.nanmax(np.nanmedian(self.normalised_flux,
                                                  axis=0)[rr])*1.1)

        ax2 = plt.subplot2grid((6, 1), (lo, 0), rowspan=4+si)
        ax2.minorticks_on()
        if vel_space:
            x1_lim = float(np.nanmin(self.vell))
            x2_lim = float(np.nanmax(self.vell))
            img = plt.imshow(
                trail.T, interpolation='nearest', cmap=plt.cm.binary, aspect='auto',
                origin='lower', extent=(x1_lim, x2_lim, phase[0], phase[-1] + 1/self.nbins)
            )
            plt.xlim(x1_lim, x2_lim)
            plt.xlabel('Velocity / km s$^{-1}$')
            plt.axvline(x=0.0, ls='--', color='DarkOrange')  # center of the line in velocity space
        else:
            x1_lim = float(np.nanmin(self.normalised_wave))
            x2_lim = float(np.nanmax(self.normalised_wave))
            img = plt.imshow(
                trail.T, interpolation='nearest', cmap=plt.cm.binary, aspect='auto',
                origin='lower', extent=(x1_lim, x2_lim, phase[0], phase[-1] + 1/self.nbins)
            )
            plt.xlim(self.lam0 - self.delw, self.lam0 + self.delw)
            plt.xlabel('Wavelength / $\\AA$')
            plt.axvline(x=self.lam0, ls='--', color='DarkOrange')
                  
        if two_orbits:
            lim_two = 2
        else:
            lim_two = 1
        plt.ylim(phase[0],lim_two+1/self.nbins/2.)
        plt.ylabel('Orbital Phase')
        plt.tight_layout(h_pad=0)



    def Syncdop(self,nri=0.9,ndi=0.7):
        '''
        Runs the fortran code dopp, using the output files from dopin
        Parameters
        ----------
        None

        Returns
        -------
        None.
        '''
        compile_flag = True
        while compile_flag == True:
            f=open('dop.in','w')
            f.write("{}     ih       type of likelihood function (ih=1 for chi-squared)\n".format(self.ih))
            f.write("{}     iw       iw=1 if error bars are to be read and used\n".format(self.iw))
            f.write("{}  {}    pb0,pb1    range of phases to be ignored\n".format(self.pb0,self.pb1))
            f.write("{}     ns       smearing width in default map\n".format(self.ns))
            f.write("{:.1e}     ac       accuracy of convergence\n".format(self.ac))
            f.write("{}     nim      max no of iterations\n".format(self.nim))
            f.write("{} {}  {}        al0,alf,nal   starting value, factor, max number of alfas\n".format(self.al0,self.alf,self.nal))
            f.write("{}     clim     'C-aim'\n".format(self.clim))
            f.write("{}     ipri     printout control for standard output channel (ipr=2 for full)\n".format(self.ipri))
            f.write("{}     norm     norm=1 for normalization to flat light curve\n".format(self.norm))
            f.write("{:2.1e}   {}  wid,af    width and amplitude central absorption fudge\n".format(self.wid,self.af))
            f.write("end of parameter input file")
            f.close()

            f=open('dopin')
            lines=f.readlines()
            f.close()
            # np == npp
            npp,nvp=int(lines[0].split()[0]),int(lines[0].split()[1])
            lines=[]

            f=open('emap_ori.par')
            lines=f.readlines()
            f.close()
            s=lines[0]
            npm=int(s[s.find('npm=')+len('npm='):s.rfind(',nvpm')])
            nvpm=int(s[s.find('nvpm=')+len('nvpm='):s.rfind(',nvm')])
            nvm=int(s[s.find('nvm=')+len('nvm='):s.rfind(')')])
            nvp = self.vell.size
            self._log(logging.DEBUG, f"nvp {nvp}")

            self._log(logging.DEBUG, f"Trail shape {self.trsp.shape}")
            nv0=int(self.overs*nvp)
            nv=max([nv0,int(min([1.5*nv0,npp/3.]))])
            self._log(logging.DEBUG, f"nv {nv} (nv0 {nv0})")
            if nv%2 == 1:
                nv+=1
            #nv=120
            nd = npm * nvpm
            nr = 0.8 * nv * nv
            nt = (nvpm * npm) + (nv * nvpm * 3) + (2 * npm * nv)
            prmsize = (0.9 * nv * nt) + (0.9 * nv * nt)

            self._log(
                logging.INFO,
                f"Estimated memory required {int(8*prmsize/1e6)} Mbytes",
            )
            self._log(logging.DEBUG, f"np={npp}; nvpm={nvp}, nvm={nv}")
            self._log(logging.DEBUG, f"ND {nd}")
            self._log(logging.DEBUG, f"NR {nr}")
            if nv != nvm or npp != npm or nvp !=nvpm:
                a1='      parameter (npm=%4d'% npp
                a2=',nvpm=%4d'%nvp
                a3=',nvm=%4d)'%nv
                a1=a1+a2+a3

                f=open('emap.par','w')
                f.write(a1+'\n')
                for i,lino in enumerate(lines[1:]):
                    #print(lino)
                    if i == 2:
                        tempo_str = '      parameter (nri={:.3f}*nvm*nt/nd,ndi={:.3f}*nvm*nt/nr)\n'.format(nri,ndi)
                        #aprint(tempo_str)
                        f.write(tempo_str)
                    elif lino !=3:
                        f.write(lino[:])
                    else:
                        f.write(lino[:]+')')
                f.close()
            self._log(logging.INFO, '>> Computing MEM tomogram <<')
            self._log(logging.INFO, '----------------------------')
            #os.system('gfortran -O -o dopp_input.txt dop.in dop.f clock.f')
            os.system('make dop.out')
            os.system('./dopp dopp.out')
            fo=open('dop.log')
            lines=fo.readlines()
            fo.close()
            #print(clim,rr)
            self._log(logging.INFO, '----------------------------')
            if lines[-1].split()[0] == 'projection':
                nri = float(lines[-1].split()[-1])/float(lines[-2].split()[-1])
                ndi = float(lines[-1].split()[-2])/float(lines[-2].split()[-2])
                self._log(logging.WARNING, '>> PROJECTION MATRIX TOO SMALL <<')
                self._log(logging.INFO, '>> Recomputing with values from Spruit:')
                self._log(logging.INFO, f'>> ndi = {ndi}, nri = {nri}')
            else:
                compile_flag=False
        clim = float(lines[-2].split()[-1])
        rr = float(lines[-2].split()[-2])
        if rr > clim:
            message = f">> NOT CONVERGED: Specified reduced chi^2 not reached: {rr} > {clim}"
            self._log(logging.ERROR, message)
            raise RuntimeError(message)
        else:
            self._log(logging.INFO, '>> Succesful Dopmap!')



    def Dopmap(self,dopout = 'dop.out',cmaps = cm.Greys_r,
               limits=None, colorbar=False, negative=False,remove_mean=False,
               corrx=0,corry=0, smooth=False):
        """
        Read output files from Henk Spruit's *.out and plot a Doppler map

        Parameters
        ----------
        dopout : str, Optional
            Name of output file to be read. Default, dop.out

        cmaps : cmap function,
            Color scheme to use for Doppler map
            - Default, cm.Greys_r

        limits : array,
            Normalised limtis e.g. [.8,1.1] for colour display. if None,
             automatic Limits will be generated
            - Default, None

        colorbar : bool,
            Generates an interactive colorbar. (unstable...)
            - Default, True

        remove_mean : bool,
            Remove an azimuthal mean of the map
            - Default, False

        corrx, corry : float, float
            Pixel correction for center of removal of azimuthal mean map

        smooth : bool,
            Apply Gaussian filter to map.
            - Default, False


        Returns
        -------
        cbar : object,
            Colorbar object for interactivity

        data : 2D-array,
            Data cube from Doppler map

        """
        if self.verbose:
            print(">> Reading {} file".format(dopout))
        fro=open(dopout,'r')
        lines=fro.readlines()
        fro.close()

        #READ ALL FILES
        nph,nvp,nv,w0,aa=int(lines[0].split()[0]),int(lines[0].split()[1]),int(lines[0].split()[2]),float(lines[0].split()[3]),float(lines[0].split()[4])
        gamma,abso,atm,dirin=float(lines[1].split()[0]),lines[1].split()[1],lines[1].split()[2],lines[1].split()[3]




        new = ''.join(lines[2:len(lines)])
        new = new.replace("E",'e')
        war = ''.join(new.splitlines()).split()
        #print(war)
        if self.verbose:
            print(">> Finished reading dop.out file")
        pha=np.array(war[:nph]).astype(float)/2.0/np.pi
        dum1=war[nph]
        dpha=np.array(war[nph+1:nph+1+nph]).astype(float)/2.0/np.pi
        last=nph+1+nph
        vp=np.array(war[last:last+nvp]).astype(float)
        dvp=vp[1]-vp[0]
        vp=vp-dvp/2.0
        last=last+nvp
        dm=np.array(war[last:last+nvp*nph]).astype(float)
        dm=dm.reshape(nvp,nph)
        last=last+nvp*nph


        #print(war[last])
        ih,iw,pb0,pb1,ns,ac,al,clim,norm,wid,af=int(war[last]),int(war[last+1]),float(war[last+2]),float(war[last+3]),int(war[last+4]),float(war[last+5]),float(war[last+6]),float(war[last+7]),int(war[last+8]),float(war[last+9]),float(war[last+10])
        nv,va,dd=int(war[last+11]),float(war[last+12]),war[last+13]
        last=last+14

        im=np.array(war[last:last+nv*nv]).astype(float)
        im=im.reshape(nv,nv)

        last=last+nv*nv
        ndum,dum2,dum3=int(war[last]),war[last+1],war[last+2]
        last=last+3
        dmr=np.array(war[last:last+nvp*nph]).astype(float)
        dmr=dmr.reshape(nvp,nph)
        last=last+nvp*nph
        ndum,dum4,dum2,dum3=int(war[last]),int(war[last+1]),war[last+2],war[last+3]
        last=last+4
        dpx=np.array(war[last:last+nv*nv]).astype(float)
        dpx=dpx.reshape(nv,nv)
        dpx = np.array(dpx)
        vp = np.array(vp)/1e5
        data = im

        data[data == 0.0] = np.nan

        new_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        #new_data = np.arcsinh(new_data)
        if limits == None:
            limits = [np.nanmax((new_data))*0.95,np.nanmax((new_data))*1.05]
        if self.verbose:
            print("Limits auto {:6.5f} {:6.5f}".format(np.nanmedian(data)*0.8,np.nanmedian(data)*1.2))
            print("Limits user {:6.5f} {:6.5f}".format(limits[0],limits[1]))
            print("Limits min={:6.5f}, max={:6.5f}".format(np.nanmin(data),np.nanmax(data)))
        # Here comes the plotting
        fig = plt.figure(num='Doppler Map',figsize=(8.57,8.57))
        plt.clf()
        ax = fig.add_subplot(111)
        ax.minorticks_on()
        ll = ~(np.isnan(data) )
        #data[~ll] = np.nan
        delvp = vp[1]-vp[0]
        #print(">>> VP",min(vp),max(vp),delvp)
        vpmin, vpmax = min(vp)-.5/delvp,max(vp)+.5/delvp,

        if smooth:
            interp_mode = 'gaussian'
        else:
            interp_mode = 'nearest'
        if remove_mean:
            rad_prof = radial_profile(data,[data[0].size/2-corrx,data[0].size/2-corry])
            meano = create_profile(data,rad_prof,[data[0].size/2-corrx,data[0].size/2-corry])
            qq = ~np.isnan(data - meano)
        if negative:
            if remove_mean:
                #print data[ll].max(),meano[qq].max()
                img = plt.imshow((data - meano)/(data - meano)[qq].max(),
                    interpolation=interp_mode, cmap=cmaps,aspect='equal',
                    origin='lower',extent=(vpmin, vpmax,vpmin, vpmax ),
                    vmin=limits[0],vmax=limits[1])
            else:
                img = plt.imshow(-(data)/data[ll].max(),
                    interpolation=interp_mode, cmap=cmaps,aspect='equal',
                    origin='lower',extent=(vpmin, vpmax,vpmin, vpmax),
                    vmin=-limits[1],vmax=-limits[0] )
        else:
            if remove_mean:
                #print data[ll].max(),meano[qq].max()
                img = plt.imshow((data - meano)/(data - meano)[qq].max(),
                    interpolation=interp_mode, cmap=cmaps,aspect='equal',
                    origin='lower',extent=(vpmin, vpmax,vpmin, vpmax),
                    vmin=limits[0],vmax=limits[1])
            else:
                #print(np.nanmin(data),np.nanmax(data))
                #new_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
                #new_data = data
                #print(np.nanmedian(data),np.nanstd(data))
                print("Limits min={:6.3f}, max={:6.3f}".format(np.nanmin(new_data),np.nanmax(new_data)))
                img = plt.imshow(new_data,interpolation=interp_mode,
                    cmap=cmaps,aspect='equal',origin='lower',
                    extent=(vpmin, vpmax,vpmin, vpmax ),
                    vmin=limits[0],vmax=limits[1] )

        axlimits=[min(vp), max(vp),min(vp), max(vp) ]
        plt.axis(axlimits)
        #plt.axvline(x=0.0,linestyle='--',color='white')

        plt.xlabel('V$_x$ / km s$^{-1}$')
        plt.ylabel('V$_y$ / km s$^{-1}$')
        plt.tight_layout()
        plt.show()
        if colorbar:
            cbar = plt.colorbar(format='%.1f',orientation='vertical',
                                fraction=0.046, pad=0.04)
            cbar.set_label('Normalised Flux')
            cbar.set_norm(MyNormalize(vmin=limits[0],vmax=limits[1],
                                                  stretch='log'))
            cbar = DraggableColorbar(cbar,img)
            cbar.connect()
        else:
            cbar=1

        '''
        if remove_mean:
            #print data.size/2
            rad_prof = radial_profile(data,[data[0].size/2,data[0].size/2])
            mean = create_profile(data,rad_prof,[data[0].size/2,data[0].size/2])
            ll = ~np.isnan(mean)
            fig = plt.figure('Mean')
            plt.clf()
            fig.add_subplot(211)
            plt.plot(rad_prof)
            fig.add_subplot(212)

            plt.show()
        '''
        return cbar,new_data


    def Reco(self, cmaps=plt.cm.binary, limits=None, colorbar=True):
        """
        Plot original and reconstructed trail spectra from Henk Spruit's *.out

        Parameters
        ----------
        cmaps : cmap function,
            Color scheme to use for Doppler map
            - Default, cm.Greys_r

        limits : array,
            Normalised limtis e.g. [.8,1.1] for colour display. if None,
             automatic Limits will be generated
            - Default, None

        colorbar : bool,
            Generates an interactive colorbar. (unstable...)
            - Default, True

        Returns
        -------
        cbar : object,
            Colorbar object for interactivity

        data : 2D-array,
            Data cube from reconstructed spectra

        """
        fro=open('dop.out','r')
        lines=fro.readlines()
        fro.close()

        #READ ALL FILES
        nph,nvp,nv,w0,aa=int(lines[0].split()[0]),int(lines[0].split()[1]),int(lines[0].split()[2]),float(lines[0].split()[3]),float(lines[0].split()[4])
        gamma,abso,atm,dirin=float(lines[1].split()[0]),lines[1].split()[1],lines[1].split()[2],lines[1].split()[3]

        #print(">> Reading dop.out file")
        #flag=0
        #for i in np.arange(3,len(lines),1):
        #    if flag==0:
        #        temp=lines[i-1]+lines[i]
        #        flag=1
        #    else:
        #        temp=temp+lines[i]
        #        war=temp.split()
        new = ''.join(lines[2:len(lines)])
        new = new.replace("E",'e')
        war = ''.join(new.splitlines()).split()
        #print(war)
        #print(">> Finished reading dop.out file")
        pha=np.array(war[:nph]).astype(float)/2.0/np.pi
        dum1=war[nph]
        dpha=np.array(war[nph+1:nph+1+nph]).astype(float)/2.0/np.pi
        last=nph+1+nph
        vp=np.array(war[last:last+nvp]).astype(float)
        dvp=vp[1]-vp[0]
        vp=vp-dvp/2.0
        last=last+nvp
        dm=np.array(war[last:last+nvp*nph]).astype(float)
        dm=dm.reshape(nvp,nph)
        last=last+nvp*nph


        #print(war[last])
        ih,iw,pb0,pb1,ns,ac,al,clim,norm,wid,af=int(war[last]),int(war[last+1]),float(war[last+2]),float(war[last+3]),int(war[last+4]),float(war[last+5]),float(war[last+6]),float(war[last+7]),int(war[last+8]),float(war[last+9]),float(war[last+10])
        nv,va,dd=int(war[last+11]),float(war[last+12]),war[last+13]
        last=last+14

        im=np.array(war[last:last+nv*nv]).astype(float)
        im=im.reshape(nv,nv)

        last=last+nv*nv
        ndum,dum2,dum3=int(war[last]),war[last+1],war[last+2]
        last=last+3
        dmr=np.array(war[last:last+nvp*nph]).astype(float)
        dmr=dmr.reshape(nvp,nph)
        last=last+nvp*nph
        ndum,dum4,dum2,dum3=int(war[last]),int(war[last+1]),war[last+2],war[last+3]
        last=last+4
        dpx=np.array(war[last:last+nv*nv]).astype(float)
        dpx=dpx.reshape(nv,nv)
        dpx = np.array(dpx)
        vp = np.array(vp)/1e5
        data = im

        data[data <= 0.0] = np.nan
        dpx[dpx <= 0.0] = np.nan
        #dmr[dmr <= 0.0] = np.nan
        #dm[dm <= 0.0] = np.nan
        #print(pha)
        #print(self.nbins)
        trail_dm,phase = rebin_trail(vp, dm.T, pha, self.nbins, self.delta_phase,
                                    rebin_wave=None)

        trail_dmr,phase = rebin_trail(vp, dmr.T, pha, self.nbins, self.delta_phase,
                                    rebin_wave=None)

        delvp = vp[1]-vp[0]
        x1_lim = min(vp)
        x2_lim = max(vp)
        #print(phase)
        if limits == None:
            limits = [np.median(dmr/np.nanmax(dmr))*0.8,
                      np.median(dmr/np.nanmax(dmr))*1.2]

        # Now lets do the plotting
        figor = plt.figure('Reconstruction',figsize=(10,8))
        plt.clf()
        ax1 = figor.add_subplot(121)
        print(np.nanmax(trail_dm))
        imgo = plt.imshow(trail_dm.T/np.nanmax(trail_dm),interpolation='nearest',
                    cmap=cmaps,aspect='auto',origin='upper',
                    extent=(x1_lim,x2_lim,phase[0],
                            phase[-1]+1/self.nbins),
                    vmin=limits[0], vmax=limits[1])

        ax1.set_xlabel('Velocity / km s$^{-1}$')
        ax1.set_ylabel('Orbital Phase')

        if colorbar:
            cbar2 = plt.colorbar(format='%.1e',orientation='vertical',
                                fraction=0.046, pad=0.04)
            cbar2.set_label('Normalised Flux')
            cbar2.set_norm(MyNormalize(vmin=np.median(dm/np.nanmax(dm))*0.8,
                                    vmax=np.median(dm/np.nanmax(dm))*1.1,
                                    stretch='linear'))
            cbar2 = DraggableColorbar(cbar2,imgo)
            cbar2.connect()
        else:
            cbar2=1
        ax2 = figor.add_subplot(122)
        print(np.nanmax(trail_dmr))
        imgo = plt.imshow(trail_dmr.T/np.nanmax(trail_dmr),interpolation='nearest',
                    cmap=cmaps,aspect='auto',origin='upper',
                    extent=(x1_lim,x2_lim,phase[0],
                            phase[-1]+1/self.nbins),
                    vmin=limits[0], vmax=limits[1])
        ax2.set_xlabel('Velocity / km s$^{-1}$')
        ax2.set_yticklabels([])
        plt.tight_layout(w_pad=0)
        if colorbar:
            cbar3 = plt.colorbar(format='%.1e',orientation='vertical',
                                fraction=0.046, pad=0.04)
            cbar3.set_label('Normalised Flux')
            cbar3.set_norm(MyNormalize(vmin=np.median(dmr/np.nanmax(dmr))*0.8,
                                    vmax=np.median(dmr/np.nanmax(dmr))*1.1,
                                    stretch='linear'))
            cbar3 = DraggableColorbar(cbar3,imgo)
            cbar3.connect()
        else:
            cbar3=1
        return cbar2,cbar3,dmr,dm

def rebin_trail(waver, flux, input_phase, nbins, delp, rebin_wave=None):
    """
    Re-bin a set of spectra (flux[i] sampled at waver) onto an orbital-phase grid.

    Parameters
    ----------
    waver : 1D array
        Common wavelength/velocity sampling for each flux[i].
    flux : 2D array, shape (nspec, nwave)
        Spectra to rebin in phase.
    input_phase : 1D array, shape (nspec,)
        Orbital phase (0..1) for each spectrum in `flux`.
    nbins : int
        Number of phase bins in [0,1).
    delp : float
        Effective phase width per spectrum for bin-weighting (typically ~1/nbins).
    rebin_wave : None
        (Kept for API compatibility; not used because `flux` is already on `waver`.)

    Returns
    -------
    trail : 2D array, shape (nwave, 2*nbins+2)
        Phase-binned trail spectrum covering ~[0,2] for plotting two orbits.
    phase : 1D array, shape (2*nbins+2,)
        Phase coordinate for `trail` columns.
    """
    # two orbits + half-bin shift, like upstream
    phase = np.linspace(0, 2, nbins * 2 + 1, endpoint=True) - 1.0 / nbins / 2.0
    phase = np.concatenate((phase, [2.0 + 1.0 / nbins / 2.0]))
    phase_dec = phase - np.floor(phase)

    nw = waver.size
    trail = np.zeros((nw, phase.size), dtype=float)
    tots = np.zeros(phase.size, dtype=float)

    inv_bin = 1.0 / nbins
    half = delp / 2.0

    for i in range(input_phase.size):
        # build trapezoidal weights for interval [phi - half, phi + half]
        phi = input_phase[i]
        wts = np.zeros_like(phase_dec)
        # right edge
        d = phase_dec - (phi + half)
        d[np.abs(d) > inv_bin] = 0.0
        d[d > 0] = 0.0
        wts += np.abs(d) / inv_bin
        # left edge
        d = phase_dec - (phi - half)
        d[np.abs(d) > inv_bin] = 0.0
        d[d > 0] = 0.0
        mask = (np.abs(d) > 0)
        wts[mask] += 1.0 - (np.abs(d[mask]) / inv_bin)

        # accumulate
        trail += np.outer(flux[i], wts)
        tots += wts

    # avoid division by zero
    tots[tots == 0] = 1.0
    trail /= tots
    return trail, phase


class DraggableColorbar(object):
    def __init__(self, cbar, mappable):
        self.cbar = cbar
        self.mappable = mappable
        self.press = None
        self.cycle = sorted([i for i in dir(plt.cm) if hasattr(getattr(plt.cm,i),'N')])
        self.index = self.cycle.index(cbar.get_cmap().name)

    def connect(self):
        """connect to all the events we need"""
        self.cidpress = self.cbar.patch.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.cbar.patch.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.cbar.patch.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.keypress = self.cbar.patch.figure.canvas.mpl_connect(
            'key_press_event', self.key_press)

    def on_press(self, event):
        """on button press we will see if the mouse is over us and store some data"""
        if event.inaxes != self.cbar.ax: return
        self.press = event.x, event.y

    def key_press(self, event):
        if event.key=='down':
            self.index += 1
        elif event.key=='up':
            self.index -= 1
        if self.index<0:
            self.index = len(self.cycle)
        elif self.index>=len(self.cycle):
            self.index = 0
        cmap = self.cycle[self.index]
        self.cbar.set_cmap(cmap)
        self.cbar.draw_all()
        self.mappable.set_cmap(cmap)
        #self.mappable.get_axes().set_title(cmap)
        self.cbar.patch.figure.canvas.draw()

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.cbar.ax: return
        xprev, yprev = self.press
        dx = event.x - xprev
        dy = event.y - yprev
        self.press = event.x,event.y
        #print 'x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f'%(x0, xpress, event.xdata, dx, x0+dx)
        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.button==1:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax -= (perc*scale)*np.sign(dy)
        elif event.button==3:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax += (perc*scale)*np.sign(dy)
        self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()


    def on_release(self, event):
        """on release we reset the press data"""
        self.press = None
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def disconnect(self):
        """disconnect all the stored connection ids"""
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidpress)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidrelease)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidmotion)


def radial_profile(data, center):
    """Calculate radial profile for Dopple map"""
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def create_profile(data,profile, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    mean = data*0.0 + 1.0
    for i in np.arange(r.max()):
    	#print i
    	ss = np.where(r == i)
    	#print profile[i]
    	mean[ss] = mean[ss] * profile[i]
    ll = ~np.isnan(mean)
    #print mean[ll]
    return mean

def stream(q,k1,porb,m1,inc,colors='k',both_lobes=False,title=True,label=None):
    """Calculate the Ballistic and Keplerian trajetories for a given binary
    system under Roche lobe geometry. This will be plotted directly in the
    Doppler tomogram
    """
    xl,yl,xi,yi,wout,wkout = stream_calculate(q,ni = 100,nj = 100)

    #print''
    azms=-70
    az=np.arctan(yi/xi)

    for i in np.arange(len(az)):
        if xi[i] < 0.0:
            az[i]=az[i] + np.pi
            #print az[i],az[i]*180/np.pi
    az=az*180/np.pi
    i=0
    for j in np.arange(az.size):
        #print az[j]
        i=i+1
        if az[j] < azms:
            break

    vxi = np.real(wout)
    vyi = np.imag(wout)
    vkxi = np.real(wkout)
    vkyi = np.imag(wkout)
    #print az[0],i
    porb=24*3600*porb           # in seconds
    omega=2*np.pi/porb
    gg=6.667e-8                     # Gravitational Constant, cgs
    msun=1.989e33
    cm=q/(1.0+q)
    nvp=1000
    vxp,vyp,vkxp,vkyp,rr=[],[],[],[],[]

    xl=xl-cm
    inc=np.pi*inc/180.0
    a=(gg*m1*msun*(1.0+q))**(1./3)/omega**(2./3)     # Orbital Separation
    vfs=1e5
    vs=omega*a/vfs
    rd=0
    r=1
    vxi=vxi[:i]
    vyi=vyi[:i]
    vkxi=vkxi[:i]
    vkyi=vkyi[:i]
    az=az[:i]
    si=np.sin(inc)
    vx=vxi*si*vs
    vy=vyi*si*vs
    vkx=vkxi*si*vs
    vky=vkyi*si*vs
    xl=xl*vs*si
    yl=yl*vs*si
    npl=len(az)
    #fig = plt.figure(num='Doppler Map')
    #ax = fig.add_subplot(111)
    #dist = np.sqrt((vx - vkx)**2 + (vy - vky)**2)
    #dist = np.abs(vy-vky)
    #print np.abs(vy-vky)[:12],dist[:12]
    #ss = np.where( dist == min(dist) )[0]
    #print vy[-1],vky[-1],vx[-1],vx[-1]
    #print vx[ss],vy[ss]
    plt.plot(vx[:],vy[:],color=colors,marker='')
    plt.plot(vkx[:],vky[:],color=colors,marker='')
    plt.plot(yl[int(yl.size/4):3*int(yl.size/4)],xl[int(xl.size/4):3*int(xl.size/4)],color=colors)
    if title: plt.title(r'$i$='+str(inc/np.pi*180.)[:5]+', M$_1$='+str(m1)+' M$_{\odot}$, $q$='+str(q)+', P$_{orb}$='+str(porb/3600.)[:4]+' hr')
    ## 0,0 systemic velocity, km/s
    vy1 = cm * vs * si
    plt.plot(0.,0.,'x',ms = 9,c = colors,alpha=0.3)
    ## 0,-K1 systemic velocity, km/s
    plt.plot(0.,-vy1,'+',ms = 10,c = colors,alpha=0.7)
    plt.plot(0.,(1.0-cm)*vs*si,'+',ms = 10,c = colors,alpha=0.7)

    if both_lobes:
        plt.plot(np.concatenate((yl[3*int(xl.size/4):],yl[:int(yl.size/4)]),axis=0),
        np.concatenate((xl[3*int(xl.size/4):],xl[:int(yl.size/4)]),axis=0),color=colors,ls='--')
        #plt.plot(yl,xl,color=colors)
    else:
        plt.plot(yl[int(yl.size/4):3*int(yl.size/4)],xl[int(xl.size/4):3*int(xl.size/4)],color=colors)

    if label != None: plt.text(0.12, 0.1,label, ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()

def stream_calculate(qm,ni = 100,nj = 100):
    '''
    calculates Roche lobes and integrates path of stream from L1
    '''
    nmax = 10000
    xout = np.zeros(nmax)
    yout = np.zeros(nmax)
    rout = np.zeros(nmax)
    wout = np.zeros(nmax,dtype=complex)
    wkout = np.zeros(nmax,dtype=complex)
    if np.abs(qm - 1.) < 1e-4: qm = 1e-4
    rd = 0.1
    if qm <= 0.0:
        print ('Mass ratio <= 0. Does not compute. Will exit.')
        return
    rl1 = rlq1(qm)

    x,y = lobes(qm,rl1,ni,nj)
    ## Center of mass relative to M1
    cm = qm / (1.0 + qm)
    ## Coordinates of M1 and M2
    z1=-cm
    z2=1-cm
    wm1=np.conj(complex(0.,-cm))
    ## Start at L1-eps with v=0
    eps=1e-3
    z = complex(rl1 - cm -eps,0.)
    w = 0
    zp,wp = eqmot(z,w,z1,z2,qm)
    t=0
    dt=1e-4
    isa=0
    it=0
    r=1
    ist=0
    ph=0.
    phmax=6
    while it < nmax and ph < phmax:
        dz,dw = intrk(z,w,dt,z1,z2,qm)
        z=z+dz
        w=w+dw
        t=t+dt
        if np.abs(dz)/np.abs(z) > 0.02: dt=dt/2.
        if np.abs(dz)/np.abs(z) < 0.005: dt=2.*dt

        dph= -np.imag(z*np.conj(z-dz))/np.abs(z)/np.abs(z-dz)
        ph=ph+dph
        ##velocity in inertial frame
        ##change by Guillaume
        wi=w+complex(0,1.)*z
        ## unit vector normal to kepler orbit
        rold=r
        r=np.abs(z-z1)

        if ist == 0 and rold < r:
            ist=1
            rmin=rold

        # kepler velocity of circular orbit in potential of M1, rel. to M1
        vk=1.0/np.sqrt(r*(1.0+qm))
        # unit vector in r
        no = np.conj(z-z1)/r
        wk = -vk*no*complex(0.,1.)
        # same but rel. to cm, this is velocity in inertial frame
        wk = wk+wm1
        # velocity normal to disk edge, in rotating frame
        dot = no * w
        # velocity parallel to disk edge
        par = np.imag(no*w)
        # reflected velocity
        wr = w - 2.0*dot*no
        #        write(*,'(f8.4,1p9e11.3)')t,z,w,wk,wr,r
        xout[it] = np.real(z)+cm
        yout[it] = -np.imag(z)
        rout[it] = np.sqrt(xout[it]**2+yout[it]**2)
        # change by Guillaume
        wout[it]= wi
        wkout[it]=np.conj(wk)
        if it > 0:
            xo=xout[it]
            yo=yout[it]
            phi=np.arctan(yo/xo)
            if rout[it] < rd and rout[it-1] >  rd:
            ## write(*,'('' r,x,y,phi,vs,vk,dot,par'',8f8.3)')
            ##    rout(it),x,y,phi,real(w),vk,dot,par
            ## write(*,'('' w,no'',4f8.3)')w,no
                xo=xout[it-1]
                yo=yout[it-1]
                phi=np.arctan(yo/xo)
            # write(*,'('' r,x,y,phi'',4f8.3)')rout(it-1),x,y,phi

        if isa == 0 and yout[it] < 0:
            isa=1
            ra=np.abs(z-z1)
            wc=np.conj(w)+complex(0.,1.)*np.conj(z-z1)
            ang=np.abs(np.imag((z-z1)*np.conj(wc)))
        it+=1
    return x,y,xout,yout,wout,wkout

def rlq1(q):
    '''
    Calulates roche lobe radius.
    '''
    if np.abs(1.0 - q) < 1e-4:
        rlq = 0.5
        return rlq
    rl = 0
    rn = 1.0 - q
    while np.abs(rl/rn-1.) > 1e-4:
        rl=rn
        f=q/(1.-rl)**2-1./rl**2+(1.+q)*rl-q
        fa=2.*q/(1-rl)**3+2/rl**3+(1.+q)
        rn=rl-f/fa
    rlq1 = rn
    return rlq1



def lobes(q,rs,ni,nj):
    '''
    SUBROUTINE
    '''
    r = np.zeros((ni,nj))
    ch = np.zeros(ni)
    ps = np.zeros(nj)
    x  = np.zeros(ni)
    y  = np.zeros(nj)
    x2 = np.zeros(ni)
    y2 = np.zeros(nj)
    nc = ni
    nop = nj

    r,ch,ps = surface(q,rs,nc,nop,r,ch,ps)
    j=0
    for i in np.arange(nc):
        x[i] = 1.0 -r[i,j]*np.cos(ch[i])
        y[i] = -r[i,j] * np.sin(ch[i])

    r,ch,ps = surface(1./q,1.-rs,nc,nop,r,ch,ps)
    j=0
    for i in np.arange(nc):
        x2[i] = r[i,j] * np.cos(ch[i])
        y2[i] = r[i,j] * np.sin(ch[i])
    xt = np.concatenate((x2[::-1],x,x[::-1],x2))
    yt = np.concatenate((y2[::-1],-y,y[::-1],-y2))
    return xt,yt


def pot(q,x,y,z):
    '''
    FUNCTION
    Roche potential. coordinates centered on M2,
    z along rotation axis, x toward M1
    pr is gradient in radius from M2
    first transform to polar coordinates w/r rotation axis
    '''
    r = np.sqrt(x*x+y*y+z*z)
    if (r == 0):
        print ('r=0 in pot')
        stop
    rh = np.sqrt(x*x+y*y)
    st=rh/r
    if rh == 0:
        cf=1
    else:
        cf=x/rh

    r2 = 1. / (1. + q)
    r1 = np.sqrt(1.0+r**2-2.0*r*cf*st)
    pot=-1.0/r-1.0/q/r1-0.5*(1.0/q+1.0)*(r2**2+(r*st)**2-2.0*r2*r*cf*st)
    pr=1.0/r**2+1.0/q/(r1**3)*(r-cf*st)-0.5*(1.0/q+1)*2.0*(r*st*st-r2*cf*st)
    return pot,pr


def surface(q,rs,nc,nop,r,ch,ps):
    '''
    SUBROUTINE
    Roche surface around M2, coordinates on surface are ch, ps.
    ch: polar angle from direction to M1; ps: corresponding azimuth, counting
    from orbital plane.
    q:mass ratio, rs: radius of surface at point facing M1
    nc, np: number of chi's, psi's.
    output:
    r(nf,nt): radius. ch, ps: chi and psi arrays
    '''
    r = np.zeros((100,100))
    chi = [],ps
    dc = np.pi/nc
    ch[0] = 0
    for i in np.arange(nc-1)+1:
        ch[i] = float((i-1.0))*np.pi/(nc-1.)
    ps[0] = 0
    for j in np.arange(nop-1)+1:
        ps[i] = float((j-1.0))*2.*np.pi/nop
    rs1 = 1.0 -rs
    fs,pr = pot(q,rs1,0.0,0.0)

    ## max no of iterations
    im = 20

    for i in np.arange(nop):
        cp = np.cos(ps[i])
        sp = np.sin(ps[i])
        rx = (1.0 - dc) * rs1
        r[0,i] = rs1

        for k in np.arange(nc-1)+1:
            x  = np.cos(ch[k])
            sc = np.sin(ch[k])
            y  = sc * cp
            z  = sc * sp
            j  = 0
            f  = 1
            while (j < im ) and np.abs(f - fs) > 1e-4 or j == 0:
                j = j+1
                r1 = rx
                f,pr = pot(q,r1*x,r1*y,r1*z)
                rx = r1 - (f - fs)/pr
                if rx > rs1: rx = rs1
            if j >= im:
                print( 'No conv in surf',k,i,ch[k],ps[i])
                stop

            r[k,i] = rx

    return r,ch,ps


def eqmot(z,w,z1,z2,qm):
    zr1 = z-z1
    zr2 = z-z2
    ## c change by Guillaume : - sign in Coriolis
    wp = -(qm*zr2/(np.abs(zr2))**3 + zr1/(np.abs(zr1))**3)/(1.0+qm) - complex(0.,2.)*w + z
    zp = w
    return zp,wp


def intrk(z,w,dt,z1,z2,qm):
    zx=z
    wx=w
    zp,wp = eqmot(zx,wx,z1,z2,qm)
    hz0=zp*dt
    hw0=wp*dt
    zx=z+hz0/2.
    wx=w+hw0/2.
    zp,wp = eqmot(zx,wx,z1,z2,qm)
    hz1=zp*dt
    hw1=wp*dt
    zx=z+hz1/2.
    wx=w+hw1/2.
    zp,wp = eqmot(zx,wx,z1,z2,qm)
    hz2=zp*dt
    hw2=wp*dt
    zx=z+hz2
    wx=w+hw2
    zp,wp = eqmot(zx,wx,z1,z2,qm)
    hz3=zp*dt
    hw3=wp*dt
    dz=(hz0+2*hz1+2*hz2+hz3)/6.
    dw=(hw0+2*hw1+2*hw2+hw3)/6.
    return dz,dw

def xy(r,phi):
  return r*np.cos(phi), r*np.sin(phi)

def resonance(j,k,k1,q,porb,m1):
    '''Plots iso-velocity for resonance, in the general
    notation of Whitehurst & King (19XX). Eq. taken from Warner 1995
    page 206-207.
    '''
    k1 *= 1e5
    porb *= 24. * 3600.
    a_1 = k1 / q / (2.0*np.pi) * porb
    a_2 = k1 / (2.0*np.pi) * porb
    a = a_1 + a_2

    r = (j-k)**(2./3.) / j**(2./3.) / (1.0 + q)**(1./3.) * a
    r_circ = a * 0.60/(1. + q)
    velo = np.sqrt(6.67e-08 * m1 *1.989e+33 / r)/1e5
    velo_circ = np.sqrt(6.67e-08 * m1 *1.989e+33 / r_circ)/1e5
    phis=np.arange(0,6.28,0.01)

    #print r/69950000000.0,velo
    fig = plt.figure(num='Doppler Map')
    xx,yy = xy(velo,phis)
    plt.plot( xx,yy-k1/1e5,c='k',ls=':')
    #ax = fig.add_subplot(111)
    #circ = plt.Circle((0-k1,0),velo,ls='--',color='k',fill=True)


    #print velo_circ,velo
    xx_circ,yy_circ = xy(velo_circ,phis)
    plt.plot( xx_circ,yy_circ-k1/1e5,c='k',ls='-',lw=2)

    plt.draw()


def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipythonp.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def test_data(
    destination: Optional[Union[Path, str]] = None,
    overwrite: bool = False,
) -> List[Path]:
    """Copy the bundled test data into *destination* (defaults to CWD)."""

    dest_dir = Path.cwd() if destination is None else Path(destination)
    copied = copy_test_data(dest_dir, overwrite=overwrite)

    LOGGER.info("Copied %d test data files to %s", len(copied), dest_dir)
    return copied

class MyNormalize(Normalize):
    '''
    # The Normalize class is largely based on code provided by Sarah Graves.
    A Normalize class for imshow that allows different stretching functions
    for astronomical images.
    '''

    def __init__(self, stretch='linear', exponent=5, vmid=None, vmin=None,
                 vmax=None, clip=False):
        '''
        Initalize an APLpyNormalize instance.

        Optional Keyword Arguments:

            *vmin*: [ None | float ]
                Minimum pixel value to use for the scaling.

            *vmax*: [ None | float ]
                Maximum pixel value to use for the scaling.

            *stretch*: [ 'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power' ]
                The stretch function to use (default is 'linear').

            *vmid*: [ None | float ]
                Mid-pixel value used for the log and arcsinh stretches. If
                set to None, a default value is picked.

            *exponent*: [ float ]
                if self.stretch is set to 'power', this is the exponent to use.

            *clip*: [ True | False ]
                If clip is True and the given value falls outside the range,
                the returned value will be 0 or 1, whichever is closer.
        '''

        if vmax < vmin:
            raise Exception("vmax should be larger than vmin")

        # Call original initalization routine
        Normalize.__init__(self, vmin=vmin, vmax=vmax, clip=clip)

        # Save parameters
        self.stretch = stretch
        self.exponent = exponent

        if stretch == 'power' and np.equal(self.exponent, None):
            raise Exception("For stretch=='power', an exponent should be specified")

        if np.equal(vmid, None):
            if stretch == 'log':
                if vmin > 0:
                    self.midpoint = vmax / vmin
                else:
                    raise Exception("When using a log stretch, if vmin < 0, then vmid has to be specified")
            elif stretch == 'arcsinh':
                self.midpoint = -1. / 30.
            else:
                self.midpoint = None
        else:
            if stretch == 'log':
                if vmin < vmid:
                    raise Exception("When using a log stretch, vmin should be larger than vmid")
                self.midpoint = (vmax - vmid) / (vmin - vmid)
            elif stretch == 'arcsinh':
                self.midpoint = (vmid - vmin) / (vmax - vmin)
            else:
                self.midpoint = None

    def __call__(self, value, clip=None):

        #read in parameters
        method = self.stretch
        exponent = self.exponent
        midpoint = self.midpoint

        # ORIGINAL MATPLOTLIB CODE

        if clip is None:
            clip = self.clip

        if cbook.iterable(value):
            vtype = 'array'
            val = ma.asarray(value).astype(float)
        else:
            vtype = 'scalar'
            val = ma.array([value]).astype(float)

        self.autoscale_None(val)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            return 0.0 * val
        else:
            if clip:
                mask = ma.getmask(val)
                val = ma.array(np.clip(val.filled(vmax), vmin, vmax),
                                mask=mask)
            result = (val - vmin) * (1.0 / (vmax - vmin))

            # CUSTOM APLPY CODE

            # Keep track of negative values
            negative = result < 0.

            if self.stretch == 'linear':

                pass

            elif self.stretch == 'log':

                result = ma.log10(result * (self.midpoint - 1.) + 1.) \
                       / ma.log10(self.midpoint)

            elif self.stretch == 'sqrt':

                result = ma.sqrt(result)

            elif self.stretch == 'arcsinh':

                result = ma.arcsinh(result / self.midpoint) \
                       / ma.arcsinh(1. / self.midpoint)

            elif self.stretch == 'power':

                result = ma.power(result, exponent)

            else:

                raise Exception("Unknown stretch in APLpyNormalize: %s" %
                                self.stretch)

            # Now set previously negative values to 0, as these are
            # different from true NaN values in the FITS image
            result[negative] = -np.inf

        if vtype == 'scalar':
            result = result[0]

        return result

    def inverse(self, value):

        # ORIGINAL MATPLOTLIB CODE

        if not self.scaled():
            raise ValueError("Not invertible until scaled")

        vmin, vmax = self.vmin, self.vmax

        # CUSTOM APLPY CODE

        if cbook.iterable(value):
            val = ma.asarray(value)
        else:
            val = value

        if self.stretch == 'linear':

            pass

        elif self.stretch == 'log':

            val = (ma.power(10., val * ma.log10(self.midpoint)) - 1.) / (self.midpoint - 1.)

        elif self.stretch == 'sqrt':

            val = val * val

        elif self.stretch == 'arcsinh':

            val = self.midpoint * \
                  ma.sinh(val * ma.arcsinh(1. / self.midpoint))

        elif self.stretch == 'power':

            val = ma.power(val, (1. / self.exponent))

        else:

            raise Exception("Unknown stretch in APLpyNormalize: %s" %
                            self.stretch)

        return vmin + val * (vmax - vmin)

