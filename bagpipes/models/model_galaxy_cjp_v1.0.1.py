from __future__ import print_function, division, absolute_import

import numpy as np
import warnings
import spectres

from copy import deepcopy

from .. import utils
from .. import config
from .. import filters
from .. import plotting

from .stellar_model import stellar
from .dust_emission_model import dust_emission
from .dust_attenuation_model import dust_attenuation
from .nebular_model import nebular
from .igm_model import igm
from .damping_wing import damping_wing
from .agn_model import agn
from .star_formation_history import star_formation_history
from ..input.spectral_indices import measure_index


class model_galaxy(object):
    """ Builds model galaxy spectra and calculates predictions for
    spectroscopic and photometric observables.

    Parameters
    ----------

    model_components : dict
        A dictionary containing information about the model you wish to
        generate.

    filt_list : list - optional
        A list of paths to filter curve files, which should contain a
        column of wavelengths in angstroms followed by a column of
        transmitted fraction values. Only required if photometric output
        is desired.

    spec_wavs : array - optional
        An array of wavelengths at which spectral fluxes should be
        returned. Only required of spectroscopic output is desired.

    spec_units : str - optional
        The units the output spectrum will be returned in. Default is
        "ergscma" for ergs per second per centimetre squared per
        angstrom, can also be set to "mujy" for microjanskys.

    phot_units : str - optional
        The units the output spectrum will be returned in. Default is
        "ergscma" for ergs per second per centimetre squared per
        angstrom, can also be set to "mujy" for microjanskys.

    index_list : list - optional
        list of dicts containining definitions for spectral indices.
    """

    def __init__(self, model_components, filt_list=None, spec_wavs=None,
                 spec_units="ergscma", phot_units="ergscma", index_list=None):

        if (spec_wavs is not None) and (index_list is not None):
            raise ValueError("Cannot specify both spec_wavs and index_list.")

        if model_components["redshift"] > config.max_redshift:
            raise ValueError("Bagpipes attempted to create a model with too "
                             "high redshift. Please increase max_redshift in "
                             "bagpipes/config.py before making this model.")

        self.spec_wavs = spec_wavs
        self.filt_list = filt_list
        self.spec_units = spec_units
        self.phot_units = phot_units
        self.index_list = index_list

        if self.index_list is not None:
            self.spec_wavs = self._get_index_spec_wavs(model_components)

        # Create a filter_set object to manage the filter curves.
        if filt_list is not None:
            self.filter_set = filters.filter_set(filt_list)

        # Calculate the optimal wavelength sampling for the model.
        self.wavelengths = self._get_wavelength_sampling()

        # Resample the filter curves onto wavelengths.
        if filt_list is not None:
            self.filter_set.resample_filter_curves(self.wavelengths)

        # Set up a filter_set for calculating rest-frame UVJ magnitudes.
        uvj_filt_list = np.loadtxt(utils.install_dir
                                   + "/filters/UVJ.filt_list", dtype="str")

        self.uvj_filter_set = filters.filter_set(uvj_filt_list)
        self.uvj_filter_set.resample_filter_curves(self.wavelengths)

        # Create relevant physical models.
        self.sfh = star_formation_history(model_components)
        self.stellar = stellar(self.wavelengths)
        ''' CJP removed ''' 
        self.igm = igm(self.wavelengths)
        ''' ''' 
        self.nebular = False
        self.dust_atten = False
        self.dust_emission = False
        self.agn = False

        ''' CJP added '''
        self.damping_wing = False
        if "damping_wing" in list(model_components):
            self.damping_wing = damping_wing(self.wavelengths,model_components["redshift"], model_components["damping_wing"])
        ''' ''' 
        
        if "nebular" in list(model_components):
            if "velshift" not in model_components["nebular"]:
                model_components["nebular"]["velshift"] = 0.

            self.nebular = nebular(self.wavelengths,
                                   model_components["nebular"]["velshift"])

            if "metallicity" in list(model_components["nebular"]):
                self.neb_sfh = star_formation_history(model_components)

        if "dust" in list(model_components):
            self.dust_emission = dust_emission(self.wavelengths)
            self.dust_atten = dust_attenuation(self.wavelengths,
                                               model_components["dust"])

        if "agn" in list(model_components):
            self.agn = agn(self.wavelengths)

        self.update(model_components)

    def _get_wavelength_sampling(self):
        """ Calculate the optimal wavelength sampling for the model
        given the required resolution values specified in the config
        file. The way this is done is key to the speed of the code. """

        max_z = config.max_redshift

        if self.spec_wavs is None:
            self.max_wavs = [(self.filter_set.min_phot_wav
                              / (1.+max_z)),
                             1.01*self.filter_set.max_phot_wav, 10**8]

            self.R = [config.R_other, config.R_phot, config.R_other]

        elif self.filt_list is None:
            self.max_wavs = [self.spec_wavs[0]/(1.+max_z),
                             self.spec_wavs[-1], 10**8]

            self.R = [config.R_other, config.R_spec, config.R_other]

        else:
            if (self.spec_wavs[0] > self.filter_set.min_phot_wav
                    and self.spec_wavs[-1] < self.filter_set.max_phot_wav):

                self.max_wavs = [self.filter_set.min_phot_wav/(1.+max_z),
                                 self.spec_wavs[0]/(1.+max_z),
                                 self.spec_wavs[-1],
                                 self.filter_set.max_phot_wav, 10**8]

                self.R = [config.R_other, config.R_phot, config.R_spec,
                          config.R_phot, config.R_other]

            elif (self.spec_wavs[0] < self.filter_set.min_phot_wav
                  and self.spec_wavs[-1] < self.filter_set.max_phot_wav):

                self.max_wavs = [self.spec_wavs[0]/(1.+max_z),
                                 self.spec_wavs[-1],
                                 self.filter_set.max_phot_wav, 10**8]

                self.R = [config.R_other, config.R_spec,
                          config.R_phot, config.R_other]

            if (self.spec_wavs[0] > self.filter_set.min_phot_wav
                    and self.spec_wavs[-1] > self.filter_set.max_phot_wav):

                self.max_wavs = [self.filter_set.min_phot_wav/(1.+max_z),
                                 self.spec_wavs[0]/(1.+max_z),
                                 self.spec_wavs[-1], 10**8]

                self.R = [config.R_other, config.R_phot,
                          config.R_spec, config.R_other]

        # Generate the desired wavelength sampling.
        x = [1.]

        for i in range(len(self.R)):
            if i == len(self.R)-1 or self.R[i] > self.R[i+1]:
                while x[-1] < self.max_wavs[i]:
                    x.append(x[-1]*(1.+0.5/self.R[i]))

            else:
                while x[-1]*(1.+0.5/self.R[i]) < self.max_wavs[i]:
                    x.append(x[-1]*(1.+0.5/self.R[i]))

        return np.array(x)

    def _get_R_curve_wav_sampling(self, oversample=4):
        """ Calculate wavelength sampling for the model to be resampled
        onto in order to apply variable spectral broadening. Only used
        if a resolution curve is supplied in model_components. Has to
        be re-run when a model is updated as it depends on redshift.

        Parameters
        ----------

        oversample : float
            Number of spectral samples per full width at half maximum.
        """

        R_curve = self.model_comp["R_curve"]
        x = [0.95*self.spec_wavs[0]]

        while x[-1] < 1.05*self.spec_wavs[-1]:
            R_val = np.interp(x[-1], R_curve[:, 0], R_curve[:, 1])
            dwav = x[-1]/R_val/oversample
            x.append(x[-1] + dwav)

        return np.array(x)

    def _get_index_spec_wavs(self, model_components):
        """ Generate an appropriate spec_wavs array for covering the
        spectral indices specified in index_list. """

        min = 9.9*10**99
        max = 0.

        indiv_ind = []
        for j in range(len(self.index_list)):
            if self.index_list[j]["type"] == "composite":
                n = 1
                while "component" + str(n) in list(self.index_list[j]):
                    indiv_ind.append(self.index_list[j]["component" + str(n)])
                    n += 1

            else:
                indiv_ind.append(self.index_list[j])

        for i in range(len(indiv_ind)):
            wavs = np.array(indiv_ind[i]["continuum"]).flatten()

            if "feature" in list(indiv_ind[i]):
                extra_wavs = np.array(indiv_ind[i]["feature"])
                wavs = np.concatenate((wavs, extra_wavs))

            min = np.min([min, np.min(wavs)])
            max = np.max([max, np.max(wavs)])

        min = np.round(0.95*min, 2)
        max = np.round(1.05*max, 2)
        sampling = np.round(np.mean([min, max])/5000., 2)

        return np.arange(min, max*(1. + config.max_redshift), sampling)

    def update(self, model_components):
        """ Update the model outputs to reflect new parameter values in
        the model_components dictionary. Note that only the changing of
        numerical values is supported. """

        self.model_comp = model_components
        self.sfh.update(model_components)
        if self.dust_atten:
            self.dust_atten.update(model_components["dust"])

        # If the SFH is unphysical do not caclulate the full spectrum
        if self.sfh.unphysical:
            warnings.warn("The requested model includes stars which formed "
                          "before the Big Bang, no spectrum generated.",
                          RuntimeWarning)

            self.spectrum_full = np.zeros_like(self.wavelengths)
            self.uvj = np.zeros(3)

        else:
            self._calculate_full_spectrum(model_components)

        if self.spec_wavs is not None:
            self._calculate_spectrum(model_components)

        # Add any AGN component:
        if self.agn:
            self.agn.update(self.model_comp["agn"])
            agn_spec = self.agn.spectrum
            agn_spec *= self.igm.trans(self.model_comp["redshift"])

            ''' CJP added ''' 
            if self.damping_wing :
                agn_spec *= self.damping_wing.attenuation

            ''' '''
                
            self.spectrum_full += agn_spec/(1. + self.model_comp["redshift"])

            if self.spec_wavs is not None:
                zplus1 = (self.model_comp["redshift"] + 1.)
                agn_interp = np.interp(self.spec_wavs, self.wavelengths*zplus1,
                                       agn_spec/zplus1, left=0, right=0)

                self.spectrum[:, 1] += agn_interp

        if self.filt_list is not None:
            self._calculate_photometry(model_components["redshift"])

        if not self.sfh.unphysical:
            self._calculate_uvj_mags()

        # Deal with any spectral index calculations.
        if self.index_list is not None:
            self.index_names = [ind["name"] for ind in self.index_list]

            self.indices = np.zeros(len(self.index_list))
            for i in range(self.indices.shape[0]):
                self.indices[i] = measure_index(self.index_list[i],
                                                self.spectrum,
                                                model_components["redshift"])

    def _calculate_full_spectrum(self, model_comp):
        """ This method combines the models for the various emission
        and absorption processes to generate the internal full galaxy
        spectrum held within the class. The _calculate_photometry and
        _calculate_spectrum methods generate observables using this
        internal full spectrum. """

        """ CJP added stellar and nebular spectra separately """

        
        t_bc = 0.01
        if "t_bc" in list(model_comp):
            t_bc = model_comp["t_bc"]

        spectrum_bc, spectrum = self.stellar.spectrum(self.sfh.ceh.grid, t_bc)
        em_lines = np.zeros(config.line_wavs.shape)

        if self.nebular:
            ''' added by CJP: '''
            stellar_only = np.copy(spectrum)
            stellar_only_bc = np.copy(spectrum_bc)
            nebular = np.zeros_like(spectrum)
            '''''' 
 
            grid = np.copy(self.sfh.ceh.grid)

            if "metallicity" in list(model_comp["nebular"]):
                nebular_metallicity = model_comp["nebular"]["metallicity"]
                neb_comp = deepcopy(model_comp)
                for comp in list(neb_comp):
                    if isinstance(neb_comp[comp], dict):
                        neb_comp[comp]["metallicity"] = nebular_metallicity

                self.neb_sfh.update(neb_comp)
                grid = self.neb_sfh.ceh.grid

            em_lines += self.nebular.line_fluxes(grid, t_bc,
                                                 model_comp["nebular"]["logU"])

            # All stellar emission below 912A goes into nebular emission
            spectrum_bc[self.wavelengths < 912.] = 0.
            spectrum_bc += self.nebular.spectrum(grid, t_bc,
                                                 model_comp["nebular"]["logU"])

            ''' CJP added '''
            stellar_only_bc[self.wavelengths < 912.] = 0.
            nebular_bc =  self.nebular.spectrum(grid, t_bc,
                                                    model_comp["nebular"]["logU"])
            ''''''
 
        # Add attenuation due to stellar birth clouds.
        if self.dust_atten:
            dust_flux = 0.  # Total attenuated flux for energy balance.

            # Add extra attenuation to birth clouds.
            eta = 1.
            if "eta" in list(model_comp["dust"]):
                eta = model_comp["dust"]["eta"]
                bc_Av_reduced = (eta - 1.)*model_comp["dust"]["Av"]
                bc_trans_red = 10**(-bc_Av_reduced*self.dust_atten.A_cont/2.5)
                spectrum_bc_dust = spectrum_bc*bc_trans_red
                dust_flux += np.trapz(spectrum_bc - spectrum_bc_dust,
                                      x=self.wavelengths)

                spectrum_bc = spectrum_bc_dust

                ''' CJP added '''
                if self.nebular : 
                    stellar_only_bc *= bc_trans_red
                    nebular_bc *= bc_trans_red
                ''''''

            # Attenuate emission line fluxes.
            bc_Av = eta*model_comp["dust"]["Av"]
            em_lines *= 10**(-bc_Av*self.dust_atten.A_line/2.5)

        spectrum += spectrum_bc  # Add birth cloud spectrum to spectrum.

        # Add attenuation due to the diffuse ISM.
        if self.dust_atten:
            trans = 10**(-model_comp["dust"]["Av"]*self.dust_atten.A_cont/2.5)
            dust_spectrum = spectrum*trans
            dust_flux += np.trapz(spectrum - dust_spectrum, x=self.wavelengths)

            spectrum = dust_spectrum
            self.spectrum_bc = spectrum_bc*trans

            # Add dust emission.
            qpah, umin, gamma = 2., 1., 0.01
            if "qpah" in list(model_comp["dust"]):
                qpah = model_comp["dust"]["qpah"]

            if "umin" in list(model_comp["dust"]):
                umin = model_comp["dust"]["umin"]

            if "gamma" in list(model_comp["dust"]):
                gamma = model_comp["dust"]["gamma"]

            spectrum += dust_flux*self.dust_emission.spectrum(qpah, umin,
                                                              gamma)

            ''' CJP added ''' 
            if self.nebular :
                nebular *= trans
                stellar_only *= trans
            ''''''

        if self.damping_wing :
            atten = self.damping_wing.attenuation
            damped_spectrum = spectrum * atten
            spectrum = damped_spectrum
            self.spectrum_bc = spectrum_bc*atten

            if self.nebular :
                nebular *= trans
                stellar_only *= trans
            ''''''
        spectrum *= self.igm.trans(model_comp["redshift"])

        if self.dust_atten:
            self.spectrum_bc *= self.igm.trans(model_comp["redshift"])

        # Convert from luminosity to observed flux at redshift z.
        self.lum_flux = 1.
        if model_comp["redshift"] > 0.:
            ldist_cm = 3.086*10**24*np.interp(model_comp["redshift"],
                                              utils.z_array, utils.ldist_at_z,
                                              left=0, right=0)

            self.lum_flux = 4*np.pi*ldist_cm**2

        spectrum /= self.lum_flux*(1. + model_comp["redshift"])

        if self.dust_atten:
            self.spectrum_bc /= self.lum_flux*(1. + model_comp["redshift"])

        em_lines /= self.lum_flux

        # convert to erg/s/A/cm^2, or erg/s/A if redshift = 0.
        spectrum *= 3.826*10**33

        if self.dust_atten:
            self.spectrum_bc *= 3.826*10**33

        em_lines *= 3.826*10**33

        self.line_fluxes = dict(zip(config.line_names, em_lines))

        self.spectrum_full = spectrum

        ''' CJP added : '''
        if self.nebular :
            nebular *= self.igm.trans(model_comp["redshift"])
            nebular /= self.lum_flux*(1. + model_comp["redshift"])
            nebular *= 3.826*10**33
            self.nebular_full = nebular

            stellar_only *= self.igm.trans(model_comp["redshift"])
            stellar_only /= self.lum_flux*(1. + model_comp["redshift"])
            stellar_only *= 3.826*10**33
            self.stellar_only = stellar_only

        ''''''

        
    def _calculate_photometry(self, redshift, uvj=False):
        """ This method generates predictions for observed photometry.
        It resamples filter curves onto observed frame wavelengths and
        integrates over them to calculate photometric fluxes. """

        if self.phot_units == "mujy" or uvj:
            unit_conv = "cgs_to_mujy"

        else:
            unit_conv = None

        if uvj:
            phot = self.uvj_filter_set.get_photometry(self.spectrum_full,
                                                      redshift,
                                                      unit_conv=unit_conv)

        else:
            phot = self.filter_set.get_photometry(self.spectrum_full,
                                                  redshift,
                                                  unit_conv=unit_conv)

        if uvj:
            return phot

        self.photometry = phot

    def _calculate_spectrum(self, model_comp):
        """ This method generates predictions for observed spectroscopy.
        It optionally applies a Gaussian velocity dispersion then
        resamples onto the specified set of observed wavelengths. """

        zplusone = model_comp["redshift"] + 1.

        if "veldisp" in list(model_comp):
            vres = 3*10**5/config.R_spec/2.
            sigma_pix = model_comp["veldisp"]/vres
            k_size = 4*int(sigma_pix+1)
            x_kernel_pix = np.arange(-k_size, k_size+1)

            kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
            kernel /= np.trapz(kernel)  # Explicitly normalise kernel

            spectrum = np.convolve(self.spectrum_full, kernel, mode="valid")
            redshifted_wavs = zplusone*self.wavelengths[k_size:-k_size]

        else:
            spectrum = self.spectrum_full
            redshifted_wavs = zplusone*self.wavelengths

        if "R_curve" in list(model_comp):
            oversample = 4  # Number of samples per FWHM at resolution R
            new_wavs = self._get_R_curve_wav_sampling(oversample=oversample)

            # spectrum = np.interp(new_wavs, redshifted_wavs, spectrum)
            spectrum = spectres.spectres(new_wavs, redshifted_wavs,
                                         spectrum, fill=0)
            redshifted_wavs = new_wavs

            sigma_pix = oversample/2.35  # sigma width of kernel in pixels
            k_size = 4*int(sigma_pix+1)
            x_kernel_pix = np.arange(-k_size, k_size+1)

            kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
            kernel /= np.trapz(kernel)  # Explicitly normalise kernel

            # Disperse non-uniformly sampled spectrum
            spectrum = np.convolve(spectrum, kernel, mode="valid")
            redshifted_wavs = redshifted_wavs[k_size:-k_size]

        # Converted to using spectres in response to issue with interp,
        # see https://github.com/ACCarnall/bagpipes/issues/15
        # fluxes = np.interp(self.spec_wavs, redshifted_wavs,
        #                    spectrum, left=0, right=0)

        fluxes = spectres.spectres(self.spec_wavs, redshifted_wavs,
                                   spectrum, fill=0)

        if self.spec_units == "mujy":
            fluxes /= ((10**-29*2.9979*10**18/self.spec_wavs**2))

        self.spectrum = np.c_[self.spec_wavs, fluxes]

    def _calculate_uvj_mags(self):
        """ Obtain (unnormalised) rest-frame UVJ magnitudes. """

        self.uvj = -2.5*np.log10(self._calculate_photometry(0., uvj=True))

    def plot(self, show=True):
        return plotting.plot_model_galaxy(self, show=show)

    def plot_full_spectrum(self, show=True):
        return plotting.plot_full_spectrum(self, show=show)
