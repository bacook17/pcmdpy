# driver.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from ..utils import utils
from . import gpu_utils
import warnings
from scipy.stats import multivariate_normal, poisson, norm
from sys import stderr


class Driver:

    def __init__(self, iso_model, max_Nim=1024, gpu=True, **kwargs):
        self.iso_model = iso_model
        self.fixed_states = None
        self.random_states = None
        self.max_Nim = max_Nim
        if self.iso_model is None:
            self.filters = None
            self.n_filters = 2
        else:
            self.filters = iso_model.filters
            self.n_filters = len(self.filters)
        if gpu:
            if gpu_utils._GPU_AVAIL:
                self.gpu_on = True
                self.update_states()
            else:
                stderr.write('GPU acceleration not available. Continuing without.')
                self.gpu_on = False
        else:
            # No GPU acceleration
            self.gpu_on = False
        # No data has been initialized
        self._data_init = False

    def update_states(self):
        if not self.gpu_on:
            return
        stderr.write('Please wait while the GPU states are initialized')
        self.fixed_states = gpu_utils.XORWOWStatesArray(self.max_Nim*self.max_Nim,
                                                        fixed_seed=True)
        self.random_states = gpu_utils.XORWOWStatesArray(self.max_Nim*self.max_Nim,
                                                         fixed_seed=False)

    def __del__(self):
        del self.fixed_states
        del self.random_states
        
    def initialize_data(self, pcmd, bins=None, **kwargs):
        if bins is None:
            magbins = np.arange(-12, 45, 0.05)
            colorbins = np.arange(-1.5, 5.6, 0.05)
            bins = [magbins]
            for _ in range(1, self.n_filters):
                bins.append(colorbins)
            bins = np.array(bins)
        self.hess_bins = bins
        self.n_data = pcmd.shape[1]

        # fit a 2D gaussian to the points
        means = np.mean(pcmd, axis=1)
        cov = np.cov(pcmd)

        self.gaussian_data = multivariate_normal(mean=means, cov=cov)

        # compute the mean magnitudes
        self.mean_mags_data = utils.mean_mags(pcmd)
        self.mean_pcmd_data = utils.make_pcmd(self.mean_mags_data)
        
        counts, hess, err = utils.make_hess(pcmd, self.hess_bins)
        self.counts_data = counts
        self.hess_data = hess
        self.err_data = err
        self._data_init = True
        self.pcmd_data = pcmd

    def loglike_map(self, pcmd, like_mode=2, signed=True):
        counts_model, hess_model, err_model = utils.make_hess(
            pcmd, self.hess_bins)
        combined_var = (self.err_data**2. + err_model**2.)
        hess_diff = (hess_model - self.hess_data)
        if like_mode == 1:  # Poisson model
            hess_model[(counts_model == 0) & (self.counts_data > 0)] = 0.1 / pcmd.shape[1]  # add 0.1 fake counts in each empty model bin
            loglike = poisson.logpmf(self.counts_data,
                                     mu=(hess_model * self.n_data))
        elif like_mode == 2:  # Gaussian model (no variance term included)
            loglike = -1. * hess_diff**2 / (2.*combined_var)
        elif like_mode == 3:  # Gaussian model (variance included, downweights high-count bins)
            loglike = norm.logpdf(hess_model,
                                  loc=self.hess_data,
                                  scale=np.sqrt(combined_var))
        else:
            raise NotImplementedError('like_mode only defined for [1,2,3]')
        if signed:
            return loglike * np.sign(hess_diff)
        else:
            return loglike

    def loglike(self, pcmd, like_mode=2, **kwargs):
        assert self._data_init, ('Cannot evaluate, as data has not been '
                                 'initialized (use driver.initialize_data)')

        # compute the mean magnitudes
        mean_mags_model = utils.mean_mags(pcmd)
        mean_pcmd_model = utils.make_pcmd(mean_mags_model)

        # add terms relating to mean magnitude and colors
        var_mag = 0.05**2
        var_color = 0.05**2
        var_pcmd = np.append([var_mag],
                             [var_color for _ in range(1, self.n_filters)])
        mean_term = -1. * np.sum((mean_pcmd_model - self.mean_pcmd_data)**2 /
                                 (2*var_pcmd))

        # ONLY use the normal approximation
        if like_mode == 0:
            # fit a multi-D gaussian to the points
            means = np.mean(pcmd, axis=1)
            cov = np.cov(pcmd)

            normal_model = multivariate_normal(mean=means, cov=cov)
            normal_term = np.sum(normal_model.logpdf(self.pcmd_data.T))
            log_like = normal_term

        elif like_mode in [1, 2, 3]:
            llmap = self.loglike_map(pcmd, like_mode=like_mode, signed=False)
            log_like = mean_term + np.sum(llmap)
        else:
            raise NotImplementedError('like_mode only defined for [0,1,2,3]')

        return log_like
            
    def simulate(self, gal_model, Nim, psf=True, fixed_seed=False,
                 shot_noise=True, sky_noise=None, downsample=5,
                 fudge_mag=0.0, gain=1.0, dark_per_sec=0.0,
                 mag_system='vega', lum_cut=np.inf, **kwargs):
        if self.gpu_on:
            if Nim > self.max_Nim:
                self.max_Nim = Nim
                self.update_states()
            if fixed_seed:
                temp_states = self.fixed_states.copy()
                states = temp_states.gpudata
            else:
                states = self.random_states.gpudata
        else:
            states = None
        IMF, mags = self.iso_model.model_galaxy(
            gal_model, downsample=downsample, mag_system=mag_system,
            return_mass=False, lum_cut=lum_cut)
        fluxes = np.array([f.mag_to_counts(m) for f, m in zip(self.filters,
                                                              mags)])
        dust_frac, dust_mean, dust_std = gal_model.dust_model.get_props()

        images = gpu_utils.draw_image(IMF, fluxes, Nim, self.filters,
                                      dust_frac, dust_mean, dust_std,
                                      fudge_mag=fudge_mag,
                                      d_states=states, gpu=self.gpu_on,
                                      fixed_seed=fixed_seed,
                                      **kwargs)
        if psf:
            images = np.array([f.psf_convolve(im, **kwargs) for f,im in zip(self.filters,images)])
        if sky_noise is not None:
            # add sky level (in counts) to each image
            for im, sky in zip(images, sky_noise):
                im += sky
        if shot_noise:
            if fixed_seed:
                np.random.seed(0)
            else:
                np.random.seed()
            # shot noise occurs at DN level, not e- level. Account for gain
            dark_vals = np.array([f._exposure for f in self.filters]) * dark_per_sec
            images += dark_vals
            images = np.random.poisson(images / gain).astype(np.float32) * gain
            images[images <= 0.] = 1e-3  # avoid nan issues by adding 0.001 counts
            images -= dark_vals#
        mags = np.array([f.counts_to_mag(im.flatten(), **kwargs) for f,im in zip(self.filters, images)])
        pcmd = utils.make_pcmd(mags)
        
        return pcmd, images
