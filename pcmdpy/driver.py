# driver.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from pcmdpy import utils
from pcmdpy import gpu_utils
import warnings

from scipy.stats import multivariate_normal


class Driver:

    def __init__(self, iso_model, gpu=True, **kwargs):
        self.iso_model = iso_model
        self.filters = iso_model.filters
        self.n_filters = len(self.filters)

        if gpu:
            if gpu_utils._GPU_AVAIL:
                self.gpu_on = True
            else:
                warnings.warn('GPU acceleration not available. Continuing without.', RuntimeWarning)
                self.gpu_on = False
        else:
            #No GPU acceleration
            self.gpu_on = False
        #No data has been initialized
        self._data_init = False
        
    def initialize_data(self, pcmd, bins=None, **kwargs):
        if bins is None:
            magbins = np.arange(-12, 15.6, 0.05)
            colorbins = np.arange(-1.5, 4.6, 0.05)
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

    def loglike(self, pcmd, like_mode=2, **kwargs):
        assert self._data_init, ('Cannot evaluate, as data has not been '
                                 'initialized (use driver.initialize_data)')

        # fit a multi-D gaussian to the points
        means = np.mean(pcmd, axis=1)
        cov = np.cov(pcmd)

        normal_model = multivariate_normal(mean=means, cov=cov)
        normal_term = np.sum(normal_model.logpdf(self.pcmd_data.T))
        
        # ONLY use the normal approximation
        if like_mode == 0:
            log_like = normal_term
            return log_like

        # compute the mean magnitudes
        mean_mags_model = utils.mean_mags(pcmd)
        mean_pcmd_model = utils.make_pcmd(mean_mags_model)

        # compute hess diagram
        counts_model, hess_model, err_model = utils.make_hess(pcmd,
                                                              self.hess_bins)

        # add terms relating to mean magnitude and colors
        var_mag = 0.01**2
        var_color = 0.05**2
        var_pcmd = np.append([var_mag],
                             [var_color for _ in range(1, self.n_filters)])
        mean_term = - np.sum((mean_pcmd_model - self.mean_pcmd_data)**2 / (2*var_pcmd))

        if like_mode == 1:
            n_model = pcmd.shape[1]
            root_nn = np.sqrt(n_model * self.n_data)
            term1 = np.log(root_nn + self.n_data * counts_model)
            term2 = np.log(root_nn + n_model * self.counts_data)
            log_like = -np.sum((term1 - term2)**2.)
            log_like += mean_term
            return log_like
            
        # add error in quadrature
        if like_mode == 2:
            combined_var = (self.err_data**2. + err_model**2.)
            hess_diff = (self.hess_data - hess_model)
            log_like = -np.sum(hess_diff**2. / (2*combined_var))
            log_like += mean_term
            return log_like
        else:
            raise NotImplementedError('like_mode only defined for [0, 1, 2]')
            
    def simulate(self, gal_model, im_scale, psf=True, fixed_seed=False,
                 shot_noise=False, sky_noise=None, **kwargs):
        IMF, mags = self.iso_model.model_galaxy(gal_model, **kwargs)
        fluxes = np.array([f.mag_to_counts(m) for f,m in zip(self.filters, mags)])
        dust_frac, dust_mean, dust_std = gal_model.dust_model.get_params()
        images = gpu_utils.draw_image(IMF, fluxes, im_scale, self.filters,
                                      dust_frac, dust_mean, dust_std,
                                      gpu=self.gpu_on, fixed_seed=fixed_seed,
                                      **kwargs)
        if shot_noise:
            images = np.random.poisson(images)
        if sky_noise is not None:
            images += np.random.poisson(lam=sky_noise,
                                        size=(im_scale, im_scale, self.n_filters)).T.reshape((self.n_filters, im_scale, im_scale))
        if psf:
            images = np.array([f.psf_convolve(im, **kwargs) for f,im in zip(self.filters,images)])

        mags = np.array([f.counts_to_mag(im.flatten(), **kwargs) for f,im in zip(self.filters, images)])
        
        pcmd = utils.make_pcmd(mags)
        
        return pcmd, images
