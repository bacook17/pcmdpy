# driver.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from scipy.stats import poisson
from scipy.misc import logsumexp
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
        self.num_sims = 0
        self.num_calls = 0
        
    def initialize_data(self, pcmd, bins=None, charlie_err=False, **kwargs):
        if bins is None:
            magbins = np.arange(-12, 15.6, 0.05)
            colorbins = np.arange(-1.5, 4.6, 0.05)
            bins = [magbins]
            for _ in range(1, self.n_filters):
                bins.append(colorbins)
            bins = np.array(bins)
        self.hess_bins = bins
        self.n_data = pcmd.shape[1]

        #fit a 2D gaussian to the points
        means = np.mean(pcmd, axis=1)
        cov = np.cov(pcmd)

        self.gaussian_data = multivariate_normal(mean=means, cov=cov)

        #compute the mean magnitudes
        mags = np.copy(pcmd)
        for i in range(1, self.n_filters):
            mags[i] += mags[0]
        mag_factor = -0.4 * np.log(10)  # convert from base 10 to base e
        weights = float(1) / mags.shape[1]  # evenly weight each pixel
        self.mean_mags_data = logsumexp(mag_factor*mags, b=weights, axis=1)
        
        counts, hess, err = utils.make_hess(pcmd, self.hess_bins,
                                            charlie_err=charlie_err)
        self.counts_data = counts
        self.hess_data = hess
        self.err_data = err
        self._data_init = True
        self.pcmd_data = pcmd

    def loglike(self, pcmd, use_gaussian=True, charlie_err=False, like_mode=0, **kwargs):
        try:
            assert(self._data_init)
        except AssertionError:
            print('Cannot evaluate, as data has not been initialized (use driver.initialize_data)')
            return

        self.num_calls += 1
        #print(self.num_calls)

        #fit a multi-D gaussian to the points
        means = np.mean(pcmd, axis=1)
        cov = np.cov(pcmd)

        normal_model = multivariate_normal(mean=means, cov=cov)
        normal_term = np.sum(normal_model.logpdf(self.pcmd_data.T))
        
        if like_mode == 0:
            #ONLY use the normal approximation
            log_like = normal_term
            return log_like
        
        #compute the mean magnitudes
        mags = np.copy(pcmd)
        mags[0] += mags[1]
        mag_factor = -0.4 * np.log(10) #convert from base 10 to base e
        weights = float(1) / mags.shape[1] #evenly weight each pixel
        mean_mags_model = logsumexp(mag_factor*mags, b=weights, axis=1)
        
        counts_model, hess_model, err_model = utils.make_hess(pcmd, self.hess_bins, charlie_err=charlie_err)
        n_model = pcmd.shape[1]

        #the fraction of bins populated by both the model and the data
        #NOT including the "everything else" bin
        frac_common = np.logical_and(counts_model, self.counts_data)[:-1].mean()
        
        if use_gaussian:
            #add error in quadrature
            combined_var = (self.err_data**2. + err_model**2.) 
            hess_diff = (self.hess_data - hess_model)
            log_like = -np.sum(hess_diff**2. / combined_var)

        else:
            #Poisson Likelihood
            counts_model += 1e-3 #get NANs if model has zeros
            counts_model *= float(self.n_data) / n_model #normalize to same number of pixels as data
            log_like = np.sum(poisson.logpmf(self.counts_data, counts_model))

        if like_mode==1:
            return log_like

        elif like_mode==2:
            #add terms relating to mean magnitude and color
            mag_data = self.mean_mags_data[0]
            mag_model = mean_mags_model[0]
            color_data = mag_data - self.mean_mags_data[1]
            color_model = mag_model - mean_mags_model[1]
            var_mag = 0.01**2
            var_color = 0.05**2
            log_like -= (mag_data - mag_model)**2. / (2.*var_mag)
            log_like -= (color_data - color_model)**2. / (2.*var_color)
            return log_like

        elif like_mode==3:
            #combine the two terms, weighted by the amount of overlap
            log_like = frac_common*log_like +  (1 - frac_common) * normal_term
            return log_like

        else:
            #default to like_mode == 1
            return log_like

    def simulate(self, gal_model, im_scale, psf=True, fixed_seed=False, **kwargs):
        IMF, mags = self.iso_model.model_galaxy(gal_model, **kwargs)
        fluxes = np.array([f.mag_to_counts(m) for f,m in zip(self.filters, mags)])
        dust_frac, dust_mean, dust_std = gal_model.dust_model.get_params()
        images = gpu_utils.draw_image(IMF, fluxes, im_scale, self.filters,
                                      dust_frac, dust_mean, dust_std,
                                      gpu=self.gpu_on, fixed_seed=fixed_seed,
                                      **kwargs)
        #images += 1e-10

        if psf:
            images = np.array([f.psf_convolve(im, **kwargs) for f,im in zip(self.filters,images)])

        mags = np.array([f.counts_to_mag(im.flatten(), **kwargs) for f,im in zip(self.filters, images)])
        
        pcmd = utils.make_pcmd(mags)
        
        self.num_sims += 1
        return pcmd, images
