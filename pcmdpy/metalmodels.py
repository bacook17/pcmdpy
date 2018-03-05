# metalmodels.py
# Ben Cook (bcook@cfa.harvard.edu)

__all__ = ['SingleFeH', 'NormMDF']

import numpy as np
from scipy.stats import norm
from pcmdpy import utils


class FeHModel:
    default_fehs = np.arange(-4., 0.6, 0.5)
    _num_feh_bins = len(default_fehs) - 1

    def __init__(self):
        pass

    def get_vals(self):
        return self.fehs, self.weights
    

class SingleFeH(FeHModel):

    _param_names = ['logfeh']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3.0, 0.5]]
    
    def __init__(self, feh_params):
        logfeh = feh_params[0]
        self.fehs, self.weights = np.array([logfeh]), np.array([1.])


class NormMDF(FeHModel):
    
    _param_names = ['feh_mean', 'feh_sig']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3.0, 0.5], [0.05, 1.0]]

    def __init__(self, feh_params):
        feh_mean, feh_sig = feh_params
        if feh_sig <= 0.:
            return np.array([feh_mean]), np.array([1.])
        lower_feh = feh_mean - 1.0
        upper_feh = feh_mean + 1.0
        in_range = np.logical_and(self.default_fehs >= lower_feh,
                                  self.default_fehs <= upper_feh)
        utils.my_assert(np.sum(in_range) > 0,
                        "logfeh0 out of range (+/- 1 dex) of given fehs.")
        self.fehs = self.default_fehs[in_range]
        self.weights = norm.pdf(self.fehs, loc=feh_mean,
                                scale=feh_sig)
        # this can happen if feh_sig is much smaller than default_fehs spacing
        if np.isclose(np.sum(self.weights), 0.):
            self.fehs = np.array([feh_mean])
            self.weights = np.array([1.])
        else:
            self.weights /= np.sum(self.weights)

