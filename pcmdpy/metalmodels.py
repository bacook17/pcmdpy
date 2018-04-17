# metalmodels.py
# Ben Cook (bcook@cfa.harvard.edu)

__all__ = ['SingleFeH', 'NormMDF', 'FixedWidthNormMDF']

import numpy as np
from scipy.stats import norm
from pcmdpy import utils


class _FeHModel:
    default_fehs = np.arange(-4., 0.6, 0.5)
    _num_feh_bins = len(default_fehs) - 1

    def __init__(self):
        pass

    def get_vals(self):
        return self.fehs, self.weights
    
    @classmethod
    def compute_mdf(cls, feh_mean, feh_sig):
        if feh_sig <= 0.:
            return np.array([feh_mean]), np.array([1.])
        else:
            lower_feh = feh_mean - 1.0
            upper_feh = feh_mean + 1.0
            in_range = np.logical_and(cls.default_fehs >= lower_feh,
                                      cls.default_fehs <= upper_feh)
            utils.my_assert(np.sum(in_range) > 0,
                            "logfeh0 out of range (+/- 1 dex) of given fehs.")
            fehs = cls.default_fehs[in_range]
            weights = norm.pdf(fehs, loc=feh_mean,
                               scale=feh_sig)
            # this can happen if feh_sig is much smaller than default_fehs spacing
            if np.isclose(np.sum(weights), 0.):
                fehs = np.array([feh_mean])
                weights = np.array([1.])
            else:
                weights /= np.sum(weights)
            return fehs, weights


class SingleFeH(_FeHModel):

    _param_names = ['logfeh']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3.0, 0.5]]
    
    def __init__(self):
        pass

    def set_params(self, feh_params):
        logfeh = feh_params[0]
        self.fehs, self.weights = np.array([logfeh]), np.array([1.])


class NormMDF(_FeHModel):
    
    _param_names = ['logfeh_mean', 'logfeh_std']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3.0, 0.5], [0.05, 1.0]]

    def __init__(self):
        pass

    def set_params(self, feh_params):
        self.feh_mean, self.feh_sig = feh_params
        self.fehs, self.weights = self.compute_mdf(self.feh_mean, self.feh_sig)


class FixedWidthNormMDF(NormMDF):

    _param_names = ['logfeh_mean']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3.0, 0.5]]

    def __init__(self, sig):
        self.feh_sig = sig

    def set_params(self, feh_params):
        self.feh_mean = feh_params[0]
        self.fehs, self.weights = self.compute_mdf(self.feh_mean, self.feh_sig)
