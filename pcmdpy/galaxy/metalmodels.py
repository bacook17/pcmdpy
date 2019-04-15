# metalmodels.py
# Ben Cook (bcook@cfa.harvard.edu)

__all__ = ['BaseMetalModel', 'SingleFeH', 'NormMDF', 'FixedWidthNormMDF',
           'ClosedBoxMDF',
           'get_metal_model', 'all_metal_models']

import numpy as np
from scipy.stats import norm


def get_metal_model(name, *args, **kwargs):
    if name.lower() == 'single':
        return SingleFeH(*args, **kwargs)
    elif name.lower() == 'fixedwidth':
        return FixedWidthNormMDF(*args, **kwargs)
    elif name.lower() == 'norm':
        return NormMDF(*args, **kwargs)
    elif 'closed' in name.lower():
        return ClosedBoxMDF(*args, **kwargs)
    else:
        raise NotImplementedError(
            "given name {} not an acceptable metal model. Choose one of:\n"
            "{}".format(name.lower(), ['single', 'fixedwidth', 'norm', 'closedbox']))


class BaseMetalModel:
    default_fehs = np.arange(-2.0, 0.51, 0.25)

    def __init__(self):
        pass

    def get_vals(self):
        return self.fehs, self.weights

    @property
    def _num_fehs(self):
        return len(self.fehs)
    

class SingleFeH(BaseMetalModel):

    _param_names = ['logfeh']
    _fancy_names = ['[Fe/H]']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3.0, 0.5]]

    feh_mean = None
    feh_sig = 0.0
    
    def __init__(self, initial_params=None):
        if initial_params is None:
            initial_params = np.array([0.])
        self.set_params(initial_params)

    @property
    def _params(self):
        return np.array([self.feh_mean], dtype=float)
            
    def set_params(self, feh_params):
        if isinstance(feh_params, float) or isinstance(feh_params, int):
            feh_params = [feh_params]
        assert len(feh_params) == self._num_params, (
            "feh_params for SingleFeH is length {:d}, "
            "should be length {:d}".format(len(feh_params), self._num_params))
        self.feh_mean = feh_params[0]
        self.fehs, self.weights = np.array([self.feh_mean]), np.array([1.])

    def copy(self):
        return SingleFeH(initial_params=[self.feh_mean])


class NormMDF(BaseMetalModel):
    
    _param_names = ['logfeh_mean', 'logfeh_std']
    _fancy_names = ['[Fe/H]', r'$\sigma$([Fe/H])']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3.0, 0.5], [0.05, 1.0]]

    feh_mean = None
    feh_sig = None

    def __init__(self, initial_params=None):
        if initial_params is None:
            initial_params = np.array([0., 0.2])
        self.set_params(initial_params)

    @property
    def _params(self):
        return np.array([self.feh_mean, self.feh_sig], dtype=float)

    def set_params(self, feh_params):
        assert len(feh_params) == self._num_params, (
            "feh_params for NormMDF is length {:d}, "
            "should be length {:d}".format(len(feh_params), self._num_params))
        self.feh_mean, self.feh_sig = feh_params

    def _compute_mdf(self, etol=1e-2):
        if self.feh_sig <= 0.:
            return np.array([self.feh_mean]), np.array([1.])
        else:
            lower_feh = self.feh_mean - 1.0
            upper_feh = self.feh_mean + 1.0
            in_range = np.logical_and(self.default_fehs >= lower_feh,
                                      self.default_fehs <= upper_feh)
            assert (np.sum(in_range) > 0), (
                "logfeh0 out of range (+/- 1 dex) of given fehs.")
            fehs = self.default_fehs[in_range]
            weights = norm.pdf(fehs, loc=self.feh_mean,
                               scale=self.feh_sig)
            # this can happen if feh_sig is much smaller than default_fehs spacing
            if np.isclose(np.sum(weights), 0.):
                fehs = np.array([self.feh_mean])
                weights = np.array([1.])
            else:
                weights /= np.sum(weights)
                # remove bins with negligible weight
                too_small = (weights < etol)
                fehs = fehs[~too_small]
                weights = weights[~too_small]
                weights /= np.sum(weights)
            assert len(fehs) == len(weights)
            if len(fehs) <= 1:
                return np.array([self.feh_mean]), np.array([1.])
            else:
                return fehs, weights

    @property
    def fehs(self):
        return self._compute_mdf()[0]

    @property
    def weights(self):
        return self._compute_mdf()[1]

    def copy(self):
        return NormMDF(initial_params=[self.feh_mean, self.feh_sig])


class FixedWidthNormMDF(NormMDF):

    _param_names = ['logfeh_mean']
    _fancy_names = ['[Fe/H]']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3.0, 0.5]]

    feh_mean = None
    feh_sig = None
    
    def __init__(self, sig=0.2, initial_params=None):
        self.feh_sig = sig
        if initial_params is None:
            initial_params = np.array([0.])
        self.set_params(initial_params)

    @property
    def _params(self):
        return np.array([self.feh_mean], dtype=float)

    def set_params(self, feh_params):
        if isinstance(feh_params, float) or isinstance(feh_params, int):
            feh_params = [feh_params]
        assert len(feh_params) == self._num_params, (
            "feh_params for FixedWidthNormMDF is length {:d}, "
            "should be length {:d}".format(len(feh_params), self._num_params))
        self.feh_mean = feh_params[0]

    def copy(self):
        return FixedWidthNormMDF(initial_params=[self.feh_mean], sig=self.feh_sig)

    
class ClosedBoxMDF(BaseMetalModel):
    _param_names = ['logfeh_scale']
    _fancy_names = ['[Fe/H]']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3.0, 0.5]]

    feh_mean = None

    def __init__(self, initial_params=None):
        if initial_params is None:
            initial_params = np.array([0.])
        self.set_params(initial_params)

    @property
    def _params(self):
        return np.array([self.feh_mean], dtype=float)

    def set_params(self, feh_params):
        if isinstance(feh_params, float) or isinstance(feh_params, int):
            self.feh_mean = float(feh_params)
        else:
            assert len(feh_params) == self._num_params, (
                'feh_params for ClosedBoxMDF is length {:d}, '
                'should be length {:d}'.format(len(feh_params), self._num_params))
        self.feh_mean = feh_params[0]

    def _compute_mdf(self, etol=1e-2):
        fehs = self.default_fehs
        z = 10.**fehs
        weights = z*np.exp(-z / 10.**self.feh_mean)
        if np.isclose(np.sum(weights), 0.):
            fehs = np.array([self.feh_mean])
            weights = np.array([1.])
        else:
            weights /= np.sum(weights)
            # remove bins with negligible weight
            too_small = (weights < etol)
            fehs = fehs[~too_small]
            weights = weights[~too_small]
            weights /= np.sum(weights)
        assert len(fehs) == len(weights)
        if len(fehs) <= 1:
            return np.array([self.feh_mean]), np.array([1.])
        else:
            return fehs, weights
        
    @property
    def fehs(self):
        return self._compute_mdf()[0]

    @property
    def weights(self):
        return self._compute_mdf()[1]

    def copy(self):
        return ClosedBoxMDF(initial_params=[self.feh_mean])
    

all_metal_models = [SingleFeH, NormMDF, FixedWidthNormMDF, ClosedBoxMDF]
