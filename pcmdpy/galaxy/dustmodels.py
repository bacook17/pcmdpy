# dustmodels.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define the DustModel classses to integrate with GalaxyModel"""

__all__ = ['BaseDustModel', 'SingleDust', 'LogNormDust',
           'FixedWidthLogNormDust', 'get_dust_model', 'all_dust_models']

import numpy as np


def get_dust_model(name, *args, **kwargs):
    if name.lower() == 'single':
        return SingleDust(*args, **kwargs)
    elif name.lower() == 'lognorm':
        return LogNormDust(*args, **kwargs)
    elif name.lower() == 'fixedwidth':
        return FixedWidthLogNormDust(*args, **kwargs)
    else:
        raise NotImplementedError(
            "given name {} not an acceptable dust model. Choose one of:\n"
            "{}".format(name.lower(), ['single', 'fixedwidth', 'lognorm']))

    
class BaseDustModel:
    dust_frac = None
    mu_dust = None
    sig_dust = None

    def __init__(self):
        pass

    def get_props(self):
        return self.dust_frac, self.mu_dust * np.log(10.), self.sig_dust  # 2nd term: ln of median

    def get_stats(self):
        mean = np.exp(self.mu_dust + 0.5*self.sig_dust**2)
        var = mean**2. * (np.exp(self.sig_dust**2) - 1.)
        return mean, np.sqrt(var)


class SingleDust(BaseDustModel):

    _param_names = ['logdust']
    _fancy_names = [r'$\log$ E(B-V)']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3., 0.5]]

    def __init__(self, initial_params=None, dust_frac=1.0):
        self.dust_frac = dust_frac
        self.sig_dust = 0.0
        if initial_params is not None:
            self.set_params(initial_params)
    
    @property
    def _params(self):
        return np.array([self.mu_dust])

    def set_params(self, dust_params):
        if isinstance(dust_params, float) or isinstance(dust_params, int):
            dust_params = [dust_params]
        assert len(dust_params) == self._num_params, (
            "dust_params for SingleDust is length {:d}, "
            "should be length {:d}".format(len(dust_params), self._num_params))
        self.mu_dust = dust_params[0]

    def copy(self):
        return SingleDust(initial_params=[self.mu_dust],
                          dust_frac=self.dust_frac)


class LogNormDust(BaseDustModel):
    
    _param_names = ['logdust_med', 'dust_sig']
    _fancy_names = [r'$\log$ E(B-V)', r'$\sigma$ E(B-V)']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3., 0.5], [0., 1.]]

    def __init__(self, initial_params=None, dust_frac=0.5):
        self.dust_frac = dust_frac
        if initial_params is not None:
            self.set_params(initial_params)

    @property
    def _params(self):
        return np.array([self.mu_dust, self.sig_dust])

    def set_params(self, dust_params):
        assert len(dust_params) == self._num_params, (
            "dust_params for LogNormDust is length {:d}, "
            "should be length {:d}".format(len(dust_params), self._num_params))
        self.mu_dust = dust_params[0]
        self.sig_dust = dust_params[1]  # dimensionless standard-deviation

    def copy(self):
        return LogNormDust(initial_params=[self.mu_dust, self.sig_dust],
                           dust_frac=self.dust_frac)


class FixedWidthLogNormDust(LogNormDust):

    _param_names = ['logdust_med']
    _fancy_names = [r'$\log$ E(B-V)']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3., 0.5]]

    def __init__(self, sig=0.2, initial_params=None, dust_frac=0.5):
        self.sig_dust = sig
        self.dust_frac = dust_frac
        if initial_params is not None:
            self.set_params(initial_params)

    @property
    def _params(self):
        return np.array([self.mu_dust])

    def set_params(self, dust_params):
        if isinstance(dust_params, float) or isinstance(dust_params, int):
            dust_params = [dust_params]
        assert len(dust_params) == self._num_params, (
            "dust_params for FixedWidthLogNorm is length {:d}, "
            "should be length {:d}".format(len(dust_params), self._num_params))
        self.mu_dust = dust_params[0]

    def copy(self):
        return FixedWidthLogNormDust(initial_params=[self.mu_dust],
                                     dust_frac=self.dust_frac,
                                     sig=self.sig_dust)


all_dust_models = [SingleDust, LogNormDust, FixedWidthLogNormDust]
