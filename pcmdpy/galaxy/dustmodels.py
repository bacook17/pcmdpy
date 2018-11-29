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

    def __init__(self):
        pass

    def get_params(self):
        return self.dust_frac, self.mu_dust, self.sig_dust

    def get_stats(self):
        mean = np.exp(self.mu_dust + 0.5*self.sig_dust**2)
        var = mean**2. * (np.exp(self.sig_dust**2) - 1.)
        return mean, np.sqrt(var)


class SingleDust(BaseDustModel):

    _param_names = ['logdust']
    _fancy_names = [r'$\log$ E(B-V)']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3., 0.5]]

    def __init__(self, dust_frac=1.0):
        self.dust_frac = dust_frac

    def set_params(self, dust_params):
        self.mu_dust = dust_params[0] * np.log(10.)  # ln of median
        self.sig_dust = 0.


class LogNormDust(BaseDustModel):
    
    _param_names = ['logdust_med', 'dust_sig']
    _fancy_names = [r'$\log$ E(B-V)', r'$\sigma$ E(B-V)']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3., 0.5], [0., 1.]]

    def __init__(self, dust_frac=0.5):
        self.dust_frac = dust_frac

    def set_params(self, dust_params):
        self.mu_dust = dust_params[0] * np.log(10.)  # ln of median
        self.sig_dust = dust_params[1]  # dimensionless standard-deviation


class FixedWidthLogNormDust(LogNormDust):

    _param_names = ['logdust_med']
    _fancy_names = [r'$\log$ E(B-V)']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3., 0.5]]

    def __init__(self, sig=0.2, dust_frac=0.5):
        self.sig_dust = sig
        super().__init__(dust_frac=dust_frac)

    def set_params(self, dust_params):
        self.mu_dust = dust_params[0] * np.log(10.)  # ln of median


all_dust_models = [SingleDust, LogNormDust, FixedWidthLogNormDust]
