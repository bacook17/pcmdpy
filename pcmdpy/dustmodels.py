# dustmodels.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define the DustModel classses to integrate with GalaxyModel"""

__all__ = ['SingleDust', 'LogNormDust']

import numpy as np
from pcmdpy import utils
###### REPLACE ALL ASSERTS WITH MY_ASSERT
from scipy.stats import lognorm


class _DustModel:

    def __init__(self):
        pass

    def get_params(self):
        return self.dust_frac, self.mu_dust, self.sig_dust

    def get_stats(self):
        mean = np.exp(self.mu_dust + 0.5*self.sig_dust**2)
        var = mean**2. * (np.exp(self.sig_dust**2) - 1.)
        return mean, np.sqrt(var)


class SingleDust(_DustModel):

    _param_names = ['logdust_med']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3., 0.5]]

    def __init__(self, dust_params):
        self.mu_dust = dust_params[0] * np.log(10.)  # ln of median
        self.sig_dust = 0.
        self.dust_frac = 1.0


class LogNormDust(_DustModel):
    
    _param_names = ['logdust_med', 'dust_sig']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3., 0.5], [0., 1.]]

    def __init__(self, dust_params):
        self.mu_dust = dust_params[0] * np.log(10.)  # ln of median
        self.sig_dust = dust_params[1]  # dimensionless standard-deviation
        self.dust_frac = 0.5
