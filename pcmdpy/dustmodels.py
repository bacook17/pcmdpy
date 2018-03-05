# dustmodels.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define the DustModel classses to integrate with GalaxyModel"""

__all__ = ['SingleDust', 'LogNormDust']

import numpy as np
from pcmdpy import utils
###### REPLACE ALL ASSERTS WITH MY_ASSERT
from scipy.stats import lognorm


class DustModel:

    def __init__(self):
        pass

    def add_dust(self, mags, filters):
        pass


class SingleDust(DustModel):

    _param_names = ['logdust']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3., 0.5]]

    def __init__(self, dust_params):
        self.dust = 10.**dust_params[0]

    def add_dust(self, mags, filters):
        n_f = len(filters)
        assert(mags.shape[0] == n_f)
        for i, f in enumerate(filters):
            mags[i] += self.logdust * f.red_per_ebv
        return mags


class LogNormDust(DustModel):
    
    _param_names = ['logdust_med', 'log_sig']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3., 0.5], [-2., 0.]]
    
    def __init__(self, dust_params):
        self.Av_med = 10.**dust_params[0]  # median Av
        self.sig = 10.**dust_params[1]  # dimensionless standard-deviation
        if self.sig <= 1e-10:
            self.sig = 1e-10
        self.dist = lognorm(self.sig, scale=self.Av_med)
        self.std_Av = self.dist.std()
        self.mean_Av, self.skew_Av = self.dist.stats(moments='ms')
        
    def add_dust(self, mags, filters):
        n_f = len(filters)
        assert(mags.shape[0] == n_f)
        size = mags.shape[1:]
        if self.sig <= 1e-10:
            Av_draw = np.ones(size) * self.mean_Av
        else:
            Av_draw = self.dist.rvs(size=size)
        for i, f in enumerate(filters):
            mags_draw = Av_draw * f.red_per_ebv
            mags[i] += mags_draw
        return mags

