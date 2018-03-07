# galaxy.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define the Galaxy_Model class"""

__all__ = ['CustomGalaxy', 'DefaultTau', 'DefaultSSP', 'DefaultNonParam',
           'MDFTau', 'LogNormTau']

import numpy as np
from pcmdpy import utils, priors, agemodels, dustmodels, metalmodels


class BaseGalaxy:

    _param_names = ['ages', 'fehs', 'SFH', 'dust_model']

    def __init__(self, ages, fehs, SFH, dust_model, params=None):
        utils.my_assert(len(ages) == len(fehs),
                        "length of first param and second param must match")
        utils.my_assert(len(ages) == len(SFH),
                        "length of first param and third param must match")
        self.ages = ages
        self.fehs = fehs
        self.SFH = SFH
        self.dust_model = dust_model
        self.Npix = np.sum(self.SFH)
        self.num_SSPs = len(self.ages)
        self._params = params

    def iter_SSPs(self):
        for i in range(self.num_SSPs):
            yield self.ages[i], self.fehs[i], self.SFH[i]


class CustomGalaxy(BaseGalaxy):

    def __init__(self, feh_model, dust_model, age_model):
        # set the metallicity model
        self.feh_model = feh_model
        self.p_feh = feh_model._num_params
        self._param_names = [feh_model._param_names]
        # set the dust model
        self.dust_model = dust_model
        self.p_dust = dust_model._num_params
        self._param_names += dust_model._param_names
        # set the age model
        self.age_model = age_model
        self.p_age = age_model._num_params
        self._param_names += age_model._param_names
        self.p_total = self.p_feh + self.p_dust + self.p_age

    def get_flat_prior(self, feh_bounds=None, dust_bounds=None,
                       age_bounds=None):
        if feh_bounds is None:
            bounds = self.feh_model._default_prior_bounds
        else:
            assert(len(feh_bounds) == self.p_feh)
            bounds = feh_bounds
        if dust_bounds is None:
            bounds += self.dust_model._default_prior_bounds
        else:
            assert(len(dust_bounds) == self.p_dust)
            bounds += dust_bounds
        if age_bounds is None:
            bounds = self.age_model._default_prior_bounds
        else:
            assert(len(age_bounds) == self.p_age)
            bounds += age_bounds
        return priors.FlatPrior(bounds)

    def get_model(self, gal_params):
        assert(len(gal_params) == self.p_total)
        feh_params = gal_params[:self.p_feh]
        dust_params = gal_params[self.p_feh:self.p_feh+self.p_dust]
        age_params = gal_params[-self.p_age:]
        fehs, feh_weights = self.feh_model(feh_params).get_vals()
        dust_model = self.dust_model(dust_params)
        ages, age_weights = self.age_model(age_params).get_vals()
        # merge the age and metallicity bins
        new_ages = []
        new_fehs = []
        SFH = []
        for i, feh in enumerate(fehs):
            SFH += list(age_weights * feh_weights[i])
            new_ages += list(ages)
            new_fehs += [feh]*len(ages)

        return BaseGalaxy(new_ages, new_fehs, SFH, dust_model,
                          params=gal_params)


DefaultTau = CustomGalaxy(metalmodels.SingleFeH, dustmodels.SingleDust,
                          agemodels.TauModel)
DefaultSSP = CustomGalaxy(metalmodels.SingleFeH, dustmodels.SingleDust,
                          agemodels.SSPModel)
DefaultNonParam = CustomGalaxy(metalmodels.SingleFeH, dustmodels.SingleDust,
                               agemodels.NonParam)
MDFTau = CustomGalaxy(metalmodels.NormMDF, dustmodels.SingleDust,
                      agemodels.TauModel)
LogNormTau = CustomGalaxy(metalmodels.SingleFeH, dustmodels.LogNormDust,
                          agemodels.TauModel)
