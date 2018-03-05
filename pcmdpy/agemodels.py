# agemodels.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define AgeModel classes to integrate with Galaxy Models"""

__all__ = ['NonParam', 'ConstSFR', 'TauModel', 'RisingTau',
           'SSPModel']

import numpy as np
from pcmdpy import utils


class AgeModel:
    default_edges = np.array([6., 7., 8., 8.5, 9., 9.5, 10., 10.2])
    _num_SFH_bins = len(default_edges) - 1

    def __init__(self):
        pass

    def get_vals(self):
        return self.ages, self.SFH


class NonParam(AgeModel):

    _param_names = ['logSFH0', 'logSFH1', 'logSFH2', 'logSFH3', 'logSFH4',
                    'logSFH5', 'logSFH6']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3.0, 3.0]] * _num_params

    def __init__(self, age_params, iso_step=0.2):
        """
        age_params:
           0:6 -- log SFH in age bin
        """
        if iso_step > 0:
            # construct list of ages, given isochrone spacing
            iso_edges = np.arange(6.0, 10.3, iso_step)
            # toss out any bins outside the default 7-bin range
            in_range = np.logical_and((iso_edges+0.001 >=
                                       self.default_edges[0]),
                                      (iso_edges-0.001 <=
                                       self.default_edges[-1]))
            iso_edges = iso_edges[in_range]
        else:
            iso_edges = self.default_edges
        self.ages = 0.5*(iso_edges[:-1] + iso_edges[1:])
        # which of the 7 bins does each isochrone belong to?
        iso_sfh_bin = np.digitize(self.ages, self.default_edges) - 1
        # compute SFH in each isochrone, given the bin SFH
        delta_t_iso = np.diff(10.**iso_edges)
        delta_t_sfh = np.diff(10.**self.default_edges)
        self.SFH = 10.**age_params[iso_sfh_bin] * delta_t_iso
        self.SFH /= delta_t_sfh[iso_sfh_bin]
        utils.my_assert(len(age_params) == self._num_params,
                        "age_params for Galaxy_Model should be length %d" %
                        self._num_params)

    def as_default(self):
        return NonParam(self._params, iso_step=-1)


class ConstantSFR(AgeModel):

    _param_names = ['logNpix']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0]]

    def __init__(self, age_params, iso_step=0.2):
        """
        age_params:
           0 -- log Npix
        iso_step: 
        """
        if iso_step > 0:
            iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            iso_edges = self.default_edges
        self.ages = 0.5*(iso_edges[1:] + iso_edges[:-1])
        utils.my_assert(len(age_params) == self._num_params,
                        "age_params for Constant_SFR should be length %d" %
                        self._num_params)

        Npix = 10.**age_params[0]
        SFH_term = 10.**iso_edges[1:] - 10.**iso_edges[:-1]
        self.SFH = Npix * SFH_term / np.sum(SFH_term)

    def as_default(self):
        return ConstantSFR(self._params, iso_step=-1)


class TauModel(AgeModel):

    _param_names = ['logNpix', 'tau']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [0.1, 20.]]
    
    def __init__(self, age_params, iso_step=0.2):
        """
        age_params:
           0 -- log Npix
           1 -- tau (SFH time-scale, in Gyr)
        iso_step: 
        """

        if iso_step > 0:
            iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            iso_edges = self.default_edges
        self.ages = 0.5*(iso_edges[1:] + iso_edges[:-1])
        utils.my_assert(len(age_params) == self._num_params,
                        "age_params for Tau_Model should be length %d" %
                        self._num_params)

        Npix = 10.**age_params[0]
        tau = age_params[1]

        ages_linear = 10.**(iso_edges - 9.)  # convert to Gyrs
        SFH_term = np.exp(ages_linear[1:]/tau) - np.exp(ages_linear[:-1]/tau)
        self.SFH = Npix * SFH_term / np.sum(SFH_term)

    def as_default(self):
        return TauModel(self._params, iso_step=-1)


class RisingTau(AgeModel):

    _param_names = ['logNpix', 'tau']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [0.1, 20.]]

    def __init__(self, age_params, iso_step=0.2):
        """
        age_params:
           0 -- log Npix
           1 -- tau (SFH time-scale, in Gyr)
        iso_step: 
        """

        if iso_step > 0:
            iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            iso_edges = BaseGalaxy.default_edges
        self.ages = 0.5*(iso_edges[1:] + iso_edges[:-1])
        utils.my_assert(len(age_params) == self._num_params,
                        "gal_params for Rising_Tau should be length %d" %
                        self._num_params)
        Npix = 10.**age_params[0]
        tau = age_params[1]

        ages_linear = 10.**(iso_edges - 9.)  # convert to Gyrs
        base_term = (ages_linear[-1]+tau-ages_linear) * np.exp(ages_linear/tau)
        SFH_term = base_term[:-1] - base_term[1:]
        self.SFH = Npix * SFH_term / np.sum(SFH_term)

    def as_default(self):
        return RisingTau(self._params, iso_step=-1)


class SSPModel(AgeModel):
    _param_names = ['logNpix', 'logage']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [8.0, 10.5]]
    
    def __init__(self, age_params):
        """
        age_params:
           0 -- log Npix
           1 -- log age (in yrs)
        """
        utils.my_assert(len(age_params) == self._num_params,
                        "gal_params for Galaxy_SSP should be length %d" %
                        self._num_params)
        Npix = 10.**age_params[0]
        self.SFH = np.array([Npix])
        self.ages = np.array([age_params[1]])
