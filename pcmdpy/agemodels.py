# agemodels.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define AgeModel classes to integrate with Galaxy Models"""

__all__ = ['NonParam', 'ConstantSFR', 'TauModel', 'RisingTau',
           'SSPModel']

import numpy as np
from pcmdpy import utils


class _AgeModel:
    default_edges = np.array([6., 8., 8.5, 9., 9.5, 10., 10.2])
    _num_SFH_bins = len(default_edges) - 1

    def __init__(self):
        if not hasattr(self, 'SFH'):
            self.SFH = np.ones((self._num_SFH_bins))
        if not hasattr(self, 'ages'):
            self.ages = np.ones((self._num_SFH_bins))
        if not hasattr(self, '_params'):
            self._params = np.ones((self._num_params))
        self.Npix = np.sum(self.SFH)

    def get_vals(self):
        return self.ages, self.SFH

    def get_cum_sfh(self, inverted=True):
        normed_sfh = self.SFH / self.Npix
        if inverted:
            return np.append(0., np.cumsum(normed_sfh[::-1]))[::-1]
        else:
            return np.cumsum(normed_sfh)


class NonParam(_AgeModel):

    _param_names = ['logSFH0', 'logSFH1', 'logSFH2', 'logSFH3', 'logSFH4',
                    'logSFH5']
    _num_params = len(_param_names)
    _default_prior_bounds = [[-3.0, 3.0]] * _num_params

    def __init__(self, iso_step=0.2):
        if iso_step > 0:
            # construct list of ages, given isochrone spacing
            iso_edges = np.arange(6.0, 10.3, iso_step)
            in_range = np.logical_and((iso_edges+0.001 >=
                                       self.default_edges[0]),
                                      (iso_edges-0.001 <=
                                       self.default_edges[-1]))
            iso_edges = iso_edges[in_range]
        else:
            iso_edges = self.default_edges
        self.ages = 0.5*(iso_edges[:-1] + iso_edges[1:])
        # which of the 7 bins does each isochrone belong to?
        self.iso_sfh_bin = np.digitize(self.ages, self.default_edges) - 1
        # include right-most bin if needed
        self.iso_sfh_bin[-1] = np.digitize(self.ages, self.default_edges,
                                           right=True)[-1] - 1
        # compute SFH in each isochrone, given the bin SFH
        self.delta_t_iso = np.diff(10.**iso_edges)
        self.delta_t_sfh = np.diff(10.**self.default_edges)
        super().__init__()

    def update_edges(self, new_edges):
        self.default_edges = new_edges
        self._num_SFH_bins = len(self.default_edges) - 1
        self._param_names = ['logSFH{:d}'.format(i)
                             for i in range(self._num_SFH_bins)]
        self._num_params = len(self._param_names)
        self._default_prior_bounds = [[-3.0, 3.0]] * self._num_params
        self.__init__()
        
    def set_params(self, age_params):
        utils.my_assert(len(age_params) == self._num_params,
                        "age_params for Galaxy_Model should be length %d" %
                        self._num_params)
        self.SFH = 10.**age_params[self.iso_sfh_bin] * self.delta_t_iso
        self.SFH /= self.delta_t_sfh[self.iso_sfh_bin]
        self._params = age_params
        super().__init__()

    def as_default(self):
        temp_model = NonParam(iso_step=-1)
        temp_model.set_params(self._params)
        return temp_model


class ConstantSFR(_AgeModel):

    _param_names = ['logNpix']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0]]

    def __init__(self, iso_step=0.2):
        """
        """
        if iso_step > 0:
            self.iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            self.iso_edges = self.default_edges
        self.ages = 0.5*(self.iso_edges[1:] + self.iso_edges[:-1])
        super().__init__()

    def set_params(self, age_params):
        utils.my_assert(len(age_params) == self._num_params,
                        "age_params for Constant_SFR should be length %d" %
                        self._num_params)

        Npix = 10.**age_params[0]
        SFH_term = 10.**self.iso_edges[1:] - 10.**self.iso_edges[:-1]
        self.SFH = Npix * SFH_term / np.sum(SFH_term)
        self._params = age_params
        super().__init__()

    def as_default(self):
        temp_model = ConstantSFR(iso_step=-1)
        temp_model.set_params(self._params)
        return temp_model


class TauModel(_AgeModel):

    _param_names = ['logNpix', 'tau']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [0.1, 20.]]
    
    def __init__(self, iso_step=0.2):
        """
        """
        if iso_step > 0:
            self.iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            self.iso_edges = self.default_edges
        self.ages = 0.5*(self.iso_edges[1:] + self.iso_edges[:-1])
        super().__init__()

    def set_params(self, age_params):
        utils.my_assert(len(age_params) == self._num_params,
                        "age_params for Tau_Model should be length %d" %
                        self._num_params)

        Npix = 10.**age_params[0]
        tau = age_params[1]

        ages_linear = 10.**(self.iso_edges - 9.)  # convert to Gyrs
        SFH_term = np.exp(ages_linear[1:]/tau) - np.exp(ages_linear[:-1]/tau)
        self.SFH = Npix * SFH_term / np.sum(SFH_term)
        self._params = age_params
        super().__init__()

    def as_default(self):
        temp_model = TauModel(iso_step=-1)
        temp_model.set_params(self._params)
        return temp_model


class RisingTau(_AgeModel):

    _param_names = ['logNpix', 'tau_rise']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [0.1, 20.]]

    def __init__(self, iso_step=0.2):
        """
        """
        if iso_step > 0:
            self.iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            self.iso_edges = self.default_edges
        self.ages = 0.5*(self.iso_edges[1:] + self.iso_edges[:-1])
        super().__init__()

    def set_params(self, age_params):
        utils.my_assert(len(age_params) == self._num_params,
                        "gal_params for Rising_Tau should be length %d" %
                        self._num_params)
        Npix = 10.**age_params[0]
        tau = age_params[1]

        ages_linear = 10.**(self.iso_edges - 9.)  # convert to Gyrs
        base_term = (ages_linear[-1]+tau-ages_linear) * np.exp(ages_linear/tau)
        SFH_term = base_term[:-1] - base_term[1:]
        self.SFH = Npix * SFH_term / np.sum(SFH_term)
        self._params = age_params
        super().__init__()

    def as_default(self):
        temp_model = RisingTau(iso_step=-1)
        temp_model.set_params(self._params)
        return temp_model


class SSPModel(_AgeModel):
    _param_names = ['logNpix', 'logage']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [8.0, 10.5]]
    
    def __init__(self, iso_step=None):
        """
        """
        super().__init__()
        
    def set_params(self, age_params):
        utils.my_assert(len(age_params) == self._num_params,
                        "gal_params for Galaxy_SSP should be length %d" %
                        self._num_params)
        Npix = 10.**age_params[0]
        self.SFH = np.array([Npix])
        self.ages = np.array([age_params[1]])
        self._params = age_params
        super().__init__()
