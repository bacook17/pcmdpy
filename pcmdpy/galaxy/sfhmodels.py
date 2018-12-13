# sfhmodels.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define SFHModel classes to integrate with Galaxy Models"""

__all__ = ['BaseSFHModel', 'NonParam', 'ConstantSFR', 'TauModel', 'RisingTau',
           'SSPModel', 'get_sfh_model', 'all_sfh_models']

import numpy as np


def get_sfh_model(name, *args, **kwargs):
    if name.lower() == 'nonparam':
        return NonParam(*args, **kwargs)
    elif name.lower() == 'constant':
        return ConstantSFR(*args, **kwargs)
    elif name.lower() == 'tau':
        return TauModel(*args, **kwargs)
    elif name.lower() == 'risingtau':
        return RisingTau(*args, **kwargs)
    elif name.lower() == 'ssp':
        return SSPModel(*args, **kwargs)
    else:
        raise NotImplementedError(
            "given name {} not an acceptable SFH model. Choose one of:\n"
            "{}".format(name.lower(), ['nonparam', 'constant', 'tau',
                                       'risingtau', 'ssp']))


class BaseSFHModel:
    default_edges = np.array([6., 8., 9., 9.5, 10., 10.2])
    _num_SFH_bins = len(default_edges) - 1

    def __init__(self):
        if not hasattr(self, 'SFH'):
            self.SFH = np.ones((self._num_SFH_bins))
        if not hasattr(self, 'ages'):
            self.ages = np.ones((self._num_SFH_bins))
        if not hasattr(self, '_params'):
            self._params = np.ones((self._num_params))

    @property
    def Npix(self):
        return np.sum(self.SFH)

    @property
    def logNpix(self):
        return np.log10(self.Npix)

    @property
    def logSFH(self):
        return np.log10(self.SFH)

    def get_vals(self):
        return self.ages, self.SFH

    def get_cum_sfh(self):
        """
        Defined such that first age bin has 100% cum SFH
        """
        normed_sfh = self.SFH / self.Npix
        cum_sfh = 1. - np.cumsum(normed_sfh)
        return np.append(1., cum_sfh)


class NonParam(BaseSFHModel):

    _params = np.array([None, None, None, None, None])

    def __init__(self, initial_params=None, iso_step=0.2,
                 sfh_edges=None):
        self.iso_step = iso_step
        if iso_step > 0:
            # construct list of ages, given isochrone spacing
            self.iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            self.iso_edges = self.default_edges
        self.sfh_edges = sfh_edges or self.default_edges
        self.ages = 0.5*(self.iso_edges[:-1] + self.iso_edges[1:])
        self.overlap_matrix = _build_overlap_matrix(10.**self.sfh_edges,
                                                    10.**self.iso_edges)
        assert np.all(np.isclose(self.overlap_matrix.sum(axis=1), 1.0)), (
            "The sums over the overlap matrix should all be near 1")
        if initial_params is not None:
            self.set_params(initial_params)
        super().__init__()

    @property
    def _num_isochrones(self):
        return len(self.ages)

    @property
    def _deltat_iso(self):
        return np.diff(10.**(self.iso_edges-9.))

    @property
    def _deltat_sfh(self):
        return np.diff(10.**(self.sfh_edges-9.))

    @property
    def _num_params(self):
        return len(self.default_edges) - 1

    @property
    def _param_names(self):
        return ['logSFH{:d}'.format(i) for i in range(self._num_params)]

    @property
    def _fancy_names(self):
        return [r'$\log_{10}$' + 'SFH{:d}'.format(i) for i in range(self._num_params)]

    @property
    def _default_prior_bounds(self):
        return [[-3.0, 3.0]] * self._num_params

    def set_params(self, sfh_params):
        assert (len(sfh_params) == self._num_params), "sfh_params for NonParam should be length {:d}".format(self._num_params)
        self.SFH = np.dot(10.**sfh_params, self.overlap_matrix)
        assert np.isclose(self.Npix, np.sum(10.**sfh_params))
        self._params = sfh_params

    def from_sfr(self, sfr_params):
        sfh_params = self._deltat_sfh * 10.**sfr_params
        self.set_params(np.log10(sfh_params))
        
    def update_edges(self, new_edges):
        self.sfh_edges = new_edges
        self._num_SFH_bins = len(self.default_edges) - 1
        self.__init__(iso_step=self.iso_step)
        return self
        
    def as_default(self):
        return type(self)(self._params, iso_step=-1).update_edges(self.default_edges)


class ConstantSFR(BaseSFHModel):

    _param_names = ['logNpix']
    _fancy_names = [r'$\log_{10} N_\mathrm{pix}$']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0]]
    _params = [None]

    def __init__(self, initial_params=None, iso_step=0.2):
        """
        """
        self.iso_step = iso_step
        if iso_step > 0:
            self.iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            self.iso_edges = self.default_edges
        self.ages = 0.5*(self.iso_edges[1:] + self.iso_edges[:-1])
        if initial_params is not None:
            self.set_params(initial_params)
        super().__init__()

    def set_params(self, sfh_params):
        assert len(sfh_params) == self._num_params, ("sfh_params for "
                                                     "ConstantSFR should be "
                                                     "length %d" %
                                                     self._num_params)
        Npix = 10.**sfh_params[0]
        SFH_term = 10.**self.iso_edges[1:] - 10.**self.iso_edges[:-1]
        self.SFH = Npix * SFH_term / np.sum(SFH_term)
        self._params = sfh_params

    def update_edges(self, new_edges):
        self.default_edges = new_edges
        self._num_SFH_bins = len(self.default_edges) - 1
        self.__init__(self._params, iso_step=self.iso_step)
        return self

    def as_default(self):
        return type(self)(self._params, iso_step=-1)


class TauModel(BaseSFHModel):

    _param_names = ['logNpix', 'tau']
    _fancy_names = [r'$\log_{10} N_\mathrm{pix}$', r'$\tau$']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [0.1, 20.]]
    _params = [None, None]
    
    def __init__(self, initial_params=None, iso_step=0.2):
        """
        """
        self.iso_step = iso_step
        if iso_step > 0:
            self.iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            self.iso_edges = self.default_edges
        self.ages = 0.5*(self.iso_edges[1:] + self.iso_edges[:-1])
        if initial_params is not None:
            self.set_params(initial_params)
        super().__init__()

    def set_params(self, sfh_params):
        assert len(sfh_params) == self._num_params, ("sfh_params for Tau_Model"
                                                     " should be length %d" %
                                                     self._num_params)

        Npix = 10.**sfh_params[0]
        tau = sfh_params[1]

        ages_linear = 10.**(self.iso_edges - 9.)  # convert to Gyrs
        SFH_term = np.exp(ages_linear[1:]/tau) - np.exp(ages_linear[:-1]/tau)
        self.SFH = Npix * SFH_term / np.sum(SFH_term)
        self._params = sfh_params

    def update_edges(self, new_edges):
        self.default_edges = new_edges
        self._num_SFH_bins = len(self.default_edges) - 1
        self.__init__(self._params, iso_step=self.iso_step)
        return self

    def as_default(self):
        return type(self)(self._params, iso_step=-1)


class RisingTau(BaseSFHModel):

    _param_names = ['logNpix', 'tau_rise']
    _fancy_names = [r'$\log_{10} N_\mathrm{pix}$', r'$\tau$']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [0.1, 20.]]
    _params = [None, None]

    def __init__(self, initial_params=None, iso_step=0.2):
        """
        """
        self.iso_step = iso_step
        if iso_step > 0:
            self.iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            self.iso_edges = self.default_edges
        self.ages = 0.5*(self.iso_edges[1:] + self.iso_edges[:-1])
        if initial_params is not None:
            self.set_params(initial_params)
        super().__init__()

    def set_params(self, sfh_params):
        assert len(sfh_params) == self._num_params, ("gal_params for Rising_Tau"
                                                     " should be length %d" %
                                                     self._num_params)
        Npix = 10.**sfh_params[0]
        tau = sfh_params[1]

        ages_linear = 10.**(self.iso_edges - 9.)  # convert to Gyrs
        base_term = (ages_linear[-1]+tau-ages_linear) * np.exp(ages_linear/tau)
        SFH_term = base_term[:-1] - base_term[1:]
        self.SFH = Npix * SFH_term / np.sum(SFH_term)
        self._params = sfh_params

    def update_edges(self, new_edges):
        self.default_edges = new_edges
        self._num_SFH_bins = len(self.default_edges) - 1
        self.__init__(self._params, iso_step=self.iso_step)
        return self

    def as_default(self):
        return type(self)(self._params, iso_step=-1)


class SSPModel(BaseSFHModel):
    _param_names = ['logNpix', 'logage']
    _fancy_names = [r'$\log_{10} N_\mathrm{pix}$', r'$\log_{10}$ age (yr)']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [8.0, 10.5]]
    _params = [None, None]
    
    def __init__(self, initial_params=None, iso_step=None):
        """
        """
        if initial_params is not None:
            self.set_params(initial_params)
        super().__init__()
        
    def set_params(self, sfh_params):
        assert (len(sfh_params) == self._num_params), (
            "gal_params for Galaxy_SSP should be length {}, "
            "is length {}".format(self._num_params), len(sfh_params))
        Npix = 10.**sfh_params[0]
        self.SFH = np.array([Npix])
        self.ages = np.array([sfh_params[1]])
        self._params = sfh_params


all_sfh_models = [NonParam, TauModel, RisingTau,
                  SSPModel, ConstantSFR]


def _overlap(left1, right1, left2, right2):
    x = (min(right1, right2) - max(left1, left2)) / (right1 - left1)
    return max(0, x)


def _build_overlap_matrix(arr1, arr2):
    result = np.zeros((len(arr1)-1, len(arr2)-1))
    for i in range(len(arr1)-1):
        for j in range(len(arr2)-1):
            result[i, j] = _overlap(arr1[i], arr1[i+1],
                                    arr2[j], arr2[j+1])
    return result
