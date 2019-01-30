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
    default_SFH_edges = np.array([6., 8., 9., 9.5, 10., 10.2])
    _num_SFH_bins = len(default_SFH_edges) - 1

    def __init__(self):
        assert hasattr(self, 'iso_edges'), ("iso_edges not set")
        assert hasattr(self, 'SFH'), ('SFH not set')
        if not hasattr(self, '_params'):
            self._params = [None]

    @property
    def ages(self):
        return 0.5*(self.iso_edges[:-1] + self.iso_edges[1:])
            
    @property
    def _num_isochrones(self):
        return len(self.iso_edges) - 1

    @property
    def delta_ts(self):
        return np.diff(10.**(self.iso_edges - 9.))

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

    def as_NonParam(self):
        current_edges = self.iso_edges
        self.update_edges(self.default_SFH_edges)
        other = NonParam(self.logSFH, iso_step=-1,
                         sfh_edges=self.default_SFH_edges)
        self.update_edges(current_edges)
        return other

    def as_default(self):
        return self.as_NonParam()

    def update_edges(self, new_edges):
        self.iso_edges = new_edges


class NonParam(BaseSFHModel):

    _params = np.array([None, None, None, None, None])

    def __init__(self, initial_params=None, iso_step=0.2,
                 sfh_edges=None):
        self.iso_step = iso_step
        if iso_step > 0:
            # construct list of ages, given isochrone spacing
            self.iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            self.iso_edges = self.default_SFH_edges
        self.update_sfh_edges(sfh_edges if sfh_edges is not None else self.default_SFH_edges)
        assert np.all(np.isclose(self.overlap_matrix.sum(axis=1), 1.0)), (
            "The sums over the overlap matrix should all be near 1")
        if initial_params is None:
            initial_params = np.zeros(self._num_params)
        self.set_params(initial_params)
        super().__init__()

    def copy(self):
        return NonParam(initial_params=self._params,
                        iso_step=self.iso_step,
                        sfh_edges=self.sfh_edges)
    
    @property
    def _deltat_sfh(self):
        return np.diff(10.**(self.sfh_edges - 9.))

    @property
    def _num_params(self):
        return len(self.sfh_edges) - 1

    @property
    def _param_names(self):
        return ['logSFH{:d}'.format(i) for i in range(self._num_params)]

    @property
    def _fancy_names(self):
        return [r'$\log\;$' + 'SFH{:d}'.format(i) for i in range(self._num_params)]

    @property
    def _default_prior_bounds(self):
        return [[-3.0, 3.0]] * self._num_params

    def set_params(self, sfh_params):
        is_valid = (hasattr(sfh_params, '__len__') and
                    len(sfh_params) == self._num_params)
        assert is_valid, ('sfh_params must be an array or list of length '
                          '{:d}, not {:d}'.format(self._num_params,
                                                  len(sfh_params)))
        self.SFH = np.dot(10.**sfh_params, self.overlap_matrix)
        assert np.isclose(self.Npix, np.sum(10.**sfh_params))
        self._params = sfh_params

    def from_sfr(self, sfr_params):
        sfh_params = np.log10(self._deltat_sfh) + sfr_params
        self.set_params(sfh_params)

    def update_sfh_edges(self, new_edges):
        self.sfh_edges = new_edges
        self.overlap_matrix = _build_overlap_matrix(10.**self.sfh_edges,
                                                    10.**self.iso_edges)
        
    def update_edges(self, new_edges):
        self.iso_edges = new_edges
        self.overlap_matrix = _build_overlap_matrix(10.**self.sfh_edges,
                                                    10.**self.iso_edges)
        
    def as_NonParam(self):
        # transform current SFH into original SFH bins
        _new_overlap = _build_overlap_matrix(10.**self.sfh_edges,
                                             10.**self.default_SFH_edges)
        sfh_params = np.log10(np.dot(10.**self._params, _new_overlap))
        return NonParam(initial_params=sfh_params, iso_step=-1,
                        sfh_edges=self.default_SFH_edges)
        

class ConstantSFR(BaseSFHModel):

    _param_names = ['logNpix']
    _fancy_names = [r'$\log\; \mathrm{N_{pix}}$']
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
            self.iso_edges = self.default_SFH_edges
        if initial_params is None:
            initial_params = np.zeros(self._num_params)
        self.set_params(initial_params)
        super().__init__()

    def copy(self):
        return ConstantSFR(initial_params=self._params,
                           iso_step=self.iso_step)
    
    def set_params(self, logNpix):
        if hasattr(logNpix, '__len__'):
            assert len(logNpix) == self._num_params, ("params for "
                                                      "ConstantSFR should be "
                                                      "length {:d}, not {:d}".format(self._num_params, len(sfh_params)))
            logNpix = logNpix[0]
        self._params = np.array([logNpix])

    @property
    def SFH(self):
        Npix = 10.**self._params[0]
        return Npix * self.delta_ts / np.sum(self.delta_ts)


class TauModel(BaseSFHModel):

    _param_names = ['logNpix', 'tau']
    _fancy_names = [r'$\log\; \mathrm{N_{pix}}$', r'$\tau$']
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
            self.iso_edges = self.default_SFH_edges
        if initial_params is None:
            initial_params = np.array([0., 1.])
        self.set_params(initial_params)
        super().__init__()

    def copy(self):
        return TauModel(initial_params=self._params,
                        iso_step=self.iso_step)
    
    def set_params(self, sfh_params):
        is_valid = (hasattr(sfh_params, '__len__') and
                    len(sfh_params) == self._num_params)
        assert is_valid, ('sfh_params must be an array or list of length '
                          '{:d}, not {:d}'.format(self._num_params,
                                                  len(sfh_params)))
        self._params = sfh_params

    @property
    def SFH(self):
        Npix = 10.**self._params[0]
        tau = self._params[1]
        ages_linear = 10.**(self.iso_edges - 9.)  # convert to Gyrs
        SFH_term = np.diff(np.exp(ages_linear/tau))
        return Npix * SFH_term / np.sum(SFH_term)


class RisingTau(BaseSFHModel):

    _param_names = ['logNpix', 'tau_rise']
    _fancy_names = [r'$\log\;\mathrm{N_{pix}}$', r'$\tau$']
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
            self.iso_edges = self.default_SFH_edges
        if initial_params is None:
            initial_params = np.array([0., 1.])
        self.set_params(initial_params)
        super().__init__()

    def copy(self):
        return RisingTau(initial_params=self._params,
                         iso_step=self.iso_step)
    
    def set_params(self, sfh_params):
        is_valid = (hasattr(sfh_params, '__len__') and
                    len(sfh_params) == self._num_params)
        assert is_valid, ('sfh_params must be an array or list of length '
                          '{:d}, not {:d}'.format(self._num_params,
                                                  len(sfh_params)))
        self._params = sfh_params
        
    @property
    def SFH(self):
        Npix = 10.**self._params[0]
        tau = self._params[1]
        ages_linear = 10.**(self.iso_edges - 9.)  # convert to Gyrs
        base_term = (ages_linear[-1]+tau-ages_linear) * np.exp(ages_linear/tau)
        SFH_term = np.diff(base_term)
        return Npix * SFH_term / np.sum(SFH_term)


class SSPModel(BaseSFHModel):
    _param_names = ['logNpix', 'logage']
    _fancy_names = [r'$\log\;\mathrm{N_{pix}}$', r'$\log$ age (yr)']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [8.0, 10.5]]
    _params = [None, None]
    
    def __init__(self, initial_params=None, iso_step=None):
        """
        """
        if initial_params is None:
            initial_params = np.array([0.0, 10.0])
        self.set_params(initial_params)
        super().__init__()
        
    def copy(self):
        return SSPModel(initial_params=self._params,
                        iso_step=self.iso_step)
    
    def set_params(self, sfh_params):
        is_valid = (hasattr(sfh_params, '__len__') and
                    len(sfh_params) == self._num_params)
        assert is_valid, ('sfh_params must be an array or list of length '
                          '{:d}, not {:d}'.format(self._num_params,
                                                  len(sfh_params)))
        Npix = 10.**sfh_params[0]
        self.SFH = np.array([Npix])
        self.iso_edges = np.array([-0.1, 0.1]) + sfh_params[1]
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
