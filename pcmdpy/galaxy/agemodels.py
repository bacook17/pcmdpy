# agemodels.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define AgeModel classes to integrate with Galaxy Models"""

__all__ = ['BaseAgeModel', 'NonParam', 'ConstantSFR', 'TauModel', 'RisingTau',
           'SSPModel', 'get_age_model', 'all_age_models']

import numpy as np


def get_age_model(name, *args, **kwargs):
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
            "given name {} not an acceptable age model. Choose one of:\n"
            "{}".format(name.lower(), ['nonparam', 'constant', 'tau',
                                       'risingtau', 'ssp']))


class BaseAgeModel:
    default_edges = np.array([6., 8., 9., 9.5, 10., 10.14])
    _num_SFH_bins = len(default_edges) - 1

    def __init__(self):
        if not hasattr(self, 'SFH'):
            self.SFH = np.ones((self._num_SFH_bins))
        if not hasattr(self, 'ages'):
            self.ages = np.ones((self._num_SFH_bins))
        if not hasattr(self, '_params'):
            self._params = np.ones((self._num_params))
        self.Npix = np.sum(self.SFH)
        self.logSFH = np.log10(self.SFH)

    def get_vals(self):
        return self.ages, self.SFH

    def get_cum_sfh(self):
        """
        Defined such that first age bin has 100% cum SFH
        """
        normed_sfh = self.SFH / self.Npix
        cum_sfh = 1. - np.cumsum(normed_sfh)
        return np.append(1., cum_sfh)


class NonParam(BaseAgeModel):

    def __init__(self, initial_params=None, iso_step=0.2):
        self.iso_step = iso_step
        if iso_step > 0:
            # construct list of ages, given isochrone spacing
            self.iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            self.iso_edges = self.default_edges
        self.ages = 0.5*(self.iso_edges[:-1] + self.iso_edges[1:])
        # which of the 7 bins does each isochrone belong to?
        self.iso_sfh_bin = np.digitize(self.ages, self.default_edges) - 1
        # include right-most bin if needed
        self.iso_sfh_bin[-1] = np.digitize(self.ages, self.default_edges,
                                           right=True)[-1] - 1
        # compute SFH in each isochrone, given the bin SFH
        self.delta_t_iso = np.diff(10.**self.iso_edges)
        self.delta_t_sfh = np.diff(10.**self.default_edges)
        if initial_params is not None:
            self.set_params(initial_params)
        super().__init__()

    @property
    def _num_params(self):
        return len(self.default_edges)

    @property
    def _param_names(self):
        return ['logSFH{:d}'.format(i) for i in range(self._num_params)]

    @property
    def _fancy_names(self):
        return [r'$\log_{10}$' + 'SFH{:d}'.format(i) for i in range(self.num_params)]

    @property
    def _default_prior_bounds(self):
        return [[-3.0, 3.0]] * self._num_params

    def set_params(self, age_params):
        assert (len(age_params) == self._num_params), "age_params for NonParam should be length {:d}".format(self._num_params)
        self.SFH = 10.**age_params[self.iso_sfh_bin] * self.delta_t_iso
        self.SFH /= self.delta_t_sfh[self.iso_sfh_bin]
        self._params = age_params
        super().__init__()
        return self

    def update_edges(self, new_edges):
        self.default_edges = new_edges
        self._num_SFH_bins = len(self.default_edges) - 1
        self._num_params = len(self._param_names)
        self._default_prior_bounds = [[-3.0, 3.0]] * self._num_params
        self.__init__(iso_step=self.iso_step)
        return self
        
    def as_default(self):
        return type(self)(self._params, iso_step=-1).update_edges(self.default_edges)


class ConstantSFR(BaseAgeModel):

    _param_names = ['logNpix']
    _fancy_names = [r'$\log_{10} N_\mathrm{pix}$']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0]]

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

    def set_params(self, age_params):
        assert len(age_params) == self._num_params, ("age_params for "
                                                     "ConstantSFR should be "
                                                     "length %d" %
                                                     self._num_params)
        Npix = 10.**age_params[0]
        SFH_term = 10.**self.iso_edges[1:] - 10.**self.iso_edges[:-1]
        self.SFH = Npix * SFH_term / np.sum(SFH_term)
        self._params = age_params
        super().__init__()
        return self

    def update_edges(self, new_edges):
        self.default_edges = new_edges
        self._num_SFH_bins = len(self.default_edges) - 1
        self.__init__(self._params, iso_step=self.iso_step)
        return self

    def as_default(self):
        return type(self)(self._params, iso_step=-1)


class TauModel(BaseAgeModel):

    _param_names = ['logNpix', 'tau']
    _fancy_names = [r'$\log_{10} N_\mathrm{pix}$', r'$\tau']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [0.1, 20.]]
    
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

    def set_params(self, age_params):
        assert len(age_params) == self._num_params, ("age_params for Tau_Model"
                                                     " should be length %d" %
                                                     self._num_params)

        Npix = 10.**age_params[0]
        tau = age_params[1]

        ages_linear = 10.**(self.iso_edges - 9.)  # convert to Gyrs
        SFH_term = np.exp(ages_linear[1:]/tau) - np.exp(ages_linear[:-1]/tau)
        self.SFH = Npix * SFH_term / np.sum(SFH_term)
        self._params = age_params
        super().__init__()
        return self

    def update_edges(self, new_edges):
        self.default_edges = new_edges
        self._num_SFH_bins = len(self.default_edges) - 1
        self.__init__(self._params, iso_step=self.iso_step)
        return self

    def as_default(self):
        return type(self)(self._params, iso_step=-1)


class RisingTau(BaseAgeModel):

    _param_names = ['logNpix', 'tau_rise']
    _fancy_names = [r'$\log_{10} N_\mathrm{pix}$', r'$\tau']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [0.1, 20.]]

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

    def set_params(self, age_params):
        assert len(age_params) == self._num_params, ("gal_params for Rising_Tau"
                                                     " should be length %d" %
                                                     self._num_params)
        Npix = 10.**age_params[0]
        tau = age_params[1]

        ages_linear = 10.**(self.iso_edges - 9.)  # convert to Gyrs
        base_term = (ages_linear[-1]+tau-ages_linear) * np.exp(ages_linear/tau)
        SFH_term = base_term[:-1] - base_term[1:]
        self.SFH = Npix * SFH_term / np.sum(SFH_term)
        self._params = age_params
        super().__init__()
        return self

    def update_edges(self, new_edges):
        self.default_edges = new_edges
        self._num_SFH_bins = len(self.default_edges) - 1
        self.__init__(self._params, iso_step=self.iso_step)
        return self

    def as_default(self):
        return type(self)(self._params, iso_step=-1)


class SSPModel(BaseAgeModel):
    _param_names = ['logNpix', 'logage']
    _fancy_names = [r'$\log_{10} N_\mathrm{pix}$', r'$\log_{10}$ age (yr)']
    _num_params = len(_param_names)
    _default_prior_bounds = [[0., 8.0], [8.0, 10.5]]
    
    def __init__(self, initial_params=None, iso_step=None):
        """
        """
        if initial_params is not None:
            self.set_params(initial_params)
        super().__init__()
        
    def set_params(self, age_params):
        assert (len(age_params) == self._num_params), (
            "gal_params for Galaxy_SSP should be length {}, "
            "is length {}".format(self._num_params), len(age_params))
        Npix = 10.**age_params[0]
        self.SFH = np.array([Npix])
        self.ages = np.array([age_params[1]])
        self._params = age_params
        super().__init__()
        return self


all_age_models = ['NonParam', 'ConstantSFR', 'TauModel', 'RisingTau',
                  'SSPModel']