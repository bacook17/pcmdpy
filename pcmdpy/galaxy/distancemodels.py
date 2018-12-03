# distancemodels.py
# Ben Cook (bcook@cfa.harvard.edu)
__all__ = ['BaseDistanceModel', 'VariableDistance', 'FixedDistance',
           'get_distance_model', 'dmod_to_mpc', 'mpc_to_dmod',
           'all_distance_models']

import numpy as np


def get_distance_model(name, *args, **kwargs):
    if name.lower() == 'fixed':
        return FixedDistance(*args, **kwargs)
    elif name.lower() == 'variable':
        return VariableDistance(*args, **kwargs)
    else:
        raise NotImplementedError(
            "given name {} not an acceptable distance model. Choose one of:\n"
            "{}".format(name.lower(), ['fixed', 'variable']))


class BaseDistanceModel:

    dmod = 25.

    def __init__(self):
        pass

    def set_params(self, *args):
        pass
    
    @property
    def d_mpc(self):
        return dmod_to_mpc(self.dmod)

    
class VariableDistance(BaseDistanceModel):

    _param_names = ['dmod']
    _fancy_names = [r'$\mu_d$']
    _num_params = len(_param_names)
    _default_prior_bounds = [[25., 30.]]  # 1 - 10 Mpc
    
    def __init__(self, initial_params=None):
        if initial_params is not None:
            self.set_params(initial_params)

    def set_params(self, dist_params):
        if isinstance(dist_params, float) or isinstance(dist_params, int):
            dist_params = [dist_params]
        assert (len(dist_params) == self._num_params), (
            "dist_params for VariableDistance should be length {}, "
            "is length {}".format(self._num_params, len(dist_params)))
        self._params = dist_params
        self.dmod = dist_params[0]


class FixedDistance(BaseDistanceModel):
    """
    To Initialize a FixedDistance model:
    mymodel = FixedDistance(dmod=30.)
    """
    
    _param_names = []
    _fancy_names = []
    _num_params = len(_param_names)
    _default_prior_bounds = []

    def __init__(self, dmod=30.):
        self.dmod = dmod

    @property
    def _params(self):
        return []
    
    
def dmod_to_mpc(dmod):
    return 10.**(0.2 * (dmod - 25.))


def mpc_to_dmod(d_mpc):
    return 25. + 5*np.log10(d_mpc)


all_distance_models = [VariableDistance, FixedDistance]
