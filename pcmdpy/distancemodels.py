# distancemodels.py
# Ben Cook (bcook@cfa.harvard.edu)


class VariableDistance:

    _param_names = ['dmod']
    _num_params = len(_param_names)
    _default_prior_bounds = [[25., 30.]]  # 1 - 10 Mpc
    
    def __init__(self):
        pass

    def set_params(self, dist_params):
        self.dmod = dist_params[0]
        self.d_mpc = 10.**(0.2 * (self.dmod - 25.))


class FixedDistance:
    """
    To Initialize a FixedDistance model:
    mymodel = FixedDistance(dmod=30.)
    """
    
    _param_names = []
    _num_params = len(_param_names)
    _default_prior_bounds = []

    def __init__(self, dmod):
        self.dmod = dmod
        self.d_mpc = 10.**(0.2 * (self.dmod - 25.))
    
    def set_params(self, *args):
        pass
