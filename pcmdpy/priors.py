# priors.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from scipy.stats import dirichlet, gamma

class FlatPrior(object):
    """
    Encapsulates an N-dimensional flat prior as an object

    Methods
    ---------
    lnprior(params)
        Compute log of prior for given parameters
    prior_transform(normed_params)
        Convert normalized parameters [0,1] to physical parameters
        using the inverse of the prior

    """
    
    def __init__(self, bounds):
        """
        Construct a FlatPrior object for given bounds

        Parameters
        ----------
        bounds : array_like (Ndim x 2)
             upper and lower bounds for each dimension

        Yields
        ------
        FlatPrior
             Object representing a FlatPrior with given bounds

        Raises
        ------
        Value Error
            If input bounds are not array_like with dimension Nx2
            OR if upper bounds are lesser than lower bounds in any dimension

        """
        if type(bounds) is not np.ndarray:
            bounds = np.array(bounds).astype(float)
        if bounds.ndim != 2:
            raise ValueError('The input bounds must be Ndim x 2')
        if bounds.shape[1] != 2:
            raise ValueError('The input bounds must be Ndim x 2')
        self.ndim = bounds.shape[0]
        self.lower_bounds = bounds[:,0]
        self.upper_bounds = bounds[:,1]
        self.widths = self.upper_bounds - self.lower_bounds
        if np.any(lower_bounds > upper_bounds):
            raise ValueError('The upper bounds must be greater than the lower bounds in all dimensions')

    def lnprior(self, params):
        """
        Return ln of prior for given parameters

        Parameters
        ----------
        params : array_like
             Physical parameters in the space of the prior

        Returns
        -------
        float
             Natural log of the computed prior (0 if inside range, -np.inf if outside range)

        Raises
        ------
        ValueError
             If length of params doesn't match prior object's number of dimensions
        
        """
        if len(params) != self.ndim:
            raise ValueError('The parameter dimension must equal {:d}'.format(self.ndim))
        if np.any(params < self.lower_bounds) or np.any(params > self.upper_bounds):
            return -np.inf
        else:
            return 0.

    def prior_transform(self, normed_params):
        """
        Return physical parameters corresponding to the normalized [0,1] parameters.

        Parameters
        ----------
        normed_params : array_like
             Array of unit-scaled [0,1] parameters

        Returns
        -------
        array of floats
             Physical parameters represented by normalized parameters

        Raises
        ------
        ValueError
             If length of normed_params doesn't match prior object's number of dimensions
             OR if any of normed_params are outside [0,1]
        
        """
        if len(normed_params) != self.ndim:
            raise ValueError('The parameter dimension must equal {:d}'.format(self.ndim))
        if np.any(normed_params < 0.) or np.any(normed_params > 1.):
            raise ValueError('All normalized parameters must be within [0,1]')
        return self.lower_bounds + self.widths*normed_params

class SSPFlatPrior(Flat_Prior):

    def __init__(self, z_bound=[-2., 0.5], dust_bound=[-3., 0.5],
                 npix_bound=[-1., 8.], age_bound=[6., 10.3]):
        bounds = np.array([z_bound, dust_bound, npix_bound, age_bound])
        FlatPrior.__init__(self, bounds)

class ConstFlatPrior(Flat_Prior):

    def __init__(self, z_bound=[-2., 0.5], dust_bound=[-3., 0.5],
                 npix_bound=[-1., 8.]):
        bounds = np.array([z_bound, dust_bound, npix_bound])
        FlatPrior.__init__(self, bounds)

class TauFlatPrior(Flat_Prior):

    def __init__(self, z_bound=[-2., 0.5], dust_bound=[-3., 0.5],
                 npix_bound=[-1., 8.], tau_bound=[.1, 20.]):
        bounds = np.array([z_bound, dust_bound, npix_bound, tau_bound])
        FlatPrior.__init__(self, bounds)
        
class FullFlatPrior(Prior):

    def __init__(self, z_bound=[-2., 0.5], dust_bound=[-3., 0.5],
                 sfh0_bound=[5e-5, 5e-3], sfh1_bound=[5e-4, 5e-2],
                 sfh2_bound=[1e-3, 1e-1], sfh3_bound=[4e-3, 4e-1],
                 sfh4_bound=[1e-2, 1e0], sfh5_bound=[4e-2, 4e0],
                 sfh6_bound=[3e-2, 3e0]):
        bounds = np.array([z_bound, dust_bound, sfh0_bound, sfh1_bound, sfh2_bound,
                           sfh3_bound, sfh4_bound, sfh5_bound, sfh6_bound])
        FlatPrior.__init__(self, bounds)


#class FullDirichletPrior(Prior):
#    def __init__(self, z_bound=[-2., 0.5], dust_bound=[-3., 0.5],
#                 alpha=[5e-4, 5e-3, 1e-2, 4e-2, 1e-1, 4e-1, 3e-3],
#                 npix_bound=[-1., 8.]):
#        self.lower_bounds = np.array([z_bound[0], dust_bound[0], npix_bound[0]])
#        self.upper_bounds = np.array([z_bound[1], dust_bound[1], npix_bound[1]])
#        self.widths = self.upper_bounds - self.lower_bounds
#        self.alpha = np.array(alpha)
#        self.alpha /= np.sum(alpha)
#        self.ndim = 9
#        
#    def ln_prior(self, params):
#        logz, logdust = params[0], params[1]
#        sfh = 10.**params[2:]
#        log_npix = np.log10(sfh.sum())
#        sfh_norm = sfh / sfh.sum()
#        p = np.array([logz, logdust, log_npix])
#        if np.any(p < self.lower_bounds) or np.any(p > self.upper_bounds):
#            return -np.inf
#        else:
#            return dirichlet.logpdf(sfh_norm, self.alpha)
#
        
