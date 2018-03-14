# priors.py
# Ben Cook (bcook@cfa.harvard.edu)

"""
Classes and methods to encapsulate various priors and prior transforms

"""

__all__ = ['FlatPrior', 'SSPFlatPrior', 'ConstFlatPrior', 'TauFlatPrior',
           'FullFlatPrior']

import numpy as np
from pcmdpy import galaxy
# from scipy.stats import dirichlet    # , gamma


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
        Yield a `FlatPrior` object for given bounds

        Parameters
        ----------
        bounds : array_like with shape (Ndim, 2)
             upper and lower bounds for each dimension

        Yields
        ------
        `FlatPrior`
             Object representing a `FlatPrior` with given bounds

        Raises
        ------
        `ValueError`
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
        self.lower_bounds = bounds[:, 0]
        self.upper_bounds = bounds[:, 1]
        self.widths = self.upper_bounds - self.lower_bounds
        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError('The upper bounds must be greater than'
                             'the lower bounds in all dimensions')

    def update_bound(self, index, bounds):
        self.lower_bounds[index] = bounds[0]
        self.upper_bounds[index] = bounds[1]
        self.widths = self.upper_bounds - self.lower_bounds
        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError('The upper bounds must be greater than'
                             'the lower bounds in all dimensions')

    def add_bound(self, bounds):
        self.lower_bounds = np.append(self.lower_bounds, 0.)
        self.upper_bounds = np.append(self.upper_bounds, 0.)
        self.update_bound(-1, bounds)
    
    def lnprior(self, params):
        """
        Return ln of prior for given parameters. Typically either 0
        if inside range or -`~numpy.inf` if outside range.

        Parameters
        ----------
        params : array_like
             Physical parameters in the space of the prior

        Returns
        -------
        float
             Natural log of the computed prior (0 if inside range,
             -`numpy.inf` if outside range)

        Raises
        ------
        `ValueError`
             If length of params doesn't match prior object's dimension
        
        """
        if len(params) != self.ndim:
            raise ValueError('len(params) must '
                             'equal {:d}'.format(self.ndim))
        if (np.any(params < self.lower_bounds)
                or np.any(params > self.upper_bounds)):
            return -np.inf
        else:
            return 0.

    def prior_transform(self, normed_params):
        """
        Return physical params corresponding to the normed [0,1] params.

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
        `ValueError`
             If length of normed_params doesn't match prior object's dimension
             OR if any of normed_params are outside [0,1]
        
        """
        if len(normed_params) != self.ndim:
            raise ValueError('len(normed_params) must '
                             'equal {0:d}. Instead is '
                             '{1:d}'.format(self.ndim, len(normed_params)))
        if np.any(normed_params < 0.) or np.any(normed_params > 1.):
            raise ValueError('All normalized parameters must be within [0,1]')
        return self.lower_bounds + self.widths*normed_params

    
class SSPFlatPrior(FlatPrior):
    """
    A `FlatPrior` object representing an SSP (Simple Stellar Population)
    with 4 free-parameters: metallicity (logzh), dust (log E(B-V)), log_Npix,
    and age (in log years).

    Corresponds to the `~pcmdpy.galaxy.Galaxy_SSP` model

    Methods
    ---------
    lnprior(params)
        Compute log of prior for given parameters
    prior_transform(normed_params)
        Convert normalized parameters [0,1] to physical parameters
        using the inverse of the prior
    """
    def __init__(self, z_bound=[-2., 0.5], dust_bound=[-3., 0.5],
                 npix_bound=[-1., 8.], age_bound=[6., 10.3]):
        """
        Yields an `SSPFlatPrior` object with specified bounds in each
        dimension.

        Parameters
        ----------
        z_bound : array-like with shape (2,), optional
             lower (default -2.) and upper (default 0.5) bounds of metallicity,
             in units log_10(z/z_solar).
        dust_bound : array-like with shape (2,), optional
             lower (default -3.) and upper (default 0.5) bounds of dust
             extinction, in units log_10 E(B-V).
        npix_bound : array-like with shape (2,), optional
             lower (default -1.) and upper (default 8.) bounds of
             star-per-pixel, in units log_10 N_pix
        age_bound : array-like with shape (2,), optional
             lower (default 6.) and upper (default 10.3) bounds of age,
             in units log_10 years
        
        Yields
        ------
        `SSPFlatPrior`
             Object representing a flat SSP prior with given bounds

        Raises
        ------
        `ValueError`
            If any key-word arguments are not array-like with length 2
            OR if upper bounds are lesser than lower bounds in any dimension
        """
        bounds = np.array([z_bound, dust_bound, npix_bound, age_bound])
        FlatPrior.__init__(self, bounds)

        
class ConstFlatPrior(FlatPrior):
    """
    A `FlatPrior` object representing a 7-part SFH (Star Formation
    History) that assumes constant star-formation, and has 3
    free-parameters: metallicity (logzh), dust (log E(B-V)), and log_Npix.

    Corresponds to the `~pcmdpy.galaxy.Const_SFR` model

    Methods
    ---------
    lnprior(params)
        Compute log of prior for given parameters
    prior_transform(normed_params)
        Convert normalized parameters [0,1] to physical parameters
        using the inverse of the prior

    """

    def __init__(self, z_bound=[-2., 0.5], dust_bound=[-3., 0.5],
                 npix_bound=[-1., 8.]):
        """
        Yields a `ConstFlatPrior` object with specified bounds in each
        dimension.

        Parameters
        ----------
        z_bound : array-like (length 2), optional
             lower (default -2.) and upper (default 0.5) bounds of metallicity,
             in units log_10(z/z_solar).
        dust_bound : array-like (length 2), optional
             lower (default -3.) and upper (default 0.5) bounds of dust
             extinction, in units log_10 E(B-V).
        npix_bound : array-like (length 2), optional
             lower (default -1.) and upper (default 8.) bounds of
             star-per-pixel, in units log_10 N_pix
        
        Yields
        ------
        `ConstFlatPrior`
             Object representing a flat prior with given bounds

        Raises
        ------
        `ValueError`
            If any key-word arguments are not array-like with length 2
            OR if upper bounds are lesser than lower bounds in any dimension
        """
        bounds = np.array([z_bound, dust_bound, npix_bound])
        FlatPrior.__init__(self, bounds)

        
class TauFlatPrior(FlatPrior):
    """
    A `FlatPrior` object representing a 7-part SFH (Star Formation
    History) that assumes a tau-model star-formation history, and has 4
    free-parameters: metallicity (logzh), dust (log E(B-V)), log_Npix,
    and tau (in Gyrs).

    Corresponds to the `~pcmdpy.galaxy.Tau_Model` model

    Methods
    ---------
    lnprior(params)
        Compute log of prior for given parameters
    prior_transform(normed_params)
        Convert normalized parameters [0,1] to physical parameters
        using the inverse of the prior

    """

    def __init__(self, z_bound=[-2., 0.5], dust_bound=[-3., 0.5],
                 npix_bound=[-1., 8.], tau_bound=[.1, 20.]):
        """
        Yields a `TauFlatPrior` object with specified bounds in each dimension.

        Parameters
        ----------
        z_bound : array-like with shape (2,), optional
             lower (default -2.) and upper (default 0.5) bounds of metallicity,
             in units log_10(z/z_solar).
        dust_bound : array-like with shape (2,), optional
             lower (default -3.) and upper (default 0.5) bounds of dust
             extinction, in units log_10 E(B-V).
        npix_bound : array-like with shape (2,), optional
             lower (default -1.) and upper (default 8.) bounds of
             star-per-pixel, in units log_10 N_pix
        tau_bound : array-like with shape (2,), optional
             lower (default 0.1) and upper (default 20.) bounds of tau, the
             star-formation timescale, in units of Gyrs

        Yields
        ------
        `TauFlatPrior`
             Object representing a flat prior with given bounds

        Raises
        ------
        Value Error
            If any key-word arguments are not array-like with length 2
            OR if upper bounds are lesser than lower bounds in any dimension
        """
        bounds = np.array([z_bound, dust_bound, npix_bound, tau_bound])
        FlatPrior.__init__(self, bounds)

        
class TauMDFFlatPrior(FlatPrior):
    """
    A `FlatPrior` object representing a 7-part SFH (Star Formation
    History) that assumes a tau-model star-formation history, and has 4
    free-parameters: metallicity (logzh), dust (log E(B-V)), log_Npix,
    and tau (in Gyrs).

    Corresponds to the `~pcmdpy.galaxy.Tau_Model` model

    Methods
    ---------
    lnprior(params)
        Compute log of prior for given parameters
    prior_transform(normed_params)
        Convert normalized parameters [0,1] to physical parameters
        using the inverse of the prior

    """

    def __init__(self, z_bound=[-2., 0.5], sigz_bound=[0., 1.],
                 dust_bound=[-3., 0.5], npix_bound=[-1., 8.],
                 tau_bound=[.1, 20.]):
        """
        Yields a `TauMDFFlatPrior` object with specified bounds in each dimension.

        Parameters
        ----------
        z_bound : array-like with shape (2,), optional
             lower (default -2.) and upper (default 0.5) bounds of metallicity,
             in units log_10(z/z_solar).
        dust_bound : array-like with shape (2,), optional
             lower (default -3.) and upper (default 0.5) bounds of dust
             extinction, in units log_10 E(B-V).
        npix_bound : array-like with shape (2,), optional
             lower (default -1.) and upper (default 8.) bounds of
             star-per-pixel, in units log_10 N_pix
        tau_bound : array-like with shape (2,), optional
             lower (default 0.1) and upper (default 20.) bounds of tau, the
             star-formation timescale, in units of Gyrs

        Yields
        ------
        `TauFlatPrior`
             Object representing a flat prior with given bounds

        Raises
        ------
        Value Error
            If any key-word arguments are not array-like with length 2
            OR if upper bounds are lesser than lower bounds in any dimension
        """
        bounds = np.array([z_bound, sigz_bound, dust_bound, npix_bound,
                           tau_bound])
        FlatPrior.__init__(self, bounds)

        
class FullFlatPrior(FlatPrior):
    """A `FlatPrior` object representing a fully non-parametric, 7-part SFH
    (Star Formation History), which has 9 free-parameters: metallicity (logzh),
    dust (log E(B-V)), and 7 star-formation history bins (log sfhX).

    Corresponds to the `~pcmdpy.galaxy.Galaxy_Model` model

    Methods
    ---------
    lnprior(params)
        Compute log of prior for given parameters
    prior_transform(normed_params)
        Convert normalized parameters [0,1] to physical parameters
        using the inverse of the prior

    """
    def __init__(self, z_bound=[-2., 0.5], dust_bound=[-3., 0.5],
                 sfh0_bound=[-3.1, -1.1], sfh1_bound=[-2.1, -0.1],
                 sfh2_bound=[-1.7, 0.3], sfh3_bound=[-1.2, 0.8],
                 sfh4_bound=[-0.5, 1.5], sfh5_bound=[0.4, 2.4],
                 sfh6_bound=[0.9, 2.9]):
        """
        Yields a `FullFlatPrior` object with specified bounds in each
        dimension.

        Parameters
        ----------
        z_bound : array-like with shape (2,), optional
             lower (default -2.) and upper (default 0.5) bounds of metallicity,
             in units log_10(z/z_solar).
        dust_bound : array-like with shape (2,), optional
             lower (default -3.) and upper (default 0.5) bounds of dust
             extinction, in units log_10 E(B-V).
        sfh0_bound, ... , sfh6_bound : array-like with shape (2,), optional
             lower and upper bounds of star-formation in each age bin, in units
             log_10 M_star.
             default is set for Npix=1e2, tau=5 SFH.

        Yields
        ------
        `FullFlatPrior`
             Object representing a flat prior with given bounds

        Raises
        ------
        `ValueError`
            If any key-word arguments are not array-like with length 2
            OR if upper bounds are lesser than lower bounds in any dimension
        """
        bounds = np.array([z_bound, dust_bound, sfh0_bound, sfh1_bound,
                           sfh2_bound, sfh3_bound, sfh4_bound, sfh5_bound,
                           sfh6_bound])
        FlatPrior.__init__(self, bounds)

