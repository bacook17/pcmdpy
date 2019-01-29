# galaxy.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define the Galaxy_Model class"""

__all__ = ['CustomGalaxy', 'SSPSimple', 'TauSimple',
           'NonParamSimple', 'TauFull', 'NonParamFull']

import numpy as np
from ..sampling import priors
from .metalmodels import (BaseMetalModel, get_metal_model)
from .sfhmodels import (BaseSFHModel, get_sfh_model)
from .distancemodels import (BaseDistanceModel, get_distance_model)
from .dustmodels import (BaseDustModel, get_dust_model)
from .imf import (salpeter_IMF, kroupa_IMF, salpeter_meanmass,
                  kroupa_meanmass)


class CustomGalaxy:

    def __init__(self, metal_model, dust_model, sfh_model, distance_model,
                 mdf_sig=None,
                 dust_sig=None,
                 dmod=None,
                 imf='salpeter',
                 imf_kwargs={},
                 initial_params=None):
        # set the metallicity model
        if not isinstance(metal_model, BaseMetalModel):
            kwargs = {}
            if mdf_sig is not None:
                kwargs['sig'] = mdf_sig
            metal_model = get_metal_model(metal_model, **kwargs)  # parse a passed string
        self.metal_model = metal_model

        # set the dust model
        if not isinstance(dust_model, BaseDustModel):
            kwargs = {}
            if dust_sig is not None:
                kwargs['sig'] = dust_sig
            dust_model = get_dust_model(dust_model, **kwargs)
        self.dust_model = dust_model

        # set the SFH model
        if not isinstance(sfh_model, BaseSFHModel):
            sfh_model = get_sfh_model(sfh_model)
        self.sfh_model = sfh_model

        # set the distance modulus
        if not isinstance(distance_model, BaseDistanceModel):
            kwargs = {}
            if dmod is not None:
                kwargs['dmod'] = dmod
            distance_model = get_distance_model(distance_model, **kwargs)
        self.distance_model = distance_model

        # set the IMF model
        self._imf = imf
        if imf.lower() == 'salpeter':
            self.imf_func = salpeter_IMF
            self.meanmass = salpeter_meanmass(**imf_kwargs)
            self.imf_kwargs = imf_kwargs
        elif imf.lower() == 'kroupa':
            self.imf_func = kroupa_IMF
            self.meanmass = kroupa_meanmass(**imf_kwargs)
            self.imf_kwargs = imf_kwargs
        else:
            raise NotImplementedError('Only salpeter and kroupa '
                                      'IMFs are currently implemented')
        
        if initial_params is not None:
            self.set_params(initial_params)
        else:
            if None not in self._params:
                self.set_params(self._params)

    def set_params(self, gal_params):
        # make sure is array, with right length
        assert(len(gal_params) == self.p_total)
        gal_params = np.array(gal_params)
        # set metal parameters
        feh_params = gal_params[:self.p_feh]
        self.metal_model.set_params(feh_params)

        # set dust parameters
        dust_params = gal_params[self.p_feh:self.p_feh+self.p_dust]
        self.dust_model.set_params(dust_params)

        # set sfh parameters
        sfh_params = gal_params[self.p_feh+self.p_dust:
                                self.p_feh+self.p_dust+self.p_sfh]
        self.sfh_model.set_params(sfh_params)
        
        # set distance parameters
        if self.p_distance > 0:
            dist_mod = gal_params[-self.p_distance:]
            self.distance_model.set_params(dist_mod)
            
    def get_flat_prior(self, feh_bounds=None, dust_bounds=None,
                       sfh_bounds=None, dmod_bounds=None, **kwargs):
        if feh_bounds is None:
            bounds = self.metal_model._default_prior_bounds
        else:
            assert(len(feh_bounds) == self.p_feh)
            bounds = feh_bounds
        if dust_bounds is None:
            bounds += self.dust_model._default_prior_bounds
        else:
            assert(len(dust_bounds) == self.p_dust)
            bounds += dust_bounds
        # for backwards compatability, allow "age_bounds" in place of "sfh_bounds"
        sfh_bounds = sfh_bounds or kwargs.get('age_bounds', None)
        if sfh_bounds is None:
            bounds += self.sfh_model._default_prior_bounds
        else:
            assert(len(sfh_bounds) == self.p_sfh)
            bounds += sfh_bounds
        if dmod_bounds is None:
            bounds += self.distance_model._default_prior_bounds
        else:
            assert len(dmod_bounds) == self.p_distance
            bounds += dmod_bounds
        return priors.FlatPrior(bounds)

    def copy(self):
        new_gal = CustomGalaxy(
            self.metal_model.copy(),
            self.dust_model.copy(),
            self.sfh_model.copy(),
            self.distance_model.copy(),
            imf=self._imf,
            imf_kwargs=self._imf_kwargs,
            initial_params=self._params)
        return new_gal
                               
    def describe(self):
        pass

    @property
    def ages(self):
        return np.tile(self.sfh_model.ages, self.metal_model._num_fehs)

    @property
    def delta_ts(self):
        return np.tile(self.sfh_model.delta_ts,
                       self.metal_model._num_feh_bins)

    @property
    def fehs(self):
        return np.repeat(self.metal_model.fehs, self.sfh_model._num_isochrones)

    @property
    def SFH(self):
        _, feh_weights = self.metal_model.get_vals()
        _, sfh_weights = self.sfh_model.get_vals()
        return np.outer(feh_weights, sfh_weights).flatten()

    @property
    def dmod(self):
        return self.distance_model.dmod
    
    @property
    def d_mpc(self):
        """
        Distance to galaxy (in Mpc)
        """
        return 10.**(0.2 * (self.dmod - 25.))

    @property
    def Npix(self):
        """
        Number of stars formed per pixel
        """
        return np.sum(self.SFH)

    @property
    def Mpix(self):
        """
        Mass of stars formed (in solar masses) per pixel
        """
        return self.Npix * self.meanmass

    @property
    def logSFH(self):
        """
        Log10 number of stars formed in each age bin, per pixel
        """
        return np.log10(self.SFH)

    @property
    def logNpix(self):
        """
        Log10 number of stars formed per pixel
        """
        return np.log10(self.Npix)

    @property
    def logMpix(self):
        """
        Log10 mass of stars (in stellar masses) formed per pixel
        """
        return np.log10(self.Mpix)

    @property
    def SFR(self):
        """
        Star-formation rate (in solar masses per Gyr) per pixel
        """
        return self.SFH * self.meanmass / self.delta_ts

    @property
    def logSFR(self):
        """
        Log10 star-formation rate (in solar masses per Gyr) per pixel
        """
        return np.log10(self.SFR)
    
    @property
    def num_SSPs(self):
        """
        Number of individual SSPs making up the galaxy model
        """
        return len(self.ages)

    def iter_SSPs(self):
        """
        Iterate through all SSPs making up the galaxy model
        
        Yields
        ------
        age      :
        feh      :
        SFH      :
        dmod :
        """
        for i in range(self.num_SSPs):
            yield self.ages[i], self.fehs[i], self.SFH[i], self.dmod

    @property
    def p_feh(self):
        return self.metal_model._num_params

    @property
    def p_dust(self):
        return self.dust_model._num_params

    @property
    def p_sfh(self):
        return self.sfh_model._num_params

    @property
    def p_distance(self):
        return self.distance_model._num_params

    @property
    def p_total(self):
        return self.p_feh + self.p_dust + self.p_sfh + self.p_distance

    @property
    def _params(self):
        all_params = []
        for mod in [self.metal_model, self.dust_model, self.sfh_model,
                    self.distance_model]:
            all_params += list(mod._params)
        return all_params

    @property
    def _num_params(self):
        return len(self._params)

    @property
    def _param_names(self):
        all_names = []
        for mod in [self.metal_model, self.dust_model, self.sfh_model,
                    self.distance_model]:
            all_names += list(mod._param_names)
        return all_names
        
    @property
    def _fancy_names(self):
        all_names = []
        for mod in [self.metal_model, self.dust_model, self.sfh_model,
                    self.distance_model]:
            all_names += list(mod._fancy_names)
        return all_names

    
class TauSimple(CustomGalaxy):

    def __init__(self, initial_params=None, dmod=30.):
        super().__init__(
            metal_model='single',
            dust_model='single',
            sfh_model='tau',
            distance_model='fixed',
            dmod=dmod,
            initial_params=initial_params)


class SSPSimple(CustomGalaxy):
    
    def __init__(self, initial_params=None, dmod=30.):
        super().__init__(
            metal_model='single',
            dust_model='single',
            sfh_model='ssp',
            distance_model='fixed',
            dmod=dmod,
            initial_params=initial_params)


class NonParamSimple(CustomGalaxy):

    def __init__(self, initial_params=None, dmod=30.):
        super().__init__(
            metal_model='single',
            dust_model='single',
            sfh_model='nonparam',
            distance_model='fixed',
            dmod=dmod,
            initial_params=initial_params)


class TauFull(CustomGalaxy):

    def __init__(self, initial_params=None):
        super().__init__(
            metal_model='fixedwidth',
            dust_model='fixedwidth',
            sfh_model='tau',
            distance_model='variable',
            mdf_sig=0.3,
            dust_sig=0.2,
            initial_params=initial_params)


class NonParamFull(CustomGalaxy):

    def __init__(self, initial_params=None):
        super().__init__(
            metal_model='fixedwidth',
            dust_model='fixedwidth',
            sfh_model='nonparam',
            distance_model='variable',
            mdf_sig=0.3,
            dust_sig=0.2,
            initial_params=initial_params)
