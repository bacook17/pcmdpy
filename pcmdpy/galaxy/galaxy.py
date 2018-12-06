# galaxy.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define the Galaxy_Model class"""

__all__ = ['BaseGalaxy', 'CustomGalaxy', 'SSPSimple', 'TauSimple',
           'NonParamSimple', 'TauFull', 'NonParamFull']

import numpy as np
from ..sampling import priors
from .metalmodels import (BaseMetalModel, get_metal_model)
from .sfhmodels import (BaseSFHModel, get_sfh_model)
from .distancemodels import (BaseDistanceModel, get_distance_model)
from .dustmodels import (BaseDustModel, get_dust_model)


class BaseGalaxy:

    _param_names = ['ages', 'fehs', 'SFH', 'dust_model']
    ages = []
    fehs = []
    SFH = []

    def __init__(self, ages, fehs, SFH, dust_model, dist_mod, params=None):
        assert (len(ages) == len(fehs)), (
            "length of first param and second param must match")
        assert (len(ages) == len(SFH)), (
            "length of first param and third param must match")
        self.ages = ages
        self.fehs = fehs
        self.SFH = SFH
        assert isinstance(dust_model, BaseDustModel), (
            "the dust_model is not a valid _DustModel object")
        self.dust_model = dust_model
        self.dist_mod = dist_mod

    @property
    def Npix(self):
        return np.sum(self.SFH)
        
    @property
    def logSFH(self):
        return np.log10(self.SFH)

    @property
    def logNpix(self):
        return np.log10(self.Npix)

    @property
    def num_SSPs(self):
        return len(self.ages)

    def iter_SSPs(self):
        for i in range(self.num_SSPs):
            yield self.ages[i], self.fehs[i], self.SFH[i], self.dist_mod

    def get_dmpc(self):
        return 10.**(0.2 * (self.dist_mod - 25.))


class CustomGalaxy(BaseGalaxy):

    def __init__(self, metal_model, dust_model, sfh_model, distance_model,
                 mdf_sig=None,
                 dust_sig=None,
                 dmod=None,
                 initial_params=None):
        # set the metallicity model
        if not isinstance(metal_model, BaseMetalModel):
            kwargs = {}
            if mdf_sig is not None:
                kwargs['sig'] = mdf_sig
            metal_model = get_metal_model(metal_model, **kwargs)  # parse a passed string
        self.metal_model = metal_model
        self.p_feh = metal_model._num_params
        self._param_names = list(metal_model._param_names)

        # set the dust model
        if not isinstance(dust_model, BaseDustModel):
            kwargs = {}
            if dust_sig is not None:
                kwargs['sig'] = dust_sig
            dust_model = get_dust_model(dust_model, **kwargs)
        self.dust_model = dust_model
        self.p_dust = dust_model._num_params
        self._param_names += dust_model._param_names

        # set the SFH model
        if not isinstance(sfh_model, BaseSFHModel):
            sfh_model = get_sfh_model(sfh_model)
        self.sfh_model = sfh_model
        self.p_sfh = sfh_model._num_params
        self._param_names += sfh_model._param_names

        # set the distance modulus
        if not isinstance(distance_model, BaseDistanceModel):
            kwargs = {}
            if dmod is not None:
                kwargs['dmod'] = dmod
            distance_model = get_distance_model(distance_model, **kwargs)
        self.distance_model = distance_model
        self.p_distance = distance_model._num_params
        self._param_names += distance_model._param_names
        
        self.p_total = self.p_feh + self.p_dust + self.p_sfh + self.p_distance
        self._num_params = len(self._param_names)
        assert self._num_params == self.p_total, ('galaxy parameter mismatch')
        if initial_params is not None:
            self.set_params(initial_params)
        else:
            if None not in self._params:
                self.set_params(self._params)

    @property
    def _params(self):
        all_params = []
        for mod in [self.metal_model, self.dust_model, self.sfh_model,
                    self.distance_model]:
            all_params += list(mod._params)
        return all_params

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

    def set_params(self, gal_params):
        # make sure is array, with right length
        assert(len(gal_params) == self.p_total)
        gal_params = np.array(gal_params)
        # set metal parameters
        feh_params = gal_params[:self.p_feh]
        self.metal_model.set_params(feh_params)
        fehs, feh_weights = self.metal_model.get_vals()

        # set dust parameters
        dust_params = gal_params[self.p_feh:self.p_feh+self.p_dust]
        self.dust_model.set_params(dust_params)

        # set sfh parameters
        sfh_params = gal_params[self.p_feh+self.p_dust:
                                self.p_feh+self.p_dust+self.p_sfh]
        self.sfh_model.set_params(sfh_params)
        ages, sfh_weights = self.sfh_model.get_vals()

        # set distance parameters
        if self.p_distance > 0:
            dist_mod = gal_params[-self.p_distance]
            self.distance_model.set_params(dist_mod)
        else:
            dist_mod = self.distance_model.dmod
        # merge the age and metallicity bins
        new_ages = []
        new_fehs = []
        SFH = []
        for i, feh in enumerate(fehs):
            SFH += list(sfh_weights * feh_weights[i])
            new_ages += list(ages)
            new_fehs += [feh]*len(ages)

        super().__init__(new_ages, new_fehs, SFH, self.dust_model, dist_mod)

    def describe(self):
        pass


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
