# galaxy.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define the Galaxy_Model class"""

__all__ = ['BaseGalaxy', 'NonParam', 'ConstantSFR', 'TauModel', 'RisingTau',
           'SSPModel', 'NonParamMDF', 'ConstantSFRMDF', 'TauModelMDF',
           'RisingTauMDF']

import numpy as np
from pcmdpy import utils
from scipy.stats import norm


class BaseGalaxy:

    default_edges = np.array([6., 7., 8., 8.5, 9., 9.5, 10., 10.2])
    _num_SFH_bins = len(default_edges) - 1
    default_fehs = np.arange(-4., 0.6, 0.5)
    _num_feh_bins = len(default_fehs) - 1

    _param_names = ['ages', 'fehs', 'SFH', 'dust']

    def __init__(self, ages, fehs, SFH, dust):
        utils.my_assert(len(ages) == len(fehs),
                        "length of first param and second param must match")
        utils.my_assert(len(ages) == len(SFH),
                        "length of first param and third param must match")
        self.ages = ages
        self.fehs = fehs
        self.SFH = SFH
        self.dust = dust
        self.Npix = np.sum(self.SFH)
        self.num_SSPs = len(self.ages)

    def iter_SSPs(self):
        for i in range(self.num_SSPs):
            yield self.ages[i], self.fehs[i], self.SFH[i]

    @classmethod
    def _get_MDF(cls, feh_mean, feh_sig):
        if feh_sig <= 0.:
            return np.array([feh_mean]), np.array([1.])
        lower_feh = feh_mean - 1.0
        upper_feh = feh_mean + 1.0
        in_range = np.logical_and(cls.default_fehs >= lower_feh,
                                  cls.default_fehs <= upper_feh)
        utils.my_assert(np.sum(in_range) > 0,
                        "logfeh0 out of range (+/- 1 dex) of given fehs.")
        iso_fehs = cls.default_fehs[in_range]
        feh_weights = norm.pdf(iso_fehs, loc=feh_mean,
                               scale=feh_sig)
        # this can happen if feh_sig is much smaller than default_fehs spacing
        if np.isclose(np.sum(feh_weights, 0.)):
            return np.array([feh_mean]), np.array([1.])
        else:
            feh_weights /= np.sum(feh_weights)
        return iso_fehs, feh_weights
            

class NonParam(BaseGalaxy):

    _param_names = ['logfeh', 'logdust', 'logSFH0', 'logSFH1', 'logSFH2',
                    'logSFH3', 'logSFH4', 'logSFH5', 'logSFH6']
    _num_params = len(_param_names)

    def __init__(self, gal_params, iso_step=0.2):
        """
        gal_params:
           0 -- log (Fe/H) metallicity in solar units
           1 -- log E(B-V) dust extinction
           2... -- log SFH in age bin
        """
        if iso_step > 0:
            # construct list of ages, given isochrone spacing
            iso_edges = np.arange(6.0, 10.3, iso_step)
            # toss out any bins outside the default 7-bin range
            in_range = np.logical_and((iso_edges+0.001 >=
                                       self.default_edges[0]),
                                      (iso_edges-0.001 <=
                                       self.default_edges[-1]))
            iso_edges = iso_edges[in_range]
        else:
            iso_edges = self.default_edges
        ages = 0.5*(iso_edges[:-1] + iso_edges[1:])
        # which of the 7 bins does each isochrone belong to?
        iso_sfh_bin = np.digitize(ages, self.default_edges) - 1
        # compute SFH in each isochrone, given the bin SFH
        delta_t_iso = np.diff(10.**iso_edges)
        delta_t_sfh = np.diff(10.**self.default_edges)
        SFH = 10.**gal_params[2:][iso_sfh_bin] * delta_t_iso
        SFH /= delta_t_sfh[iso_sfh_bin]
        utils.my_assert(len(gal_params) == self._num_params,
                        "gal_params for Galaxy_Model should be length %d" %
                        self._num_params)
        fehs = np.repeat(gal_params[0], len(ages))
        dust = 10.**gal_params[1]
        super().__init__(ages, fehs, SFH, dust)

        self._params = gal_params

    def as_default(self):
        return NonParam(self._params, iso_step=-1)


class ConstantSFR(BaseGalaxy):

    _param_names = ['logfeh', 'logdust', 'logNpix']
    _num_params = len(_param_names)

    def __init__(self, gal_params, iso_step=0.2):
        """
        gal_params:
           0 -- log (Fe/H) metallicity in solar units
           1 -- log E(B-V) dust extinction
           2 -- log Npix
        iso_step: 
        """
        if iso_step > 0:
            iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            iso_edges = self.default_edges
        ages = 0.5*(iso_edges[1:] + iso_edges[:-1])
        utils.my_assert(len(gal_params) == self._num_params,
                        "gal_params for Constant_SFR should be length %d" %
                        self._num_params)

        fehs = np.repeat(gal_params[0], len(ages))
        dust = 10.**gal_params[1]
        Npix = 10.**gal_params[2]
        
        SFH_term = 10.**iso_edges[1:] - 10.**iso_edges[:-1]
        SFH = Npix * SFH_term / np.sum(SFH_term)
        super().__init__(ages, fehs, SFH, dust)
        
        self._params = gal_params

    def as_default(self):
        return ConstantSFR(self._params, iso_step=-1)


class TauModel(BaseGalaxy):

    _param_names = ['logfeh', 'logdust', 'logNpix', 'tau']
    _num_params = len(_param_names)

    def __init__(self, gal_params, iso_step=0.2):
        """
        gal_params:
           0 -- log (Fe/H) metallicity in solar units
           1 -- log E(B-V) dust extinction
           2 -- log Npix
           3 -- tau (SFH time-scale, in Gyr)
        iso_step: 
        """

        if iso_step > 0:
            iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            iso_edges = self.default_edges
        ages = 0.5*(iso_edges[1:] + iso_edges[:-1])
        utils.my_assert(len(gal_params) == self._num_params,
                        "gal_params for Tau_Model should be length %d" %
                        self._num_params)

        fehs = np.repeat(gal_params[0], len(ages))
        dust = 10.**gal_params[1]
        Npix = 10.**gal_params[2]
        tau = gal_params[3]

        ages_linear = 10.**(iso_edges - 9.)  # convert to Gyrs
        SFH_term = np.exp(ages_linear[1:]/tau) - np.exp(ages_linear[:-1]/tau)
        SFH = Npix * SFH_term / np.sum(SFH_term)
        super().__init__(ages, fehs, SFH, dust)
        self._params = gal_params

    def as_default(self):
        return TauModel(self._params, iso_step=-1)


class RisingTau(BaseGalaxy):

    _param_names = ['logfeh', 'logdust', 'logNpix', 'tau']
    _num_params = len(_param_names)
    
    def __init__(self, gal_params, iso_step=0.2):
        """
        gal_params:
           0 -- log (Fe/H) metallicity in solar units
           1 -- log E(B-V) dust extinction
           2 -- log Npix
           3 -- tau (SFH time-scale, in Gyr)
        iso_step: 
        """

        if iso_step > 0:
            iso_edges = np.arange(6.0, 10.3, iso_step)
        else:
            iso_edges = BaseGalaxy.default_edges
        ages = 0.5*(iso_edges[1:] + iso_edges[:-1])
        utils.my_assert(len(gal_params) == self._num_params,
                        "gal_params for Rising_Tau should be length %d" %
                        self._num_params)
        fehs = np.repeat(gal_params[0], len(ages))
        dust = 10.**gal_params[1]
        Npix = 10.**gal_params[2]
        tau = gal_params[3]

        ages_linear = 10.**(iso_edges - 9.)  # convert to Gyrs
        base_term = (ages_linear[-1]+tau-ages_linear) * np.exp(ages_linear/tau)
        SFH_term = base_term[:-1] - base_term[1:]
        SFH = Npix * SFH_term / np.sum(SFH_term)
        super().__init__(ages, fehs, SFH, dust)
        self._params = gal_params

    def as_default(self):
        return RisingTau(self._params, iso_step=-1)


class SSPModel(BaseGalaxy):
    _param_names = ['logfeh', 'logdust', 'logNpix', 'logage']
    _num_params = len(_param_names)

    def __init__(self, gal_params):
        """
        gal_params:
           0 -- log (Fe/H) metallicity in solar units
           1 -- log E(B-V) dust extinction
           2 -- log Npix
           3 -- log age (in yrs)
        """
        utils.my_assert(len(gal_params) == self._num_params,
                        "gal_params for Galaxy_SSP should be length %d" %
                        self._num_params)
        self._params = gal_params
        fehs = np.array([gal_params[0]])
        dust = 10.**gal_params[1]
        Npix = 10.**gal_params[2]
        SFH = np.array([Npix])
        ages = np.array([gal_params[3]])
        super().__init__(ages, fehs, SFH, dust)


def _make_galaxy_MDF(base_class):
    class MDFGalaxy(base_class):

        _param_names = (['logfeh_mean', 'logfeh_std'] +
                        base_class._param_names[1:])
        _num_params = len(_param_names)
        
        def __init__(self, gal_params, **kwargs):
            utils.my_assert(len(gal_params) == base_class._num_params + 1,
                            "need one additional parameter to make MDF")
            self.feh_mean = gal_params[0]
            self.feh_std = gal_params[1]
            other_params = gal_params[2:]
            iso_fehs, feh_weights = base_class._get_MDF(self.feh_mean,
                                                        self.feh_std)
            num_fehs = len(iso_fehs)
            temp_gal = base_class(np.append(self.feh_mean, other_params),
                                  **kwargs)
            ages_array = temp_gal.ages
            num_ages = len(ages_array)
            SFH_array = temp_gal.SFH
            dust = temp_gal.dust
            ages = np.tile(ages_array, num_fehs)
            SFH = np.outer(SFH_array, feh_weights).T.flatten()
            fehs = np.repeat(iso_fehs, num_ages)

            BaseGalaxy.__init__(self, ages, fehs, SFH, dust)
            self._params = gal_params

        def as_default(self):
            return MDFGalaxy(self._params, iso_step=-1)

    return MDFGalaxy


NonParamMDF = _make_galaxy_MDF(NonParam)
ConstantSFRMDF = _make_galaxy_MDF(ConstantSFR)
TauModelMDF = _make_galaxy_MDF(TauModel)
RisingTauMDF = _make_galaxy_MDF(RisingTau)
