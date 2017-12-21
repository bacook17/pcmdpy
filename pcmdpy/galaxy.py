# galaxy.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define the Galaxy_Model class"""

import numpy as np
from pcmdpy import utils


class Galaxy_Model:

    age_edges = np.array([6., 7., 8., 8.5, 9., 9.5, 10., 10.2])
    _num_ages = len(age_edges) - 1

    _param_names = ['logfeh', 'logdust', 'logSFH0', 'logSFH1', 'logSFH2',
                    'logSFH3', 'logSFH4', 'logSFH5', 'logSFH6']
    _num_params = len(_param_names)
    _meta_names = ['logNpix']
    
    def __init__(self, gal_params, iso_step=0.2):
        """
        gal_params:
           0 -- log (Fe/H) metallicity in solar units
           1 -- log E(B-V) dust extinction
           2... -- log SFH in age bin
        """
        if iso_step > 0:
            self.iso_bins = np.arange(6.0, 10.3, iso_step)
            self.ages = 0.5*(self.iso_bins[:-1] + self.iso_bins[1:])
            in_range = np.logical_and(self.iso_bins+0.001 >= self.age_edges[0],
                                      self.iso_bins-0.001 <= self.age_edges[-1])
            self.iso_bins = self.iso_bins[in_range]
            self.ages = 0.5*(self.iso_bins[:-1] + self.iso_bins[1:])
            iso_sfh_bin = np.digitize(self.ages, self.age_edges) - 1

            delta_t_iso = np.diff(10.**self.iso_bins)
            delta_t_sfh = np.diff(10.**self.age_edges)
            self.SFH = 10.**gal_params[2:] * delta_t_iso
            self.SFH /= delta_t_sfh[iso_sfh_bin]
        else:
            self.ages = 0.5*(self.age_edges[:-1] + self.age_edges[1:])
            self.SFH = 10.**gal_params[2:]

        utils.my_assert(len(gal_params) == self._num_params,
                        "gal_params for Galaxy_Model should be length %d" %
                        self._num_params)

        self.feh = gal_params[0]
        self.dust = 10.**gal_params[1]
        self.Npix = np.sum(self.SFH)
        self._params = gal_params
        self._meta_params = np.array([np.log10(self.Npix)])

        
class Constant_SFR(Galaxy_Model):

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
            self.age_edges = np.arange(6.0, 10.3, iso_step)
        self.ages = 0.5*(self.age_edges[1:] + self.age_edges[:-1])
        utils.my_assert(len(gal_params) == self._num_params,
                        "gal_params for Constant_SFR should be length %d" %
                        self._num_params)

        self.feh = gal_params[0]
        self.dust = 10.**gal_params[1]
        self.Npix = 10.**gal_params[2]
        
        SFH_term = 10.**self.age_edges[1:] - 10.**self.age_edges[:-1]
        self.SFH = self.Npix * SFH_term / np.sum(SFH_term)
        self._params = gal_params

    def as_full(self):
        mock = Constant_SFR(self._params, iso_step=-1)
        params = np.append(self._params[:2], np.log10(mock.SFH))
        return Galaxy_Model(params, iso_step=-1)

        
class Tau_Model(Galaxy_Model):

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
            self.age_edges = np.arange(6.0, 10.3, iso_step)
        self.ages = 0.5*(self.age_edges[1:] + self.age_edges[:-1])
        utils.my_assert(len(gal_params) == self._num_params,
                        "gal_params for Tau_Model should be length %d" %
                        self._num_params)
        self.feh = gal_params[0]
        self.dust = 10.**gal_params[1]
        self.Npix = 10.**gal_params[2]
        tau = gal_params[3]

        ages_linear = 10.**(self.age_edges - 9.)  # convert to Gyrs
        SFH_term = np.exp(ages_linear[1:]/tau) - np.exp(ages_linear[:-1]/tau)
        self.SFH = self.Npix * SFH_term / np.sum(SFH_term)
        self._params = gal_params

    def as_full(self):
        mock = Tau_Model(self._params, iso_step=-1)
        params = np.append(self._params[:2], np.log10(mock.SFH))
        return Galaxy_Model(params, iso_step=-1)
        

class Rising_Tau(Galaxy_Model):

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
            self.age_edges = np.arange(6.0, 10.3, iso_step)
        self.ages = 0.5*(self.age_edges[1:] + self.age_edges[:-1])
        utils.my_assert(len(gal_params) == self._num_params,
                        "gal_params for Rising_Tau should be length %d" %
                        self._num_params)
        self.feh = gal_params[0]
        self.dust = 10.**gal_params[1]
        self.Npix = 10.**gal_params[2]
        tau = gal_params[3]

        ages_linear = 10.**(self.age_edges - 9.)  # convert to Gyrs
        base_term = (ages_linear[-1]+tau-ages_linear) * np.exp(ages_linear/tau)
        SFH_term = base_term[:-1] - base_term[1:]
        self.SFH = self.Npix * SFH_term / np.sum(SFH_term)
        self._params = gal_params

    def as_full(self):
        mock = Rising_Tau(self._params, iso_step=-1)
        params = np.append(self._params[:2], np.log10(mock.SFH))
        return Galaxy_Model(params, iso_step=-1)


class Galaxy_SSP:
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
        self.feh = gal_params[0]
        self.dust = 10.**gal_params[1]
        Npix = 10.**gal_params[2]
        self.SFH = np.array([Npix])
        self.ages = np.array([gal_params[3]])
        self._num_ages = 1
    
