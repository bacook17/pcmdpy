# galaxy.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define the Galaxy_Model class"""

import numpy as np

class Galaxy_Model:

    age_edges = np.array([6., 7., 8., 8.5, 9., 9.5, 10., 10.2])
    age_arr = np.array([6.5, 7.5, 8.25, 8.75, 9.25, 9.75, 10.1])
    _num_ages = len(age_arr)

    _param_names = ['logz', 'logdust', 'logSFH0', 'logSFH1', 'logSFH2', 'logSFH3', 'logSFH4', 'logSFH5', 'logSFH6']
    _num_params = len(_param_names)
    _meta_names = ['logNpix']
    
    def __init__(self, gal_params):
        """
        gal_params:
           0 -- log (z / z_solar) metallicity
           1 -- log E(B-V) dust extinction
           2... -- log SFH in age bin
        """
        self.ages = self.age_arr
        assert(len(gal_params) == self._num_params)

        self.z = gal_params[0]
        self.dust = 10.**gal_params[1]
        self.SFH = 10.**gal_params[2:]
        self.Npix = np.sum(self.SFH)
        self._params = gal_params
        self._meta_params = np.array([np.log10(self.Npix)])

class Constant_SFR(Galaxy_Model):

    _param_names = ['logz', 'logdust', 'logNpix']
    _num_params = len(_param_names)
    def __init__(self, gal_params):
        """
        gal_params:
           0 -- log (z / z_solar) metallicity
           1 -- log E(B-V) dust extinction
           2 -- log Npix
        """
        self.ages = self.age_arr
        assert(len(gal_params) == 3)

        self.z = gal_params[0]
        self.dust = 10.**gal_params[1]
        self.Npix = 10.**gal_params[2]
        
        SFH_term = 10.**self.age_edges[1:] - 10.**self.age_edges[:-1]
        self.SFH = self.Npix * SFH_term / np.sum(SFH_term)
        self._params = gal_params

class Tau_Model(Galaxy_Model):

    _param_names = ['logz', 'logdust', 'logNpix', 'tau']
    _num_params = len(_param_names)
    def __init__(self, gal_params):
        """
        gal_params:
           0 -- log (z / z_solar) metallicity
           1 -- log E(B-V) dust extinction
           2 -- log Npix
           3 -- tau (SFH time-scale, in Gyr)
        """

        self.ages = self.age_arr
        assert(len(gal_params) == 4)

        self.z = gal_params[0]
        self.dust = 10.**gal_params[1]
        self.Npix = 10.**gal_params[2]
        tau = gal_params[3]

        ages_linear= 10.**(self.age_edges - 9.) #convert to Gyrs
        SFH_term = np.exp(ages_linear[1:] / tau) - np.exp(ages_linear[:-1] / tau)
        self.SFH = self.Npix * SFH_term / np.sum(SFH_term)
        self._params = gal_params

class Rising_Tau(Galaxy_Model):

    _param_names = ['logz', 'logdust', 'logNpix', 'tau']
    _num_params = len(_param_names)
    def __init__(self, gal_params):
        """
        gal_params:
           0 -- log (z / z_solar) metallicity
           1 -- log E(B-V) dust extinction
           2 -- log Npix
           3 -- tau (SFH time-scale, in Gyr)
        """

        self.ages = self.age_arr
        assert(len(gal_params) == 4)

        self.z = gal_params[0]
        self.dust = 10.**gal_params[1]
        self.Npix = 10.**gal_params[2]
        tau = gal_params[3]

        ages_linear= 10.**(self.age_edges - 9.) #convert to Gyrs
        base_term = (ages_linear[-1] + tau - ages_linear) * np.exp(ages_linear / tau)
        SFH_term = base_term[:-1] - base_term[1:]
        self.SFH = self.Npix * SFH_term / np.sum(SFH_term)
        self._params = gal_params

class Galaxy_SSP:
    _param_names = ['logz', 'logdust', 'logNpix', 'logage']
    _num_params = len(_param_names)

    def __init__(self, gal_params):
        """
        gal_params:
           0 -- log (z / z_solar) metallicity
           1 -- log E(B-V) dust extinction
           2 -- log Npix
           3 -- log age (in yrs)
        """
        assert(len(gal_params) == self._num_params)
        self._params = gal_params
        self.z = gal_params[0]
        self.dust = 10.**gal_params[1]
        Npix = 10.**gal_params[2]
        self.SFH = np.array([Npix])
        self.ages = np.array([gal_params[3]])
        self._num_ages = 1
    
