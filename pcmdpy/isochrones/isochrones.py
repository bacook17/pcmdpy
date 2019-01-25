# isochrones.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define the Isocrhone_Model class"""

import numpy as np
import pandas as pd
import os
import glob
import sys
from warnings import warn
from pkg_resources import resource_filename

##########################
# Useful Utilities


def _interp_arrays(arr1, arr2, f):
    """Linearly interpolate between two (potentially unequal length) arrays
    
    Arguments:
    arr1 -- first (lower) array (len N1 or N1xD)
    arr2 -- second (upper) array (len N2 or N2xD, N2 doesn't have to equal N1)
    f -- linear interpolation fraction (float between 0 and 1)
    Output: interpolated array (len max(N1,N2) or max(N1,N2)xD)
    """
    assert (arr1.ndim == arr2.ndim), (
        "The two interpolated arrays must have same dimensions")

    l1, l2 = len(arr1), len(arr2)
    # If arrays are unequal length, extrapolate shorter using trend of longer
    if (l1 < l2):
        delta = arr2[l1:] - arr2[l1-1]
        added = arr1[-1] + delta
        arr1 = np.append(arr1, added, axis=0)
    elif (l1 > l2):
        delta = arr1[l2:] - arr1[l2-1]
        added = arr2[-1] + delta
        arr2 = np.append(arr2, added, axis=0)
    return (1-f)*arr1 + f*arr2


def _feh_from_str(feh_str):
    """Converts a metallicity value to MIST string
    Example Usage:
    _feh_from_str("m0.53") -> -0.53
    _feh_from_str("p1.326")   -> 1.326
    
    Arguments:
    feh_str -- metallicity (as a string)
    Output: float value of metallicity
    """
    value = float(feh_str[1:])
    if feh_str[0] == 'm':
        value *= -1
    elif feh_str[0] != 'p':
        raise ValueError('feh string not of valid format')
    return value


def _feh_to_str(feh):
    """Converts a metallicity value to MIST string
    Example Usage:
    _feh_to_str(-0.5313) -> "m0.53"
    _feh_to_str(1.326)   -> "p1.33"
    
    Arguments:
    feh -- metallicity (float)
    Output: string representing metallicity
    """
    result = ''
    if (feh < 0):
        result += 'm'
    else:
        result += 'p'
    result += '%1.2f' % (np.abs(feh))
    return result


def _interp_df_by_mass(df, dm_min):
    ages = np.unique(df.age.values)
    fehs = np.unique(df['[Fe/H]_init'].values)
    new_rows = []
    for age in ages:
        for feh in fehs:
            iso_df = df[np.isclose(df.age, age) & np.isclose(df['[Fe/H]_init'], feh)]
            # add more points until reached desired spacing
            mass = iso_df.initial_mass.values
            frac_dm = np.diff(mass) / mass[:-1]
            id_too_large = np.where(frac_dm > dm_min)[0]
            for i_max in id_too_large:
                # add additional 5 points spacing by interpolating 0.1 between points
                row_low = iso_df.iloc[i_max]
                row_high = iso_df.iloc[i_max + 1]
                for f in np.linspace(0.1, 0.9, 5):
                    new_rows.append(f*row_low + (1-f)*row_high)
    df = df.append(pd.DataFrame(new_rows))
    return df


class Isochrone_Model:
    """Models Isochrones (IMF, and magnitudes in particular Filters) using
       linear interpolation of MIST models

    An Isocrhone_Model incorporates a collection of MIST models, and
       allows for interpolating the IMF and magnitudes (for given Filter
       objects) at any arbitrary metallicity and mass

    Attributes:
       MIST_df-- A pandas Dataframe containing all pre-computed MIST datapoints
       ages -- An array of ages (in log years) which are valid for the model
    Methods:
       get_magnitudes -- Pass a Galaxy_Model object, return IMF and magnitudes
                         for each mass, age, metallicity bin
    Constructors:
       __init__ -- Pass a list of Filter objects, path to MIST model files,
                   and array of metallicities.
    """
    def __init__(self, filters, MIST_path=None, iso_append=".iso.cmd",
                 mag_system='vega', dm_interp=-1):
        """Creates a new Isochrone_Model, given a list of Filter objects
        
        Arguments:
           filters -- list of Filter objects
        Keyword Arguments:
           MIST_path -- directory containing MIST model files
           feh_arr -- array of MIST metallicity values to use
           dm_interp -- 
        """

        # Locate MIST files
        if MIST_path is None:
            MIST_path = resource_filename('pcmdpy', 'isochrones/MIST_v1.2/')
        
        # Import all MIST model files into Pandas dataframe
        self.MIST_df = pd.DataFrame()
        self.num_filters = len(filters)

        # Use optional conversions from VEGA to AB or ST, etc
        self.conversions = {}
        self.conversions['vega'] = np.zeros(len(filters), dtype=float)
        self.conversions['ab'] = np.array([f._zpts['ab'] - f._zpts['vega']
                                           for f in filters])
        self.conversions['st'] = np.array([f._zpts['ab'] - f._zpts['vega']
                                           for f in filters])
        self.default_system = mag_system.lower()
        assert self.default_system in self.conversions.keys(), (
            "the given mag_system is not valid. Please choose one of: "
            "['vega', 'ab', 'st']")
        
        _feh_arr = []
        self.filters = filters
        self.filter_names = [f.tex_name for f in self.filters]
        self.colnames = pd.read_table(MIST_path + 'columns.dat',
                                      delim_whitespace=True).columns
        # load all MIST files found in directory
        for MIST_doc in glob.glob(os.path.join(MIST_path, '*'+iso_append)):
            try:
                feh_str = MIST_doc.split('feh_')[-1][:5]
                feh = _feh_from_str(feh_str)
                new_df = pd.read_table(MIST_doc, names=self.colnames,
                                       comment='#', delim_whitespace=True,
                                       dtype=float, na_values=['Infinity'])
                new_df[new_df.isna()] = 100.
                new_df['feh'] = feh
                self.MIST_df = self.MIST_df.append([new_df], ignore_index=True)
                _feh_arr.append(_feh_from_str(feh_str))
            except Exception:
                warn('File not properly formatted: %s' % (MIST_doc))
                sys.exit(1)
            
        self._feh_arr = np.sort(_feh_arr)
        self.MIST_df.rename(columns={'log10_isochrone_age_yr': 'age'},
                            inplace=True)
        if dm_interp > 0.:
            print('starting manual interpolation')
            self.MIST_df = _interp_df_by_mass(self.MIST_df, dm_interp)
            print('done with interpolation')
        self.MIST_df = self.MIST_df.sort_values(by=['[Fe/H]_init', 'age',
                                                    'initial_mass'])
        self.MIST_df = self.MIST_df.reset_index(drop=True)
        self.ages = self.MIST_df.age.unique()
        # The MIST columns that will be interpolated (mass, logIMF,
        # and all input filters)
        self._interp_cols = ['initial_mass', 'star_mass']
        for f in self.filters:
            c = f.MIST_column
            c_alt = f.MIST_column_alt
            if c in self.MIST_df.columns:
                self._interp_cols.append(c)
            elif c_alt in self.MIST_df.columns:
                self._interp_cols.append(c_alt)
            else:
                print((c, c_alt))
                raise ValueError('Filter does not have a valid MIST_column')
        self.MIST_gb = self.MIST_column.groupby(['age', 'feh'])[self._interp_cols]
        return None
    
    def get_isochrone(self, age, feh, downsample=5, mag_system=None, use_gb=False):
        """Interpolate MIST isochrones for given age and metallicity
        
        Arguments:
           age ---
           feh ---
           downsample ---
           mag_system ---
        Output:
           mags -- 2D array of magnitudes (DxN, where D is number of filters
                   the model was initialized with)
           imass -- array of initial masses (N)
           cmass -- array of current masses (N)
        """

        mag_system = mag_system or self.default_system
        mag_system = mag_system.lower()
        if mag_system not in self.conversions.keys():
            warn(('mag_system {0:s} not in list of magnitude '
                  'conversions. Reverting to Vega'.format(mag_system)))
            conversions = self.conversions['vega']
        else:
            conversions = self.conversions[mag_system]
        
        # Find closest age in MIST database
        if age not in self.ages:
            age = self.ages[np.abs(self.ages - age).argmin()]
        if feh in self._feh_arr:
            if use_gb:
                inter = self.MIST_gb.get_group((age, feh))
            else:
                this_age = self.MIST_df[np.isclose(self.MIST_df.age.values, age)]
                inter = this_age[np.isclose(this_age.feh.values, feh)][self._interp_cols].values
        # Interpolate/extrapolate for other metallicities
        else:
            this_age = self.MIST_df[np.isclose(self.MIST_df.age.values, age)]
            i = self._feh_arr.searchsorted(feh)
            if (i == 0):
                i = 1  # will extrapolate low
            elif (i == len(self._feh_arr)):
                i = -1  # will extrapolate high
            fehlow, fehhigh = self._feh_arr[i-1:i+1]  # bounding metallicities
            frac_between = (feh - fehlow) / (fehhigh - fehlow)
            if (frac_between >= 2) or (frac_between <= -1):
                raise ValueError('Extrapolating metallicity more than one '
                                 'entire metallicity bin')
            dflow = this_age[np.isclose(this_age.feh.values, fehlow)][self._interp_cols]
            dfhigh = this_age[np.isclose(this_age.feh.values, fehhigh)][self._interp_cols]
            inter = _interp_arrays(dflow.values, dfhigh.values, frac_between)
            
        initial_mass = inter[::downsample, 0]
        current_mass = inter[::downsample, 1]

        mags = (inter[::downsample, 2:] + conversions).T
        
        return mags, initial_mass, current_mass

    def model_galaxy(self, galaxy, lum_cut=np.inf, mag_system=None,
                     downsample=5, return_mass=False):
        weights = np.empty((1, 0), dtype=float)
        magnitudes = np.empty((self.num_filters, 0), dtype=float)
        initial_mass = np.empty((1, 0), dtype=float)
        current_mass = np.empty((1, 0), dtype=float)
        # Collect the isochrones from each bin
        for age, feh, sfh, d_mod in galaxy.iter_SSPs():
            mags, i_mass, c_mass = self.get_isochrone(age, feh,
                                                      mag_system=mag_system,
                                                      downsample=downsample)
            imf = galaxy.imf_func(i_mass, **galaxy.imf_kwargs)
            weights = np.append(weights, imf*sfh)
            mags += d_mod
            magnitudes = np.append(magnitudes, mags, axis=-1)
            initial_mass = np.append(initial_mass, i_mass)
            current_mass = np.append(current_mass, c_mass)
        if not np.isinf(lum_cut):
            lum = np.power(10., -0.4*magnitudes)
            mean_lum = np.average(lum, weights=weights, axis=1)
            to_keep = (lum.T / mean_lum >= lum_cut).sum(axis=1) == 0
            weights = weights[to_keep]
            magnitudes = magnitudes[:, to_keep]
            initial_mass = initial_mass[to_keep]
            current_mass = current_mass[to_keep]
        if return_mass:
            return weights, magnitudes, initial_mass, current_mass
        else:
            return weights, magnitudes

    def get_stellar_mass(self, galaxy, downsample=5):
        imf, _, _, c_mass = self.model_galaxy(galaxy, downsample=downsample,
                                              return_mass=True)
        return (imf * c_mass).sum()
