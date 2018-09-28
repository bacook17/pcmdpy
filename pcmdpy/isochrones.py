# isochrones.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define the Isocrhone_Model class"""

import numpy as np
import pandas as pd
from pcmdpy import utils
import os
import glob
import sys
from warnings import warn
from pkg_resources import resource_filename

##########################
# Useful Utilities


def salpeter_IMF(mass, lower=0.08, upper=300., norm_by_mass=True, **kwargs):
    mids = 0.5 * (mass[1:] + mass[:-1])  # midpoints between masses
    m_low = np.append([mass[0]], mids)  # (lowest bin stays same)
    m_high = np.append(mids, [mass[-1]])  # (highest bin stays same)
    imf = (np.power(m_low, -1.35) - np.power(m_high, -1.35)) / 1.35
    imf[mass < lower] = 0.
    min_mass = max(lower, mass[0])
    max_mass = upper
    if norm_by_mass:
        imf *= .35 / (np.power(min_mass, -.35) - np.power(max_mass, -.35))
    else:
        imf *= 1.35 / (np.power(min_mass, -1.35) - np.power(max_mass, -1.35))
    return imf


def _interp_arrays(arr1, arr2, f):
    """Linearly interpolate between two (potentially unequal length) arrays
    
    Arguments:
    arr1 -- first (lower) array (len N1 or N1xD)
    arr2 -- second (upper) array (len N2 or N2xD, N2 doesn't have to equal N1)
    f -- linear interpolation fraction (float between 0 and 1)
    Output: interpolated array (len max(N1,N2) or max(N1,N2)xD)
    """
    utils.my_assert(arr1.ndim == arr2.ndim,
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


def _z_from_str(z_str):
    """Converts a metallicity value to MIST string
    Example Usage:
    _z_from_str("m0.53") -> -0.53
    _z_from_str("p1.326")   -> 1.326
    
    Arguments:
    z_str -- metallicity (as a string)
    Output: float value of metallicity
    """
    value = float(z_str[1:])
    if z_str[0] == 'm':
        value *= -1
    elif z_str[0] != 'p':
        raise ValueError('z string not of valid format')
    return value


def _z_to_str(z):
    """Converts a metallicity value to MIST string
    Example Usage:
    _z_to_str(-0.5313) -> "m0.53"
    _z_to_str(1.326)   -> "p1.33"
    
    Arguments:
    z -- metallicity (float)
    Output: string representing metallicity
    """
    result = ''
    if (z < 0):
        result += 'm'
    else:
        result += 'p'
    result += '%1.2f' % (np.abs(z))
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
                 dm_interp=-1):
        """Creates a new Isochrone_Model, given a list of Filter objects
        
        Arguments:
           filters -- list of Filter objects
        Keyword Arguments:
           MIST_path -- directory containing MIST model files
           z_arr -- array of MIST metallicity values to use
           dm_interp -- 
        """

        # Locate MIST files
        if MIST_path is None:
            MIST_path = resource_filename('pcmdpy', 'isoc_MIST_v1.2/')
        
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
        _z_arr = []
        self.filters = filters
        self.filter_names = [f.tex_name for f in self.filters]
        self.colnames = pd.read_table(MIST_path + 'columns.dat',
                                      delim_whitespace=True).columns
        # load all MIST files found in directory
        for MIST_doc in glob.glob(os.path.join(MIST_path, '*'+iso_append)):
            try:
                z_str = MIST_doc.split('feh_')[-1][:5]
                z = _z_from_str(z_str)
                new_df = pd.read_table(MIST_doc, names=self.colnames,
                                       comment='#', delim_whitespace=True,
                                       dtype=float, na_values=['Infinity'])
                new_df[new_df.isna()] = 100.
                new_df['z'] = z
                self.MIST_df = self.MIST_df.append([new_df], ignore_index=True)
                _z_arr.append(_z_from_str(z_str))
            except Exception:
                warn('File not properly formatted: %s' % (MIST_doc))
                sys.exit(1)
            
        self._z_arr = np.sort(_z_arr)
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
        self._interp_cols = ['initial_mass']
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
        return None
    
    def get_isochrone(self, age, z, imf_func=salpeter_IMF, rare_cut=0.,
                      downsample=1, system="vega", **kwargs):
        """Interpolate MIST isochrones for given age and metallicity
        
        Arguments:
           age ---
           z ---
           imf_func ---
           rare_cut ---
           downsample --- 
           system ---
        Output:
           imf ---
           mags -- 2D array of magnitudes (DxN, where D is number of filters
                   the model was initialized with)
        """

        system = system.lower()
        if system not in self.conversions.keys():
            warn(('system {0:s} not in list of magnitude '
                  'conversions. Reverting to Vega'.format(system)))
            conversions = self.conversions['vega']
        else:
            conversions = self.conversions[system]
        
        # Find closest age in MIST database
        if age not in self.ages:
            age = self.ages[np.abs(self.ages - age).argmin()]
        this_age = self.MIST_df[np.isclose(self.MIST_df.age.values, age)]
        # Output MIST values for known metallicities
        if z in self._z_arr:
            inter = this_age[np.isclose(this_age.z.values, z)][self._interp_cols].values
        # Interpolate/extrapolate for other metallicities
        else:
            i = self._z_arr.searchsorted(z)
            if (i == 0):
                i = 1  # will extrapolate low
            elif (i == len(self._z_arr)):
                i = -1  # will extrapolate high
            zlow, zhigh = self._z_arr[i-1:i+1]  # bounding metallicities
            frac_between = (z - zlow) / (zhigh - zlow)
            if (frac_between >= 2) or (frac_between <= -1):
                raise ValueError('Extrapolating metallicity more than one '
                                 'entire metallicity bin')
            dflow = this_age[np.isclose(this_age.z.values, zlow)][self._interp_cols]
            dfhigh = this_age[np.isclose(this_age.z.values, zhigh)][self._interp_cols]
            inter = _interp_arrays(dflow.values, dfhigh.values, frac_between)
            
        IMF = imf_func(inter[::downsample, 0], **kwargs)

        mags = (inter[::downsample, 1:] + conversions).T
        # lum = np.power(10., -0.4*mags)
        # mean_lum = np.average(lum, weights=IMF, axis=1)
        
        # remove stars that are extremely rare
        to_keep = (IMF >= rare_cut)

        return IMF[to_keep], mags[:, to_keep]

    def model_galaxy(self, galaxy, lum_cut=np.inf, system='vega',
                     downsample=1,
                     **kwargs):
        weights = np.empty((1, 0), dtype=float)
        mags = np.empty((self.num_filters, 0), dtype=float)
        # Collect the isochrones from each bin
        for age, feh, sfh, d_mod in galaxy.iter_SSPs():
            imf, m = self.get_isochrone(age, feh, system=system,
                                        downsample=downsample, **kwargs)
            weights = np.append(weights, imf*sfh)
            m += d_mod
            mags = np.append(mags, m, axis=-1)
        lum = np.power(10., -0.4*mags)
        mean_lum = np.average(lum, weights=weights, axis=1)
        to_keep = (lum.T / mean_lum >= lum_cut).sum(axis=1) == 0
        return weights[to_keep], mags[:, to_keep]

