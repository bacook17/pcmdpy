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
    dm = np.diff(mass)
    m_low = mass - 0.5*np.append([0.], dm)  # (lowest bin stays same)
    m_high = mass + 0.5*np.append(dm, [0.])  # (highest bin stays same)
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
    def __init__(self, filters, MIST_path=None, iso_append=".iso.cmd"):
        """Creates a new Isochrone_Model, given a list of Filter objects
        
        Arguments:
           filters -- list of Filter objects
        Keyword Arguments:
           MIST_path -- directory containing MIST model files
           z_arr -- array of MIST metallicity values to use
        """

        # Locate MIST files
        if MIST_path is None:
            MIST_path = resource_filename('pcmdpy', 'isoc_MIST_v1.1/')
        
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
                                       dtype=float)
                new_df['z'] = z
                self.MIST_df = self.MIST_df.append([new_df], ignore_index=True)
                _z_arr.append(_z_from_str(z_str))
            except Exception:
                warn('File not properly formatted: %s' % (MIST_doc))
                sys.exit(1)
            
        self._z_arr = np.sort(_z_arr)
        self.MIST_df.rename(columns={'log10_isochrone_age_yr': 'age'},
                            inplace=True)
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
        this_age = self.MIST_df[self.MIST_df.age == age]
        # Output MIST values for known metallicities
        if z in self._z_arr:
            inter = this_age[this_age.z == z][self._interp_cols].values
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
            dflow = this_age[this_age.z == zlow][self._interp_cols]
            dfhigh = this_age[this_age.z == zhigh][self._interp_cols]
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
        for age, feh, sfh in galaxy.iter_SSPs():
            imf, m = self.get_isochrone(age, feh, system=system,
                                        downsample=downsample, **kwargs)
            weights = np.append(weights, imf*sfh)
            mags = np.append(mags, m, axis=-1)
        lum = np.power(10., -0.4*mags)
        mean_lum = np.average(lum, weights=weights, axis=1)
        to_keep = (lum.T / mean_lum >= lum_cut).sum(axis=1) == 0
        return weights[to_keep], mags[:, to_keep]

    def plot_isochrone(self, galaxy, axes=None, system='vega', downsample=1,
                       **kwargs):
        if axes is None:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(ncols=(self.num_filters-1), sharey=True)
        names = self.filter_names
        for age, feh, _ in galaxy.iter_SSPs():
            _, mags = self.get_isochrone(age, feh, system=system,
                                         downsample=downsample)
            if self.num_filters == 2:
                axes.plot(mags[1]-mags[0], mags[0], 'k-',
                          label='age:{0:.1f}, feh:{1:.1f}'.format(age, feh),
                          **kwargs)
                axes.set_xlabel('{0:s} - {1:s}'.format(names[1], names[0]),
                                fontsize='x-large')
                axes.set_ylabel(names[0], fontsize='x-large')
                yl = axes.get_ylim()
                axes.set_ylim([max(yl), min(yl)])
            else:
                for i, ax in enumerate(axes):
                    ax.plot(mags[i+1]-mags[0], mags[0], 'k-',
                            label='age:{0:.1f}, feh:{1:.1f}'.format(age, feh),
                            **kwargs)
                    ax.set_xlabel('{0:s} - {1:s}'.format(names[i+1], names[0]),
                                  fontsize='x-large')
                    ax.set_ylabel(names[0], fontsize='x-large')
                    yl = ax.get_ylim()
                    ax.set_ylim([max(yl), min(yl)])
        return axes
