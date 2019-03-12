# filter.py
# Ben Cook (bcook@cfa.harvard.edu)

__all__ = ['Filter', 'ACS_WFC_F435W', 'ACS_WFC_F475W', 'ACS_WFC_F555W',
           'ACS_WFC_F814W', 'ACS_WFC_F850LP',
           'm31_filter_sets', 'm49_filter_sets', 'm51_filter_sets',
           'default_m31_filters', 'default_m49_filters', 'default_m51_filters',
           'default_NGC3377_filters', 'AVAILABLE_FILTERS',
           'm31_summer_filters', 'm31_winter_filters']

"""Define classes for Filters and other similar objects"""

from astropy.io import fits
from .psf import PSF_Model
from ..simulation.gpu_utils import gpu_log10 as log10
from pkg_resources import resource_filename
from warnings import warn


class Filter:
    """Models observations in a single band
    
    A Filter specifies the conversions between magnitudes and counts, and PSF convolution
    
    Attributes:
       name -- descriptive name of the filter (string, eg: F475W)
       tex_name -- LaTeX formatted name of the filter, eg for use in plotting (string, eg: r"g$_{475}$")
       MIST_column -- column name in MIST tables (string, eg: vmag, fuv, or f275)
       props -- dictionary of other properties
    Methods:
       mag_to_counts -- convert absolute magnitudes to photon counts
       counts_to_mag -- convert photon counts to absolute magnitudes
       psf_convolve -- convolve (2D) array with the instrumental PSF
    Constructors:
       __init__ -- default, manual entry of all parameters
    """

    def __init__(self, exposure, zpt_vega, zpt_ab, zpt_st, red_per_ebv,
                 psf, name="", tex_name="", MIST_column="", MIST_column_alt="",
                 **kwargs):
        """Create a new Filter, given input properties of observation

        Arguments:
           exposure -- exposure time of the observation, in seconds (int or float)
           zero_point -- apparent (VEGA) magnitude corresponding to 1 count / second (int or float)
                   this value is affected by telescope aperture, sensitivity, etc.
           red_per_ebv -- the Reddening value [A_x / E(B-V)], such as from Schlafly & Finkbeiner 2011, Table 6 (float)
           psf -- the PSF kernel, should be normalized to one (2D square array of floats)
        Keyword Argments:
           name -- descriptive name of the filter (string)
           tex_name -- LaTeX formatted name of the filter, eg for use in plotting (string, eg: r"g$_{475}$")
           MIST_column -- column name in MIST tables (string)
           **kwargs -- all other keyword arguments will be saved as a dictionary
        """

        #validate and initialize internal attributes
        try:
            self._exposure = float(exposure)
            self._zpts = {}
            self._zpts['vega'] = float(zpt_vega)
            self._zpts['ab'] = float(zpt_ab)
            self._zpts['st'] = float(zpt_st)
            self.red_per_ebv = float(red_per_ebv)
        except TypeError:
            print('First six arguments must each be either a float or integer')
            raise

        # initialize public attributes
        if isinstance(psf, PSF_Model):
            self.psf_model = psf
        else:
            self.psf_model = PSF_Model(psf,
                                       dither_by_default=kwargs.get('dither_by_default',
                                                                    False))
        self.psf_convolve = self.psf_model.convolve

        self.name = name
        self.tex_name = tex_name
        self.MIST_column = MIST_column
        self.MIST_column_alt = MIST_column_alt
        self.props = kwargs

    #########################
    # Filter methods
    
    def mag_to_counts(self, mags, mag_system='vega'):
        """Convert apparent magnitudes to photon counts (no reddening assumed)

        Arguments:
           mags -- apparent magnitudes (int or float or array or ndarray)
        Output:
           counts -- photon counts (same type as input)
        """
        if mag_system in self._zpts:
            zpt = self._zpts[mag_system]
        else:
            warn(('mag_system {0:s} not in list of magnitude '
                  'conversions. Reverting to Vega'.format(mag_system)))
            zpt = self._zpts['vega']

        return 10.**(-0.4 * (mags - zpt)) * self._exposure

    def counts_to_mag(self, counts, mag_system='vega', **kwargs):
        """Convert photon counts to apparent magnitudes

        Arguments:
           counts -- photon counts (int or float or array or ndarray)
        Output:
           mags -- apparent magnitudes (same type as input)
        """
        if mag_system in self._zpts:
            zpt = self._zpts[mag_system]
        else:
            warn(('mag_system {0:s} not in list of magnitude '
                  'conversions. Reverting to Vega'.format(mag_system)))
            zpt = self._zpts['vega']

        return -2.5*log10(counts / self._exposure, **kwargs) + zpt


##############################
# Pre-defined Filters


class ACS_WFC_F435W(Filter):
    """Return a Filter with HST F435W default params
    Arguments:
    Output: Filter with default F435W attributes
    """
    def __init__(self, **kwargs):
        args = {}
        # set defaults
        args['exposure'] = 2720.
        args['zpt_vega'] = 25.7885  # see filter_setup.ipynb
        args['zpt_ab'] = 25.6903
        args['zpt_st'] = 25.1801
        args['red_per_ebv'] = 3.610
        args['psf'] = PSF_Model.from_fits('ACS_WFC_F435W')
        args['name'] = "F435W"
        args['tex_name'] = r"B$_{435}$"
        args['MIST_column'] = "ACS_WFC_F435W"
        args['MIST_column_alt'] = "Bmag"
        # update with manual entries
        args.update(kwargs)
        super().__init__(**args)

        
class ACS_WFC_F475W(Filter):
    """Return a Filter with HST F475W default params
    Arguments:
    Output: Filter with default F475W attributes
    """
    def __init__(self, **kwargs):
        args = {}
        # set defaults
        args['exposure'] = 3620.
        args['zpt_vega'] = 26.1511  # see filter_setup.ipynb
        args['zpt_ab'] = 26.0586
        args['zpt_st'] = 25.7483
        args['red_per_ebv'] = 3.248
        args['psf'] = PSF_Model.from_fits('ACS_WFC_F475W')
        args['name'] = "F475W"
        args['tex_name'] = r"g$_{475}$"
        args['MIST_column'] = "ACS_WFC_F475W"
        args['MIST_column_alt'] = "bmag"
        # update with manual entries
        args.update(kwargs)
        super().__init__(**args)
    

class ACS_WFC_F555W(Filter):
    """Return a Filter with HST F555W default params
    Arguments:
    Output: Filter with default F555W attributes
    """
    def __init__(self, **kwargs):
        args = {}
        # set defaults
        args['exposure'] = 1360.
        args['zpt_vega'] = 25.7318  # see filter_setup.ipynb
        args['zpt_ab'] = 25.7319
        args['zpt_st'] = 25.6857
        args['red_per_ebv'] = 2.792
        args['psf'] = PSF_Model.from_fits('ACS_WFC_F555W')
        args['name'] = "F555W"
        args['tex_name'] = r"V$_{555}$"
        args['MIST_column'] = "ACS_WFC_F555W"
        args['MIST_column_alt'] = "vmag"
        # update with manual entries
        args.update(kwargs)
        super().__init__(**args)


class ACS_WFC_F814W(Filter):
    """Return a Filter with HST F814W default params
    Arguments:
    Output: Filter with default F814W attributes
    """
    def __init__(self, **kwargs):
        args = {}
        # set defaults
        args['exposure'] = 1360.
        args['zpt_vega'] = 25.5283  # see filter_setup.ipynb
        args['zpt_ab'] = 25.9565
        args['zpt_st'] = 26.7927
        args['red_per_ebv'] = 1.536
        args['psf'] = PSF_Model.from_fits('ACS_WFC_F814W')
        args['name'] = "F814W"
        args['tex_name'] = r"I$_{814}$"
        args['MIST_column'] = "ACS_WFC_F814W"
        args['MIST_column_alt'] = "imag"
        # update with manual entries
        args.update(kwargs)
        super().__init__(**args)


class ACS_WFC_F850LP(Filter):
    """Return a Filter with HST F850LP default params
    Arguments:
    Output: Filter with default F850LP attributes
    """

    def __init__(self, **kwargs):
        args = {}
        # set defaults
        args['exposure'] = 560.0
        args['zpt_vega'] = 24.3530
        args['zpt_ab'] = 24.8788
        args['zpt_st'] = 25.9668
        args['red_per_ebv'] = 1.243
        args['psf'] = PSF_Model.from_fits('ACS_WFC_F850LP')
        args['name'] = 'F850LP'
        args['tex_name'] = r'z$_{850}$'
        args['MIST_column'] = 'ACS_WFC_F850LP'
        args['MIST_column_alt'] = 'zmag'
        # update with manual entries
        args.update(kwargs)
        super().__init__(**args)

        
AVAILABLE_FILTERS = {
    'F435W': ACS_WFC_F435W,
    'F475W': ACS_WFC_F475W,
    'F555W': ACS_WFC_F555W,
    'F814W': ACS_WFC_F814W,
    'F850LP': ACS_WFC_F850LP,
}

        
m31_filter_sets = [ACS_WFC_F814W, ACS_WFC_F475W]
m51_filter_sets = [ACS_WFC_F814W, ACS_WFC_F555W, ACS_WFC_F435W]
m49_filter_sets = [ACS_WFC_F850LP, ACS_WFC_F475W]


def default_m31_filters(exp_F814W=3235.0,
                        exp_F475W=3620.0):
    filts = [ACS_WFC_F814W(exposure=exp_F814W),
             ACS_WFC_F475W(exposure=exp_F475W)]
    return filts


def m31_summer_filters(exp_F814W=3040.0,
                       exp_F475W=3440.0):
    filts = [ACS_WFC_F814W(exposure=exp_F814W),
             ACS_WFC_F475W(exposure=exp_F475W)]
    return filts


def m31_winter_filters(exp_F814W=3430.0,
                       exp_F475W=3800.0):
    filts = [ACS_WFC_F814W(exposure=exp_F814W),
             ACS_WFC_F475W(exposure=exp_F475W)]
    return filts


def default_m51_filters():
    return [f() for f in m51_filter_sets]


def default_m49_filters():
    filts = [ACS_WFC_F850LP()]
    filts += [ACS_WFC_F475W(exposure=375., zpt_vega=26.1746,
                            zpt_ab=26.0820, zpt_st=25.7713)]
    return filts


def default_NGC3377_filters():
    red = ACS_WFC_F850LP(
        exposure=3005.0,
        zpt_vega=24.3512,
        zpt_ab=24.8770,
        zpt_st=25.9650)
    blue = ACS_WFC_F475W(
        exposure=1380.0,
        zpt_vega=26.1702,
        zpt_ab=26.0776,
        zpt_st=25.7668)
    return [red, blue]


def m31_narrow_psf(alpha_F814W=1.147, alpha_F475W=1.109):
    filts = []
    psf_f814w = PSF_Model.from_fits('ACS_WFC_F814W',
                                    narrow_alpha=alpha_F814W)
    psf_f475w = PSF_Model.from_fits('ACS_WFC_F475W',
                                    narrow_alpha=alpha_F475W)
    filts.append(ACS_WFC_F814W(exposure=3235., psf=psf_f814w))
    filts.append(ACS_WFC_F475W(exposure=3620., psf=psf_f475w))
    return filts


