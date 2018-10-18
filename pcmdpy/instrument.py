# instrument.py
# Ben Cook (bcook@cfa.harvard.edu)

__all__ = ['Filter', 'ACS_WFC_F435W', 'ACS_WFC_F475W', 'ACS_WFC_F555W',
           'ACS_WFC_F814W', 'm31_filters', 'm51_filters',
           'default_m31_filters', 'default_m51_filters']

"""Define classes for Filters and other similar objects"""

import numpy as np
from astropy.io import fits
from pcmdpy import utils
from pcmdpy.gpu_utils import gpu_log10 as log10
from scipy.signal import fftconvolve, gaussian
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
       HST_F475W -- the Hubble F475W filter (only free parameter is distance)
       HST_F814W -- the Hubble F814W filter (only free parameter is distance)
    """

    def __init__(self, exposure, zpt_vega, zpt_ab, zpt_st, red_per_ebv,
                 psf, name="", tex_name="", MIST_column="", MIST_column_alt="",
                 tiled_psf=True, **kwargs):
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
        if not isinstance(psf, np.ndarray):
            psf = np.array(psf)
        if (psf.shape[-2] != psf.shape[-1]) or (psf.dtype != float):
            raise TypeError('The seventh argument (psf) must be a square array (or 2D-array of square arrays) of floats')
        else:
            utils.my_assert((psf.ndim == 2) or (psf.ndim == 4),
                            fail_message='The seventh argument (psf) must be 2 or 4-dimensional (square array, or 2D-array of square arrays)')
            if (psf.ndim == 2):
                # create 4x4 grid of PSFs
                if tiled_psf:
                    psf = utils.generate_image_dithers(psf, norm=True)
                else:
                    psf /= np.sum(psf)
            else:
                psf = np.array([[psf[i, j] / np.sum(psf[i, j])
                                 for j in range(psf.shape[1])]
                                for i in range(psf.shape[0])])
            self._psf = psf
            

        #initialize public attributes
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

    def psf_convolve(self, image, multi_psf=True, convolve_func=None, **kwargs):
        """Convolve image with instrumental PSF
        
        Arguments:
           image -- counts, or flux, in each pixel of image (2D array of integers or floats)
        Keyword Arguments:
           multi_psf -- set to TRUE if 
           convolve_func -- function to convolve the image and PSF (default: scipy.signal.fftconvolve)
           **kwargs -- any additional keyword arguments will be passed to convolve_func
        Output:
           convolved_image -- image convolved with PSF (2D array of floats;
                                       guaranteed same shape as input if default convolve_func used)
        """

        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if (image.ndim != 2):
            raise TypeError('The first argument (image) must be a 2D array of integers or floats')    

        if self._psf.ndim != 4:
            multi_psf = False
        
        if convolve_func is None:
            N = image.shape[0]
            p = self._psf.shape[-1]
            if (p != self._psf.shape[-2]):
                message = 'each psf must be a square array'
                raise NotImplementedError(message)
            if multi_psf:
                utils.my_assert(self._psf.ndim == 4,
                                "psf must have 4 dimensions for multi_psf")
                d_sub = self._psf.shape[0]
                utils.my_assert(d_sub == self._psf.shape[1],
                                "psfs are not square")
                #add border and subdivide
                sub_im_matrix = utils.subdivide_image(image, d_sub, w_border=p-1)
                convolved_matrix = np.array([[fftconvolve(sub_im_matrix[i,j], self._psf[i,j], mode='valid') for j in range(d_sub)] for i in range(d_sub)])
                im_convolved = np.concatenate(np.concatenate(convolved_matrix, axis=-2), axis=-1)
            else:
                #add border
                im_new = utils.wrap_image(image, p-1)
                if self._psf.ndim == 2:
                    im_convolved = fftconvolve(im_new, self._psf, mode='valid')
                else:
                    im_convolved = fftconvolve(im_new, self._psf[0,0], mode='valid')
        elif (convolve_func=="gaussian"):
            if "width" in list(kwargs.keys()):
                width = kwargs['width']
            else:
                width = 3.
            kernel = np.outer(gaussian(30, width), gaussian(30, width))
            im_convolved = fftconvolve(image, kernel, mode='valid')
        else:
            im_convolved = convolve_func(image, self._psf, **kwargs)

        utils.my_assert(im_convolved.shape == image.shape,
                        "Image shape has changed: %s to %s" %
                        (str(image.shape), str(im_convolved.shape)))
        return im_convolved

    ##############################
    # Alternative constructors
    @classmethod
    def HST_F475W(cls, **kwargs):
        """
        Deprecated. Use: ACS_WFC_F475W
        """
        return ACS_WFC_F475W(**kwargs)

    @classmethod
    def HST_F814W(cls, **kwargs):
        """
        Deprecated. Use: ACS_WFC_F814W
        """
        return ACS_WFC_F814W(**kwargs)

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
        psf_path = resource_filename('pcmdpy', 'psf/')
        psf_file = psf_path + 'ACS_WFC_F435W.fits'
        args['psf'] = fits.open(psf_file)[0].data.astype(float)
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
        psf_path = resource_filename('pcmdpy', 'psf/')
        psf_file = psf_path + 'ACS_WFC_F475W.fits'
        args['psf'] = fits.open(psf_file)[0].data.astype(float)
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
        psf_path = resource_filename('pcmdpy', 'psf/')
        psf_file = psf_path + 'ACS_WFC_F555W.fits'
        args['psf'] = fits.open(psf_file)[0].data.astype(float)
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
        psf_path = resource_filename('pcmdpy', 'psf/')
        psf_file = psf_path + 'ACS_WFC_F814W.fits'
        args['psf'] = fits.open(psf_file)[0].data.astype(float)
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
        psf_path = resource_filename('pcmdpy', 'psf/')
        psf_file = psf_path + 'ACS_WFC_F850LP.fits'
        args['psf'] = fits.open(psf_file)[0].data.astype(float)
        args['name'] = 'F850LP'
        args['tex_name'] = r'z$_{850}$'
        args['MIST_column'] = 'ACS_WFC_F850LP'
        args['MIST_column_alt'] = 'zmag'
        # update with manual entries
        args.update(kwargs)
        super().__init__(**args)


m31_filter_sets = [ACS_WFC_F814W, ACS_WFC_F475W]
m51_filter_sets = [ACS_WFC_F814W, ACS_WFC_F555W, ACS_WFC_F435W]
m49_filter_sets = [ACS_WFC_F850LP, ACS_WFC_F475W]


def default_m31_filters():
    filts = [ACS_WFC_F814W(exposure=3235.),
             ACS_WFC_F475W(exposure=3620.)]
    return filts


def default_m51_filters():
    return [f() for f in m51_filter_sets]


def default_m49_filters():
    filts = [ACS_WFC_F850LP()]
    filts += [ACS_WFC_F475W(exposure=375., zpt_vega=26.1746,
                            zpt_ab=26.0820, zpt_st=25.7713)]
    return filts


def m31_narrow_psf(F814W=True, F475W=True, extra=False):
    psf_path = resource_filename('pcmdpy', 'psf/')
    if extra:
        psf1 = fits.open(psf_path + 'F814W_25p_narrow.fits')[1].data.astype(float)
        psf2 = fits.open(psf_path + 'F475W_25p_narrow.fits')[1].data.astype(float)
    else:
        psf1 = fits.open(psf_path + 'F814W_10p_narrow.fits')[1].data.astype(float)
        psf2 = fits.open(psf_path + 'F475W_10p_narrow.fits')[1].data.astype(float)
    filts = []
    if F814W:
        filts.append(ACS_WFC_F814W(exposure=3235., psf=psf1))
    else:
        filts.append(ACS_WFC_F814W(exposure=3235.))
    if F475W:
        filts.append(ACS_WFC_F475W(exposure=3620., psf=psf2))
    else:
        filts.append(ACS_WFC_F475W(exposure=3620.))
    return filts
