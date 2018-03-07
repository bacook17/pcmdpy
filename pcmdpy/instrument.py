# instrument.py
# Ben Cook (bcook@cfa.harvard.edu)

"""Define classes for Filters and other similar objects"""

import numpy as np
from astropy.io import fits
from pcmdpy import utils
from pcmdpy.gpu_utils import gpu_log10
from scipy.signal import fftconvolve, gaussian
from pkg_resources import resource_filename


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

    
    def __init__(self, exposure, zero_point, d_mpc, red_per_ebv, 
                 vega_to_ab, vega_to_st, psf,
                 name="", tex_name="", MIST_column="", MIST_column_alt="",
                 tiled_psf=False, **kwargs):
        """Create a new Filter, given input properties of observation

        Arguments:
           exposure -- exposure time of the observation, in seconds (int or float)
           zero_point -- apparent (VEGA) magnitude corresponding to 1 count / second (int or float)
                   this value is affected by telescope aperture, sensitivity, etc.
           d_mpc -- the assumed distance to the source in Mpc (int or float)
           red_per_ebv -- the Reddening value [A_x / E(B-V)], such as from Schlafly & Finkbeiner 2011, Table 6 (float)
           psf -- the PSF kernel, should be normalized to one (2D square array of floats)
           vega_to_ab --
           vega_to_st --
        Keyword Argments:
           name -- descriptive name of the filter (string)
           tex_name -- LaTeX formatted name of the filter, eg for use in plotting (string, eg: r"g$_{475}$")
           MIST_column -- column name in MIST tables (string)
           **kwargs -- all other keyword arguments will be saved as a dictionary
        """

        #validate and initialize internal attributes
        try:
            self._exposure = float(exposure)
            self._zero_point = float(zero_point)
            self._dmod = 25. + 5.*np.log10(d_mpc)  # distance modulus
            self.red_per_ebv = float(red_per_ebv)
            self._vega_to_ab = float(vega_to_ab)
            self._vega_to_st = float(vega_to_st)
        except TypeError:
            print('First six arguments must each be either a float or integer')
            raise
        if np.isnan(self._dmod):
            raise ValueError('The third argument (d_mpc) must be greater than zero')
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
    
    def mag_to_counts(self, mags):
        """Convert absolute magnitudes to photon counts (no reddening assumed)

        Arguments:
           mags -- absolute magnitudes (int or float or array or ndarray)
        Output:
           counts -- photon counts (same type as input)
        """

        return 10.**(-0.4 * (mags + self._dmod - self._zero_point)) * self._exposure

    def counts_to_mag(self, counts, **kwargs):
        """Convert photon counts to absolute magnitudes (assuming reddening)

        Arguments:
           counts -- photon counts (int or float or array or ndarray)
        Output:
           mags -- absolute magnitudes (same type as input)
        """

        return -2.5*gpu_log10(counts / self._exposure, **kwargs) + self._zero_point - self._dmod

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
    def HST_F475W(cls, d_mpc, **kwargs):
        """
        Deprecated. Use: ACS_WFC_F475W
        """
        return ACS_WFC_F475W(d_mpc, **kwargs)

    @classmethod
    def HST_F814W(cls, d_mpc, **kwargs):
        """
        Deprecated. Use: ACS_WFC_F814W
        """
        return ACS_WFC_F814W(d_mpc, **kwargs)

##############################
# Pre-defined Filters
class ACS_WFC_F435W(Filter):
    """Return a Filter with HST F435W default params
    Arguments:
       d_mpc -- the assumed distance to the source
       exposure -- exposure time (in sec) DEFAULT=3235.0
    Output: Filter with default F435W attributes
    """
    def __init__(self, d_mpc, exposure=3235.):
        utils.my_assert(isinstance(d_mpc, int) or isinstance(d_mpc, float),
                        "d_mpc must be real number")
        if (d_mpc < 0.):
            raise ValueError('Argument (d_mpc) must be greater than zero')
        utils.my_assert(isinstance(exposure, int) or
                        isinstance(exposure, float),
                        "exposure must be real number")
        if (exposure < 0.):
            raise ValueError('Argument (exposure) must be greater than zero')

        args = {}
        args['exposure'] = exposure
        args['zero_point'] = 25.571   # VEGAmag. see filter_setup.ipynb
        args['d_mpc'] = d_mpc
        args['red_per_ebv'] = 3.610
        args['vega_to_ab'] = 0.1017
        args['vega_to_st'] = 0.6127
        psf_path = resource_filename('pcmdpy', 'psf/')
        psf_file = psf_path + 'ACS_WFC_F435W.fits'
        args['psf'] = fits.open(psf_file)[0].data.astype(float)
        args['name'] = "F435W"
        args['tex_name'] = r"B$_{435}$"
        args['MIST_column'] = "ACS_WFC_F435W"
        args['MIST_column_alt'] = "Bmag"
        super().__init__(**args)

class ACS_WFC_F475W(Filter):
    """Return a Filter with HST F475W default params
    Arguments:
       d_mpc -- the assumed distance to the source
       exposure -- exposure time (in sec) DEFAULT=3620.0
    Output: Filter with default F475W attributes
    """
    def __init__(self, d_mpc, exposure=3620.):
        utils.my_assert(isinstance(d_mpc, int) or isinstance(d_mpc, float),
                        "d_mpc must be real number")
        if (d_mpc < 0.):
            raise ValueError('Argument (d_mpc) must be greater than zero')
        utils.my_assert(isinstance(exposure, int) or
                        isinstance(exposure, float),
                        "exposure must be real number")
        if (exposure < 0.):
            raise ValueError('Argument (exposure) must be greater than zero')

        args = {}
        args['exposure'] = exposure
        args['zero_point'] = 26.0593
        args['d_mpc'] = d_mpc
        args['red_per_ebv'] = 3.248
        args['vega_to_ab'] = 0.0979
        args['vega_to_st'] = 0.4086
        psf_path = resource_filename('pcmdpy', 'psf/')
        psf_file = psf_path + 'ACS_WFC_F475W.fits'
        args['psf'] = fits.open(psf_file)[0].data.astype(float)
        args['name'] = "F475W"
        args['tex_name'] = r"g$_{475}$"
        args['MIST_column'] = "ACS_WFC_F475W"
        args['MIST_column_alt'] = "bmag"
        super().__init__(**args)
    
class ACS_WFC_F555W(Filter):
    """Return a Filter with HST F555W default params
    Arguments:
       d_mpc -- the assumed distance to the source
       exposure -- exposure time (in sec) DEFAULT=3235.0
    Output: Filter with default F555W attributes
    """
    def __init__(self, d_mpc, exposure=3235.):
        utils.my_assert(isinstance(d_mpc, int) or isinstance(d_mpc, float),
                        "d_mpc must be real number")
        if (d_mpc < 0.):
            raise ValueError('Argument (d_mpc) must be greater than zero')
        utils.my_assert(isinstance(exposure, int) or
                        isinstance(exposure, float),
                        "exposure must be real number")
        if (exposure < 0.):
            raise ValueError('Argument (exposure) must be greater than zero')

        args = {}
        args['exposure'] = exposure
        args['zero_point'] = 25.712  # VEGAmag see filter_setup.ipynb
        args['d_mpc'] = d_mpc
        args['red_per_ebv'] = 2.792
        args['vega_to_ab'] = 0.0063
        args['vega_to_st'] = 0.0525
        psf_path = resource_filename('pcmdpy', 'psf/')
        psf_file = psf_path + 'ACS_WFC_F555W.fits'
        args['psf'] = fits.open(psf_file)[0].data.astype(float)
        args['name'] = "F555W"
        args['tex_name'] = r"V$_{555}$"
        args['MIST_column'] = "ACS_WFC_F555W"
        args['MIST_column_alt'] = "vmag"
        super().__init__(**args)

class ACS_WFC_F814W(Filter):
    """Return a Filter with HST F814W default params
    Arguments:
       d_mpc -- the assumed distance to the source
       exposure -- exposure time (in sec) DEFAULT=3235.0
    Output: Filter with default F814W attributes
    """
    def __init__(self, d_mpc, exposure=3235.):
        utils.my_assert(isinstance(d_mpc, int) or isinstance(d_mpc, float),
                        "d_mpc must be real number")
        if (d_mpc < 0.):
            raise ValueError('Argument (d_mpc) must be greater than zero')
        utils.my_assert(isinstance(exposure, int) or
                        isinstance(exposure, float),
                        "exposure must be real number")
        if (exposure < 0.):
            raise ValueError('Argument (exposure) must be greater than zero')

        args = {}
        args['exposure'] = exposure
        args['zero_point'] = 26.36
        args['d_mpc'] = d_mpc
        args['red_per_ebv'] = 1.536
        args['vega_to_ab'] = -0.4237
        args['vega_to_st'] = -1.2632
        psf_path = resource_filename('pcmdpy', 'psf/')
        psf_file = psf_path + 'ACS_WFC_F814W.fits'
        args['psf'] = fits.open(psf_file)[0].data.astype(float)
        args['name'] = "F814W"
        args['tex_name'] = r"I$_{814}$"
        args['MIST_column'] = "ACS_WFC_F814W"
        args['MIST_column_alt'] = "imag"
        super().__init__(**args)


def m31_filters(dist=0.78):
    return [ACS_WFC_F814W(dist), ACS_WFC_F475W(dist)]


def m51_filters(dist=8.58):
    return [ACS_WFC_F814W(dist), ACS_WFC_F555W(dist), ACS_WFC_F435W(dist)]
