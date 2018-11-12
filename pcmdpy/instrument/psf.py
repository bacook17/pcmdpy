# psf.py
# Ben Cook (bcook@cfa.harvard.edu)

__all__ = ['PSF_Model']

"""Define class to represent PSF models"""

import numpy as np
from scipy.signal import fftconvolve
from warnings import warn
from astropy.io import fits
from pkg_resources import resource_filename


class PSF_Model:
    """A model for instrumental PSF

    A PSF model specifies how flux is spread across detector pixels.

    Attributes
    ----------

    """

    def __init__(self, psf, dither_by_default=True):
        if not isinstance(psf, np.ndarray):
            psf = np.array(psf)
        if (psf.ndim != 2) or (psf.shape[-2] != psf.shape[-1]):
            raise TypeError(
                'The input psf must be a square, 2D array of floats')
        self._d_psf = psf.shape[-1]
        self.psf = np.copy(psf) / np.sum(psf)
        self.dithered_psf = _generate_dithered_images(psf, norm=True)
        self.n_dither = self.dithered_psf.shape[0]
        assert self.dithered_psf.shape[1] == self.n_dither, (
            "Should never reach here. Dithering should be symmetric")
        self.dither_by_default = dither_by_default

    def convolve(self, image, dither=None, convolve_func=None,
                 convolve_kwargs={}, **kwargs):
        """Convolve image with instrumental PSF
        
        Arguments:
           image -- counts, or flux, in each pixel of image (2D array of integers or floats)
        Keyword Arguments:
           dither --  
           convolve_func -- function to convolve the image and PSF (default: scipy.signal.fftconvolve)
           convolve_kwargs -- any additional keyword arguments will be passed to convolve_func
        Output:
           convolved_image -- image convolved with PSF (2D array of floats;
                                       guaranteed same shape as input if default convolve_func used)
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if (image.ndim != 2):
            raise TypeError('The input image must be a 2D array of integers '
                            'or floats')
        if dither is None:
            dither = self.dither_by_default
        if image.shape[-1] != image.shape[-2]:
            dither = False  # unable to subdivide image properly if not square
        if convolve_func is None:
            convolve_func = fftconvolve
            convolve_kwargs['mode'] = 'valid'
        if dither:
            # add border and subdivide
            sub_im_matrix = _subdivide_image(image, self.n_dither,
                                             w_border=self._d_psf-1)
            convolved_matrix = np.array([[
                convolve_func(sub_im_matrix[i, j], self.dithered_psf[i, j],
                              **convolve_kwargs) for j in range(self.n_dither)]
                                         for i in range(self.n_dither)])
            im_final = np.concatenate(np.concatenate(convolved_matrix,
                                                     axis=-2), axis=-1)
        else:
            im_final = convolve_func(image, self.psf, **convolve_kwargs)
        if (im_final.shape != image.shape):
            warn(
                "Image shape has changed: {} to {}".format(image.shape,
                                                           im_final.shape),
                RuntimeWarning)
        return im_final

    @classmethod
    def from_fits(cls, filter_name, dither_by_default=True,
                  narrow_alpha=None):
        psf_file = resource_filename('pcmdpy', 'instrument/PSFs/') + filter_name + '.fits'
        psf = fits.open(psf_file)[0].data.astype(float)
        if narrow_alpha is not None:
            assert isinstance(narrow_alpha, float)
            psf = psf**narrow_alpha
        return cls(psf, dither_by_default=dither_by_default)


def _subpixel_shift(image, dx, dy):
    if not (-1 <= dx <= 1):
        raise NotImplementedError(
            "_subpixel_shift only applies for shifts less than one pixel. "
            "Given {}".format(dx))
    if not (-1 <= dy <= 1):
        raise NotImplementedError(
            "_subpixel_shift only applies for shifts less than one pixel. "
            "Given {}".format(dx))
    # roll the image by -1, 0, +1 in x and y
    rolls = np.zeros((3, 3, image.shape[0], image.shape[1]))
    for i, x in enumerate([-1, 0, 1]):
        for j, y in enumerate([-1, 0, 1]):
            rolls[j, i] = np.roll(np.roll(image, x, axis=1), y, axis=0)
    # make the coefficients for each corresponding rolled image
    coeffs = np.ones((3, 3))
    if np.isclose(dx, 0.):
        coeffs[:, 0] = coeffs[:, 2] = 0.
    elif dx < 0.:
        coeffs[:, 2] = 0.
        coeffs[:, 0] *= -dx
        coeffs[:, 1] *= 1 + dx
    else:
        coeffs[:, 0] = 0.
        coeffs[:, 2] = dx
        coeffs[:, 1] *= 1 - dx
        
    if np.isclose(dy, 0.):
        coeffs[0, :] = coeffs[2, :] = 0.
    elif dy < 0.:
        coeffs[2, :] = 0.
        coeffs[0, :] *= -dy
        coeffs[1, :] *= 1 + dy
    else:
        coeffs[0, :] = 0.
        coeffs[2, :] *= dy
        coeffs[1, :] *= 1 - dy
    assert(np.isclose(np.sum(coeffs), 1.))
    result = np.zeros((image.shape[0], image.shape[0]))
    for i in range(3):
        for j in range(3):
            result += coeffs[i, j] * rolls[i, j]
    return result


def _generate_dithered_images(image, shifts=[0., 0.25, 0.5, 0.75], norm=False):
    n = len(shifts)
    X = image.shape[0]
    Y = image.shape[1]
    tiles = np.zeros((n, n, X, Y))
    for i, dx in enumerate(shifts):
        for j, dy in enumerate(shifts):
            tiles[j, i] = _subpixel_shift(image, dx, dy)
            if norm:
                tiles[j, i] /= np.sum(tiles[j, i])
    return tiles


def _wrap_image(image, w_border):
    assert (image.ndim == 2), "images must be 2-dimensional"
    Nx, Ny = image.shape
    if (w_border >= Nx) or (w_border >= Ny):
        message = ("wrap_image is not implemented for cases where border is "
                   "wider than existing image")
        raise NotImplementedError(message)
    w_roll = w_border // 2
    im_temp = np.tile(image, [2, 2])
    im_temp = np.roll(np.roll(im_temp, w_roll, axis=0), w_roll, axis=1)

    return im_temp[:Nx+w_border, :Ny+w_border]


def _subdivide_image(image, d_sub, w_border=0):
    assert (image.ndim == 2), ("image must be 2-dimensional")
    Nx, Ny = image.shape
    if (Nx != Ny):
        message = "image must be square"
        raise NotImplementedError(message)
    if (Nx % d_sub != 0):
        message = ("subdivide_image is only implemented if image can be "
                   "cleanly subdivided")
        raise NotImplementedError(message)
    Nx_sub, Ny_sub = Nx // d_sub, Ny // d_sub

    if w_border > 0:
        image = _wrap_image(image, w_border)

    sub_im_matrix = np.zeros((d_sub, d_sub,
                              Nx_sub + w_border, Ny_sub + w_border))
    for i in range(d_sub):
        for j in range(d_sub):
            x_slice = slice(Nx_sub*i, Nx_sub*(i+1) + w_border)
            y_slice = slice(Ny_sub*j, Ny_sub*(j+1) + w_border)
            sub_im_matrix[i, j] = image[x_slice, y_slice]
    return sub_im_matrix

