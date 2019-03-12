# utils.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from scipy.misc import logsumexp
from astropy.io import fits
import os, sys


# A module to create various utility functions
def make_pcmd(mags):
    pcmd = np.copy(mags)
    n_filters = pcmd.shape[0]
    for i in range(1, n_filters):
        pcmd[i] = mags[i] - mags[i-1]
    return pcmd


def pcmd_to_mags(pcmd):
    mags = np.copy(pcmd)
    n_filters = mags.shape[0]
    for i in range(1, n_filters):
        mags[i] = pcmd[i] + mags[i-1]
    return mags


def mean_mags(pcmd):
    mags = pcmd_to_mags(pcmd)
    mag_factor = -0.4 * np.log(10)  # convert from base 10 to base e
    weights = float(1) / mags.shape[1]  # evenly weight each pixel
    return logsumexp(mag_factor*mags, b=weights, axis=1)


def make_hess(pcmd, bins, err_min=2.):
    mags = pcmd[0]
    colors = pcmd[1:]
    n_colors = colors.shape[0]
    n = pcmd.shape[1]  # total number of pixels
    counts = []
    for i in range(n_colors):
        c, _, _ = np.histogram2d(mags, colors[i],
                                 bins=[bins[0], bins[i+1]])
        if np.sum(c) < n:  # if some pixels fell outside bins, add to corner of Hess grid
            c[0, 0] += (n - np.sum(c))
        counts += [c]
    if n_colors == 0:
        c, _ = np.histogram(mags, bins=bins[0])
        counts += [c]
    counts = np.array(counts)
    counts[counts <= 0.] = 0.
    err = np.sqrt(counts)
    
    # inflate small errors
    err[err <= err_min] = err_min
    # err += err_min * np.exp(-err)
    
    # normalize by number of pixels
    hess = counts / n
    err /= n
    
    return counts, hess, err


class DataSet(object):
    
    def __init__(self, file_names, filter_classes):
        assert(len(file_names) == len(filter_classes))
        self.n_bands = len(filter_classes)
        headers = []
        with fits.open(file_names[0]) as hdu:
            if len(hdu) > 1:
                data = hdu['SCI'].data
            else:
                data = hdu['PRIMARY'].data
            self.im_shape = data.shape
            self.images = np.zeros((self.im_shape[0], self.im_shape[1],
                                    self.n_bands))
            headers.append(hdu[0].header)
            self.images[:, :, 0] = data
        for i, f in enumerate(file_names[1:]):
            with fits.open(f) as hdu:
                if len(hdu) > 1:
                    data = hdu['SCI'].data
                else:
                    data = hdu['PRIMARY'].data
                self.images[:, :, i+1] = data
                headers.append(hdu[0].header)
        self.headers = np.array(headers)
        assert(self.images.ndim == 3)  # else the images weren't matching sizes
        filters = []
        for filt, header in zip(filter_classes, headers):
            filters.append(filt(exposure=header['EXPTIME']))
        self.filters = np.array(filters)
        
    def get_pcmd(self, bool_matrix, bands=None):
        if bands is not None:
            assert(max(bands) < self.n_bands)
            assert(min(bands) <= 0)
            pixels = self.images[bool_matrix, bands]
        else:
            bands = np.arange(self.n_bands)
            pixels = self.images[bool_matrix, :]
        assert(bool_matrix.shape == self.im_shape)
        filts = self.filters[bands]
        mags = np.zeros_like(pixels.T)
        for i in bands:
            flux = pixels[:, i] * filts[i]._exposure  # convert to counts
            mags[i] = filts[i].counts_to_mag(flux)
        pcmd = make_pcmd(mags)
        return pcmd

    def get_image(self, bool_matrix=None, downsample=1, bands=None):
        if bands is None:
            bands = np.arange(self.n_bands)
        if bool_matrix is None:
            images = np.copy(self.images[::downsample, ::downsample, :])
            for i, b in enumerate(bands):
                images[:, :, i] *= self.filters[b]._exposure
            xmin, ymin = 0, 0
            xmax, ymax = self.im_shape
        else:
            assert(bool_matrix.shape == self.images.shape[:2])
            bool_matrix = bool_matrix[::downsample, ::downsample]
            x, y = np.where(bool_matrix)
            xmin, xmax = min(x), max(x)+1
            ymin, ymax = min(y), max(y)+1
            images = np.zeros((xmax-xmin, ymax-ymin, 3))
            for a in [xmin, xmax, ymin, ymax]:
                a *= downsample
            bools = bool_matrix[xmin:xmax, ymin:ymax]
            for i, b in enumerate(bands):
                images[bools, i] = self.images[::downsample,
                                               ::downsample][bool_matrix, b]
                images[:, :, i] *= self.filters[b]._exposure
        return images, (ymin, ymax, xmin, xmax)

    
class PrintRedirect:
    """
    Returns a context within which all stdout is redirected
    """
    
    def __init__(self, logfile=None):
        self._original_stdout = sys.stdout
        if logfile is None:
            self.logfile = os.devnull
        else:
            self.logfile = logfile
            
    def __enter__(self):
        sys.stdout = open(self.logfile, 'a')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class RegularPrint:
    """
    Context within print behaves as usual
    """

    def __init__(self):
        pass
    
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass




