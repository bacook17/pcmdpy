# utils.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
import matplotlib
matplotlib.use('pdf')

import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.ndimage import gaussian_filter as norm_kde
import cv2
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
    mag_factor = 0.4 * np.log(10)  # convert from base 10 to base e
    weights = float(1) / mags.shape[1]  # evenly weight each pixel
    return logsumexp(mag_factor*mags, b=weights, axis=1)/mag_factor


def mean_mags_old(pcmd):
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


def get_contours(pcmd, levels, smooth=0.01, span=None):
    """
    Copied mostly verbatim from Dynesty
    """
    y, x = pcmd
    data = [x, y]
    if span is None:
        span = [0.999999426697 for i in range(2)]
    span = list(span)
    if len(span) != 2:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = np.array([0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]])
            span[i] = np.percentile(data[i], list(100.*q))
    if (isinstance(smooth, int) or isinstance(smooth, float)):
        smooth = [smooth, smooth]
    bins, svalues = [], []
    for s in smooth:
        if isinstance(s, int):
            bins.append(s)
            svalues.append(0.)
        else:
            bins.append(int(round(2./s)))
            svalues.append(2.)
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                             range=list(map(np.sort, span)))
    if not np.all(svalues == 0.):
        H = norm_kde(H, svalues)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = (np.diff(V) == 0)
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = (np.diff(V) == 0)
    V.sort()
    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    
    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])
    
    fig, ax = plt.subplots()
    contours = ax.contour(X2, Y2, H2.T, V)
    plt.close(fig)
    
    return dict(zip(levels[::-1], contours.allsegs))
        

def in_contour(pcmd, contour):
    ROUND_FACTOR = 1e4
    y, x = pcmd
    assert len(x) == len(y)
    if len(contour) == 0:
        return np.array([False]*len(x))
    d = len(contour)
    if d == 1:
        contour = contour[0]
        n = len(contour)
        cv2_contour = (contour.reshape((n, 1, 2))*ROUND_FACTOR).astype(int)
        return np.array([cv2.pointPolygonTest(cv2_contour, (a, b), False) for
                         a, b in zip(x*ROUND_FACTOR, y*ROUND_FACTOR)]) > 0
    else:
        n_points = len(x)
        not_in = np.ones(n_points, dtype=bool)
        for sub_contour in contour:
            n = len(sub_contour)
            cv2_contour = (sub_contour.reshape((n, 1, 2))*ROUND_FACTOR).astype(int)
            not_in[not_in] = (np.array([cv2.pointPolygonTest(cv2_contour, (a, b), False) for
                                        a, b in zip(x[not_in]*ROUND_FACTOR, y[not_in]*ROUND_FACTOR)]) < 0)
        return np.logical_not(not_in)


def contour_fracs(pcmd, contours):
    ys = {}
    xs = np.array(sorted(contours.keys())[::-1])
    is_good = np.array([True]*pcmd.shape[1])
    # Check if a point is in each contour, decreasing in size
    for k in xs:
        cont = contours[k]
        # Ignore all points that weren't in larger contours
        is_good[is_good] = in_contour(pcmd[:, is_good], cont)
        ys[k] = is_good.mean()
    return xs, np.array([ys[k] for k in xs])
        

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

