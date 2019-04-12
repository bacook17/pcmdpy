# extraction.py
# Ben Cook (bcook@cfa.harvard.edu)
import numpy as np
import pyregion
from astropy.io import fits
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from .utils import filter_from_fits
from ..instrument.filter import Filter


def compute_regions(image_file, region_file, xc=None, yc=None,
                    do_quadrants=True, separate_regions=False):
    with fits.open(image_file) as hdulist:
        header = hdulist[0].header
        shape = hdulist['SCI'].shape
        Y, X = np.mgrid[:shape[0], :shape[1]]
        d = hdulist['SCI'].data
        good_pixels = (hdulist['FLAGS'].data == 0)
        max_val = np.max(d[~np.isnan(d) & good_pixels])
        if (xc is None) or (yc is None):
            [yc], [xc] = np.where(d == max_val)
    Q1 = (Y >= yc) & (X >= xc)
    Q2 = (Y >= yc) & (X < xc)
    Q3 = (Y < yc) & (X < xc)
    Q4 = (Y < yc) & (X >= xc)
    regions = pyregion.open(region_file).as_imagecoord(header=header)
    matrix = np.zeros(shape, dtype=np.float32)
    for i in tqdm(range(len(regions))):
        mask = regions[i:i+1].get_mask(shape=shape)
        if separate_regions:
            matrix[mask] = i+1
        else:
            matrix[mask] += 1.0
    if (do_quadrants) and (not separate_regions):
        matrix[Q2] += 0.25
        matrix[Q3] += 0.5
        matrix[Q4] += 0.75
        matrix *= 4.0
        matrix = np.round(matrix, decimals=0).astype(np.int32)
        matrix -= 3
    else:
        matrix = np.round(matrix, decimals=0).astype(np.int32)
    matrix[matrix <= 0] = 0
    matrix[~good_pixels] = -1
    return matrix


def add_regions(input_dict, region_file,
                base_filter=None, xc=None, yc=None, do_quadrants=True,
                separate_regions=False):
    all_filters = list(input_dict.keys())
    filt = base_filter or all_filters[0]
    regions_matrix = compute_regions(input_dict[filt], region_file,
                                     xc=xc, yc=yc, do_quadrants=do_quadrants,
                                     separate_regions=separate_regions)
    reg_hdu = fits.ImageHDU(data=regions_matrix)
    reg_hdu.header['EXTNAME'] = 'REGIONS'
    h = reg_hdu.header
    h.add_history('Regions extracted from DS9 Contours')
    h.add_history('Region file used: {:s}'.format(region_file))
    h.add_history('Base filter used: {:s}'.format(filt))
    for f in all_filters:
        with fits.open(input_dict[f], mode='update') as h:
            while 'REGIONS' in h:
                h.pop('REGIONS')
            h[0].header['REGIONS'] = 'COMPLETE'
            h.insert(2, reg_hdu)
    return regions_matrix


def save_pcmds(input_dict, red_filter, blue_filter,
               min_points=1,
               mag_system='vega', path='./', name_append='region_'):
    if path[-1] != '/':
        path += '/'
    if isinstance(red_filter, str):
        red = filter_from_fits(input_dict[red_filter])
    else:
        assert isinstance(red_filter, Filter)
        red = red_filter
    if isinstance(blue_filter, str):
        blue = filter_from_fits(input_dict[blue_filter])
    else:
        assert isinstance(blue_filter, Filter)
        blue = blue_filter
    with fits.open(input_dict[red.name]) as h:
        red_mags = red.counts_to_mag(h['SCI'].data,
                                     mag_system=mag_system)
        regions = h['REGIONS'].data
        flags = h['FLAGS'].data
    with fits.open(input_dict[blue.name]) as h:
        blue_mags = blue.counts_to_mag(h['SCI'].data,
                                       mag_system=mag_system)
    pcmds = {}
    for i in tqdm(range(1, regions.max()+1)):
        mask = (regions == i) & (flags == 0)
        mag = red_mags[mask]
        color = blue_mags[mask] - mag
        to_use = (~np.isnan(mag)) & (~np.isnan(color))
        if np.sum(to_use) >= min_points:
            pcmds[i] = np.array([mag[to_use], color[to_use]])
            header = '{:s} mags\n# Region {:d}\n'.format(mag_system, i)
            header += '{:s} {:s}-{:s}\n'.format(red.name, blue.name, red.name)
            filename = path + name_append + '_{:d}.pcmd'.format(i)
            np.savetxt(filename, pcmds[i].T, fmt='%.6f', delimiter=' ',
                       header=header)
    return pcmds


# Functions to check gradient across regions
def get_XY(image):
    ny, nx = image.shape
    Y, X = np.mgrid[:ny, :nx]
    return X, Y


def get_RTheta(image, xc=0, yc=0):
    X, Y = get_XY(image)
    X = X - xc
    Y = Y - yc
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    return R, Theta


def plot_image(image, mask, ax=None):
    """
    Plot an image, with only a particular mask of a region shown
    """
    im = np.zeros_like(image)
    data = image[mask]
    im[mask] = data
    im[~mask] = np.nan
    if ax is None:
        fig, ax = plt.subplots()
    plt.subplot(ax)
    plt.imshow(im, norm=mpl.colors.Normalize(vmin=np.percentile(data, 1),
                                             vmax=np.percentile(data, 99)),
               cmap='viridis_r')
    ys, xs = np.where(mask)
    ax.set_xlim([xs.min(), xs.max()])
    ax.set_ylim([ys.min(), ys.max()])
    plt.colorbar()
    return ax


def correct_angle(theta):
    """
    Convert an angle theta (or array of angles) to [-pi, pi)
    """
    if isinstance(theta, float):
        if theta < -np.pi:
            return theta + 2*np.pi
        elif theta >= np.pi:
            return theta - 2*np.pi
        else:
            return theta
    else:
        theta[theta < -np.pi] += 2*np.pi
        theta[theta >= np.pi] -= 2*np.pi
    return theta


def is_between(thetas, lower, upper):
    """
    Evaluate if angles thetas are between lower and upper (including circularity)
    """
    lower, upper = correct_angle(np.array([lower, upper]))
    if lower <= upper:
        return (lower <= thetas) & (thetas <= upper)
    else:  # if upper loops around
        return (upper <= thetas) & (thetas <= lower)


def getEllipseParamsManual(mask, xc, yc, dtheta=0.05):
    """
    Given an image mask corresponding to a roughly-elliptical region, compute the elliptical
    parameters corresponding to the inner edge.
    
    Parameters
    ==========
    mask : 
    xc, yc : center position of the ellipse
    dtheta : allowed angular region to search for semi-major axis
    
    Returns
    =======
    xc, yc : central positions of the ellipse
    a : semi-major axis (scaled to unit 1)
    b : semi-minor axis (scaled to unit 1)
    r : scale radius of the ellipse (in pixels)
    phi : position angle
    """
    R, Theta = get_RTheta(mask, xc=xc, yc=yc)
    # Compute the radius to nearest inner edge (semi-minor axis)
    R = R[mask]
    Theta = Theta[mask]
    b = R.min()
    Theta_min = Theta[R.argmin()]
    # Look at points at +/- 90 degrees from inner-most edge
    t1, t2 = correct_angle(np.array([-dtheta, dtheta]) + Theta_min - np.pi/2)
    t3, t4 = correct_angle(np.array([-dtheta, dtheta]) + Theta_min + np.pi/2)
    far_mask_1 = is_between(Theta, t1, t2)
    far_mask_2 = is_between(Theta, t3, t4)
    # Compute the radius to farthest inner edge (semi-major axis)
    a_1 = R[far_mask_1].min() if far_mask_1.sum() > 0 else None
    a_2 = R[far_mask_2].min() if far_mask_2.sum() > 0 else None
    if (a_1 is not None) and (a_2 is not None):
        a = 0.5 * (a_1 + a_2)
    elif (a_1 is not None):
        a = a_1
    elif (a_2 is not None):
        a = a_2
    else:
        a = b
    Theta_max = correct_angle(Theta_min + np.pi/2.)
    r = np.sqrt(a**2 + b**2)
    phi = Theta_max
    return xc, yc, a/r, b/r, r, phi


class EllipticalFit:

    def __init__(self, mask, xc, yc, dtheta=0.05):
        self.xc = xc
        self.yc = yc
        _, _, self.a, self.b, self.r, self.phi = getEllipseParamsManual(
            mask, xc=xc, yc=yc, dtheta=dtheta)
        self.mask = mask
        
    def plot_ellipse(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        R = np.arange(0, 2*np.pi, 0.01)
        t1, t2 = self.a*self.r, self.b*self.r
        xx = self.xc + t1*np.cos(R)*np.cos(self.phi) - t2*np.sin(R)*np.sin(self.phi)
        yy = self.yc + t1*np.cos(R)*np.sin(self.phi) + t2*np.sin(R)*np.cos(self.phi)
        ax.plot(xx, yy, 'r:')
        return ax

    def transform(self, mask=None):
        if mask is None:
            mask = self.mask
        Xs, Ys = get_XY(mask)
        X, Y = (Xs[mask] - self.xc), (Ys[mask] - self.yc)
        X_rot = X*np.cos(-self.phi) - Y*np.sin(-self.phi)
        Y_rot = Y*np.cos(-self.phi) + X*np.sin(-self.phi)
        r = np.sqrt((X_rot / self.a)**2 + (Y_rot / self.b)**2)
        theta = np.arctan2(Y_rot/self.b, X_rot/self.a)
        return r, theta
        
    def plot_radii(self, mask=None, ax=None):
        if mask is None:
            mask = self.mask
        rs, thetas = self.transform(mask)
        im = np.zeros_like(mask, dtype=float)
        im[mask] = rs
        im[~mask] = np.nan
        if ax is None:
            fig, ax = plt.subplots()
        plt.subplot(ax)
        plt.imshow(im, norm=mpl.colors.Normalize(vmin=rs.min(), vmax=rs.max()))
        ys, xs = np.where(mask)
        ax.set_xlim([xs.min(), xs.max()])
        ax.set_ylim([ys.min(), ys.max()])
        plt.colorbar()
        return ax
    
    def plot_radial_gradient(self, values, mask=None, ax=None, n_bins=10, **kwargs):
        if mask is None:
            mask = self.mask
        R, Theta = self.transform(mask)
        y = values[mask]
        kw = {'x_estimator': np.median,
              'x_bins': n_bins,
              'x_ci': 'sd'}
        kw.update(kwargs)
        if ax is None:
            fig, ax = plt.subplots()
        sns.regplot(x=R, y=y, ax=ax, **kw)
        return ax

    def plot_angular_gradient(self, values, mask=None, ax=None, n_bins=10, **kwargs):
        if mask is None:
            mask = self.mask
        R, Theta = self.transform(mask)
        y = values[mask]
        kw = {'x_estimator': np.median,
              'x_bins': n_bins,
              'x_ci': 'sd'}
        kw.update(kwargs)
        if ax is None:
            fig, ax = plt.subplots()
        sns.regplot(x=Theta, y=y, ax=ax, **kw)
        return ax
    
    def delta_val(self, values, mask=None, n_bins=10, func=np.median,
                  subtract=True):
        if mask is None:
            mask = self.mask
        R, Theta = self.transform(mask)
        y = values[mask]
        bins = np.linspace(R.min()-0.01, R.max()+0.01, n_bins+1)
        ids = np.digitize(R, bins)
        inner = (ids == 1)
        outer = (ids == n_bins)
        if subtract:
            return func(y[outer]) - func(y[inner])
        else:
            return func(y[outer]) / func(y[inner])

