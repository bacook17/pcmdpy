# extraction.py
# Ben Cook (bcook@cfa.harvard.edu)
import numpy as np
import pyregion
from astropy.io import fits
from tqdm import tqdm
from .utils import filter_from_fits
from ..instrument.filter import Filter


def compute_regions(image_file, region_file, xc=None, yc=None):
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
    for i in tqdm(range(len(regions) - 1)):
        mask = regions[i:i+1].get_mask(shape=shape)
        matrix[mask] += 1.0
    matrix[Q2] += 0.25
    matrix[Q3] += 0.5
    matrix[Q4] += 0.75
    matrix *= 4.0
    matrix = np.round(matrix, decimals=0).astype(np.int32)
    matrix -= 3
    matrix[matrix <= 0] = 0
    matrix[~good_pixels] = -1
    return matrix


def add_regions(input_dict, region_file,
                base_filter=None, xc=None, yc=None):
    all_filters = list(input_dict.keys())
    filt = base_filter or all_filters[0]
    regions_matrix = compute_regions(input_dict[filt], region_file,
                                     xc=xc, yc=yc)
    reg_hdu = fits.ImageHDU(data=regions_matrix)
    reg_hdu.header['EXTNAME'] = 'REGIONS'
    h = reg_hdu.header
    h.add_history('Regions extracted from DS9 Contours')
    h.add_history('Region file used: {:s}'.format(region_file))
    h.add_history('Base filter used: {:s}'.format(filt))
    for f in all_filters:
        with fits.open(input_dict[f], mode='update') as h:
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
