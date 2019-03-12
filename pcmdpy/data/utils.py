import numpy as np
from astropy.io import fits
from ..instrument.filter import AVAILABLE_FILTERS
import pyregion
import pysynphot as pyS

ALL_FLAGS = {
    'CTX': 2**0,
    'EXPOSURE': 2**1,
    'DARK': 2**2,
    'SEXTRACTOR': 2**3,
    'MANUAL': 2**4,
}


def regions_to_mask(region_file, image_file):
    hdulist = fits.open(image_file)
    mask = pyregion.open(region_file).as_imagecoord(
        header=hdulist[0].header).get_mask(shape=hdulist['SCI'].shape)
    return mask


def _add_masked(image_file, mask_val=np.nan,
                mask_flags=ALL_FLAGS.values()):
    hdulist = fits.open(image_file, mode='update')
    if 'MASKED' in hdulist:
        hdulist.pop('MASKED')
    clean_hdu = hdulist['SCI'].copy()
    clean_hdu.header['EXTNAME'] = 'MASKED'
    mask = np.zeros_like(hdulist['FLAGS'].data, dtype=bool)
    for f in mask_flags:
        mask |= (hdulist['FLAGS'].data & f).astype(np.bool)
    clean_hdu.data[mask] = mask_val
    if 'BKGDSUB' in hdulist:
        if 'MASKEDSUB' in hdulist:
            hdulist.pop('MASKEDSUB')
        sub_hdu = hdulist['BKGDSUB'].copy()
        sub_hdu.header['EXTNAME'] = 'MASKEDSUB'
        sub_hdu.data[hdulist['FLAGS'].data > 0] = mask_val
        hdulist.insert(2, sub_hdu)
    hdulist.insert(2, clean_hdu)
    hdulist.close()


def combine_flags(file_dict):
    all_filters = list(file_dict.keys())
    h1 = fits.open(file_dict[all_filters[0]])
    flags = np.zeros_like(h1['FLAGS'].data, dtype=np.int32)
    h1.close()
    for i, filt in enumerate(all_filters):
        with fits.open(file_dict[filt]) as h:
            flags += 2**i * h['FLAGS'].data
    for filt in all_filters:
        with fits.open(file_dict[filt], mode='update') as h:
            h['FLAGS'].data = flags
        _add_masked(file_dict[filt])


def compute_zpts(instrument, detector, band, mjd):
    bandpass = pyS.ObsBandpass('{:s},{:s},{:s},mjd#{:d}'.format(instrument, detector, band, mjd))
    spec_bb = pyS.BlackBody(50000)
    spec_bb_norm = spec_bb.renorm(1, 'counts', bandpass)
    obs = pyS.Observation(spec_bb_norm, bandpass)
    zps = {}
    zps['vega'] = obs.effstim('vegamag')
    zps['st'] = obs.effstim('stmag')
    zps['ab'] = obs.effstim('abmag')
    return zps


def filter_from_fits(file_name):
    with fits.open(file_name) as hdu:
        header = hdu[0].header
    instrument = header['INSTRUME'].lower().strip(' ')
    detector = header['DETECTOR'].lower().strip(' ')
    if detector == 'wfc':
        detector = 'wfc1'
    band = None
    for k in ['FILTER', 'FILTER1', 'FILTER2']:
        if k in header:
            b_temp = header[k]
            if 'CLEAR' in b_temp:
                continue
            else:
                band = b_temp
                break
    if band is None:
        raise KeyError('Unable to identify filter from FITS file')
    mjd = int(header['EXPSTART'])
    exposure = header['EXPTIME']
    zpts = compute_zpts(instrument, detector, band, mjd)
    if band in AVAILABLE_FILTERS:
        filt = AVAILABLE_FILTERS[band](
            exposure=exposure,
            zpt_vega=zpts['vega'],
            zpt_ab=zpts['ab'],
            zpt_st=zpts['st'])
    else:
        filt = None
    print('Filter: {:s}'.format(band))
    print('Observation Date: {:d} (MJD)'.format(mjd))
    print('Vega ZeroPoint: {:.4f}'.format(zpts['vega']))
    print('AB ZeroPoint: {:.4f}'.format(zpts['ab']))
    print('ST ZeroPoint: {:.4f}'.format(zpts['st']))
    print('Exposure Time: {:.1f}'.format(exposure))
    if filt is not None:
        print('A pre-made filter is available')
    else:
        print("A custom filter must be made (no matching filter found)")
    return filt
