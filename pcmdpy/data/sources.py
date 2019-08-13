# sources.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from astropy.io import fits
import sep
from .utils import ALL_FLAGS, regions_to_mask, _add_masked
from .alignment import _add_flag


def estimate_background(image_file, bkg_kwargs={}):
    """
    Estimate the smooth background of an image
    """
    hdulist = fits.open(image_file, mode='update')
    image = hdulist['SCI'].data
    kwargs = {}
    kwargs['bw'] = kwargs['bh'] = 8  # size of background boxes
    kwargs['fw'] = kwargs['fh'] = 3  # size of the filters
    kwargs['fthresh'] = 0  # filter threshold
    kwargs.update(bkg_kwargs)
    bkg = sep.Background(image.astype(np.float64),
                         **bkg_kwargs)
    bkg_hdu = fits.ImageHDU(data=bkg.back())
    bkg_hdu.header['EXTNAME'] = 'BKGD'
    rms_hdu = fits.ImageHDU(data=bkg.rms())
    rms_hdu.header['EXTNAME'] = 'BKGDRMS'
    sub_hdu = fits.ImageHDU(data=(image / bkg.back()))
    sub_hdu.header['EXTNAME'] = 'BKGDSUB'
    for h in [bkg_hdu.header, rms_hdu.header, sub_hdu.header, hdulist['FLAGS'].header]:
        h.add_history('SExtractor Background Params:')
        h.add_history('   Background Box Size (pixels): {:d}'.format(kwargs['bw']))
        h.add_history('   Background Filter Size (pixels): {:d}'.format(kwargs['fw']))
        h.add_history('   Background Filter Threshold (pixels): {:.2f}'.format(kwargs['fthresh']))
        h['GLOBBKG'] = bkg.globalback
        h['GLOBRMS'] = bkg.globalrms
        for k, v in kwargs.items():
            if k not in ['bw', 'bh', 'fw', 'fh', 'fthresh']:
                h.add_history('   {:}: {:}'.format(k, v))
    if 'BKGD' in hdulist:
        hdulist.pop('BKGD')
    if 'BKGDRMS' in hdulist:
        hdulist.pop('BKGDRMS')    
    if 'BKGDSUB' in hdulist:
        hdulist.pop('BKGDSUB')
    hdulist.insert(-1, sub_hdu)
    hdulist.insert(-1, rms_hdu)
    hdulist.insert(-1, bkg_hdu)
    hdulist[0].header['BKGDCOMP'] = "COMPLETE"
    hdulist.close()
    return bkg_hdu


def mask_sources_manual(image_file, region_file):
    hdulist = fits.open(image_file, mode='update')
    if 'FLAGS' not in hdulist:
        hdulist.close()
        _add_flag(image_file)
        hdulist = fits.open(image_file, mode='update')
    mask = regions_to_mask(region_file, image_file)
    # Unset pixels already flagged
    old_mask = (hdulist['FLAGS'].data & ALL_FLAGS['MANUAL']).astype(np.bool)
    hdulist['FLAGS'].data[old_mask] -= ALL_FLAGS['MANUAL']
    hdulist['FLAGS'].data[mask] += ALL_FLAGS['MANUAL']
    h = hdulist['FLAGS'].header
    h.add_history('Manual Regions Masked from file:')
    h.add_history('   {:s}'.format(region_file))
    hdulist[0].header['MANUAL'] = "COMPLETE"
    hdulist.close()
    _add_masked(image_file)


def mask_sources_auto(image_file, threshold=10.0, r_scale=10.0,
                      global_rms=False, max_npix_object=500,
                      obj_kwargs={}, **kwargs):
    hdulist = fits.open(image_file, mode='update')
    if 'FLAGS' not in hdulist:
        hdulist.close()
        _add_flag(image_file)
        hdulist = fits.open(image_file, mode='update')
    if 'BKGD' not in hdulist:
        hdulist.close()
        estimate_background(image_file, kwargs.get('bkg_kwargs', {}))
        hdulist = fits.open(image_file, mode='update')
    kwargs = {
        'thresh': threshold,
        'minarea': 9,
    }
    kwargs.update(obj_kwargs)
    image = np.copy(hdulist['SCI'].data)
    sub_im = image - hdulist['BKGD'].data
    # Undo previous SEXTRACTOR source masks
    old_mask = (hdulist['FLAGS'].data & ALL_FLAGS['SEXTRACTOR']).astype(np.bool)
    hdulist['FLAGS'].data[old_mask] -= ALL_FLAGS['SEXTRACTOR']
    mask = np.zeros_like(image, dtype=np.bool)
    if global_rms:
        err = hdulist['BKGD'].header['GLOBRMS']
    else:
        err = hdulist['BKGDRMS'].data.byteswap().newbyteorder()
    objects = sep.extract(sub_im, err=err,
                          mask=mask, segmentation_map=False,
                          **kwargs)
    to_use = (objects['npix'] < max_npix_object)
    mask = np.zeros_like(image, dtype=np.bool)
    sep.mask_ellipse(mask, objects['x'][to_use], objects['y'][to_use],
                     objects['a'][to_use], objects['b'][to_use],
                     objects['theta'][to_use], r=r_scale)
    # unset pixels already flagged
    hdulist['FLAGS'].data[mask] += ALL_FLAGS['SEXTRACTOR']
    h = hdulist['FLAGS'].header
    h.add_history('SExtractor Regions Masked')
    h.add_history('   Detection Threshold (sigma): {:.2f}'.format(kwargs['thresh']))
    h.add_history('   Min Source Size (pixels): {:d}'.format(kwargs['minarea']))
    hdulist[0].header['SEXTRACT'] = "COMPLETE"
    hdulist.close()
    _add_masked(image_file)
