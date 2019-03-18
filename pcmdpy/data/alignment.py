# reduction.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from astropy.io import fits
from astropy.table import Table
from ..utils.utils import PrintRedirect, RegularPrint
from .utils import ALL_FLAGS
import os
with PrintRedirect():
    from drizzlepac import tweakreg, astrodrizzle, tweakback

def _copy_dark_header(image_file, dark_file, outname=None):
    """
    Create and save a dark image, with same header (especially WCS) as an input image
    """
    with fits.open(image_file) as h, fits.open(dark_file) as dark:
        h['SCI',1].data = dark['SCI',1].data * h[0].header['DARKTIME']
        h['SCI',2].data = dark['SCI',2].data * h[0].header['DARKTIME']
        h['ERR',1].data = dark['ERR',1].data
        h['ERR',2].data = dark['ERR',2].data
        h['DQ',1].data = dark['DQ',1].data.astype(np.int16)
        h['DQ',2].data = dark['DQ',2].data.astype(np.int16)
        h[0].header['FILETYPE'] = dark[0].header['FILETYPE']
        if outname is None:
            outname = image_file.replace('.fits', '_drk.fits')
        h.writeto(outname, overwrite=True)
    return outname


def _add_flag(image_file, exp_ratio=0.9):
    hdulist = fits.open(image_file, mode='update')
    flag_hdu = fits.ImageHDU(data=np.zeros_like(hdulist['SCI'].data,
                                                dtype=np.int32))
    flag_hdu.header['EXTNAME'] = 'FLAGS'
    h = flag_hdu.header
    h.add_history('Flags and Masks to ignore in analysis')
    h.add_history('Flags are sum of:')
    h.add_history('   {:d}: CTX mismatch (only partial image coverage)'.format(ALL_FLAGS['CTX']))
    h.add_history('   {:d}: Dark current high'.format(ALL_FLAGS['DARK']))
    h.add_history('   {:d}: Manual source'.format(ALL_FLAGS['MANUAL']))
    h.add_history('   {:d}: SExtractor source'.format(ALL_FLAGS['SEXTRACTOR']))
    h.add_history('   {:d}: Exposure lower than total'.format(ALL_FLAGS['EXPOSURE']))
    N = hdulist[0].header.get('NDRIZIM', 1)
    ctx_full = np.sum([2**i for i in range(0, N)]) // 3
    mask = (hdulist['CTX'].data != ctx_full) & (hdulist['CTX'].data != 2*ctx_full)
    flag_hdu.data[mask] += ALL_FLAGS['CTX']
    exp_mask = (hdulist['WHT'].data < exp_ratio*hdulist[0].header['EXPTIME'])
    flag_hdu.data[exp_mask] += ALL_FLAGS['EXPOSURE']
    if 'FLAGS' in hdulist:
        hdulist.pop('FLAGS')
    hdulist.insert(-1, flag_hdu)
    hdulist.close()


# def _add_dark(image_file, dark_file, dark_med_factor=3):
#     """
#     Add the median dark current to an image, masking out pixels with dark
#     current larger than some factor, and return the median dark current
#     """
#     hdulist = fits.open(image_file, mode='update')
#     if 'FLAGS' not in hdulist:
#         hdulist.close()
#         _add_flag(image_file)
#         hdulist = fits.open(image_file, mode='update')
#     mask = (hdulist['FLAGS'].data == 0)
#     with fits.open(dark_file) as dark:
#         dark_med = np.median(dark['SCI'].data[mask])
#         too_dark = (dark['SCI'].data > dark_med_factor*dark_med)
#         hdulist['FLAGS'].data[too_dark] += ALL_FLAGS['DARK']
#         hdulist['SCI'].data += dark_med
#         dark_hdu = fits.ImageHDU(data=dark['SCI'].data)
#         dark_hdu.header['EXTNAME'] = 'DARK'
#         h = dark_hdu.header
#         h.add_history('Dark Current Image')
#         h.add_history('Median added to image: {:.3f}'.format(dark_med))
#         h.add_history('Factor used as upper limit: {:d} x'.format(dark_med_factor))
#         hdulist.insert(-1, dark_hdu)
#     hdulist[0].header['DRKNOISE'] = (dark_med, 'Median Dark counts added')
#     hdulist[0].header['DARKFACT'] = (dark_med_factor, 'Factor times DRKNOISE masked out')
#     hdulist.close()  # save results
#     return dark_med

def myalign(input_files,
            updatewcs=False, clean=True, logfile=None, verbose=False,
            wcsname='ALIGN_ALL', refimage=None,
            **kwargs):
    logger = RegularPrint() if verbose else PrintRedirect(logfile)
    if isinstance(input_files, str):
        input_files = [input_files]
    already_updated = []
    for file in input_files:
        if 'WCSMAN' not in fits.getheader(file, extname='PRIMARY'):
            fits.setval(file, 'WCSMAN', 'PRIMARY', value=False,
                        comment='Has WCS been manually updated?')
        # Only update WCS if has not been performed on ANY of the inputs
        elif fits.getval(file, 'WCSMAN', 'PRIMARY') is True:
            already_updated.append(file.rpartition('/')[-1])
    if updatewcs:
        if len(already_updated) > 0:
            updatewcs = False
            print('--NOT UPDATING WCS because manually updated files exist: '
                  f'{already_updated}')
        else:
            print('--Updating WCS manually as requested')
    # Set default options
    t_kwgs = {
        'conv_width': 4.0,
        'threshold': 200,
        'shiftfile': False,
        'updatehdr': True,
        'writecat': False,
        'clean': True,
        'residplot': 'NoPlot',
        'see2dplot': False,
        'reusename': True,
        'interactive': False,
        'updatewcs': updatewcs,
        'enforce_user_order': True,
        'expand_refcat': True,
        'wcsname': wcsname,
        'refimage': (refimage or ''),
    }
    t_kwgs.update(kwargs)
    # Initial Alignment
    with logger:
        print(t_kwgs)
        tweakreg.TweakReg(input_files, **t_kwgs)
    if t_kwgs['updatewcs']:
        for file in input_files:
            fits.setval(file, 'WCSMAN', 'PRIMARY', value=True)
    

def mydrizzle(input_files, output_file,
              kernel='lanczos3', refimage=None,
              logfile=None, verbose=False, clean=True, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    N_files = len(input_files)
    logger = RegularPrint() if verbose else PrintRedirect(logfile)
    d_kwgs = {
        'clean': clean,
        'build': True,
        'skysub': False,
        'final_units': 'counts',
        'mdriztab': False,
        'combine_type': ('median' if N_files >= 4 else 'minmed'),
        'combine_nhigh': (1 if N_files == 4 else 0),
        'final_kernel': kernel,
        # 'wcskey': wcsname,
        'final_rot': 0.,
        'final_wcs': True,
        'final_refimage': (refimage or ''),
    }
    d_kwgs.update(kwargs)
    print(f'    Drizzling Images with {kernel} kernel to {output_file}')
    with logger:
        print(d_kwgs)
        astrodrizzle.AstroDrizzle(
            input_files, output=output_file, **d_kwgs)


def drizzle_many(inputs_dict, output_dict={},
                 # darks_dict={},
                 kernel='lanczos3',
                 reference_filter=None, updatewcs=False,
                 clean=True, logfile=None, verbose=False,
                 wcsname=None, exp_ratio=0.9,
                 realign_only=False,
                 # dark_med_factor=3,
                 align_kwargs={}, drizzle_kwargs={},
                 realign_kwargs={}, tweakback_kwargs={},
                 redrizzle_kwargs={}):
    """
    Drizzle together multiple sets of images to same aligned pixel scale


    Parameters
    ----------
    inputs_dict : dictionary (Filter -> list of input files)
    output_dict : dictionary (Filter -> name of output file)
    darks_dict : dictionary (Filter -> name of dark file)

    """
    logger = RegularPrint() if verbose else PrintRedirect(logfile)
    all_filters = list(inputs_dict.keys())
    if reference_filter is None:
        reference_filter = all_filters[0]
    assert reference_filter in all_filters, (
        f"{reference_filter} not in provided dictionary")
    # put reference filter first
    all_filters.remove(reference_filter)
    all_filters = [reference_filter] + list(all_filters)

    kwargs = {
            'updatewcs': updatewcs, 'clean': clean, 'logfile': logfile,
            'verbose': verbose}
    align_kwargs = {**kwargs, **align_kwargs}
    kwargs = {
        'kernel': kernel, 'refimage': None, 'logfile': logfile,
        'verbose': verbose, 'clean': clean}
    drizzle_kwargs = {**kwargs, **drizzle_kwargs}

    # Align and drizzle individual filters
    if not realign_only:
        print('*******************************')
        print('Initial Alignment and Drizzling')
        for filt in all_filters:
            print(f'--{filt}: Aligning')
            input_files = inputs_dict[filt]
            output_file = output_dict[filt]
            align_kwargs['wcsname'] = (f'INITIAL_{filt}')
            myalign(input_files, **align_kwargs)
            print(f'--{filt}: Drizzling')
            mydrizzle(input_files, output_file, **drizzle_kwargs)

    # Align drizzled images
    kwargs = {
        'threshold': 200, 'nclip': 10,
        'refimage': output_dict[reference_filter],
        'wcsname': f'TWEAK_{reference_filter}',
        'clean': clean, 'verbose': verbose,
        'logfile': logfile, 'updatewcs': False,
        'refimagefindcfg': {'threshold': 200},
    }
    realign_kwargs = {**kwargs, **realign_kwargs}
    kwargs = {
        'force': True,
        'verbose': verbose
    }
    tweakback_kwargs = {**kwargs, **tweakback_kwargs}
    kwargs = {
        'final_refimage': output_dict[reference_filter],
        'final_wcs': True,
        'final_rot': None,
    }
    kwargs = {**drizzle_kwargs, **kwargs}
    redrizzle_kwargs = {**redrizzle_kwargs, **kwargs}
    print('*****************************************************')
    print(f'Realign and Redrizzle relative to {reference_filter}')
    for filt in all_filters[1:]:
        input_files = inputs_dict[filt]
        output_file = output_dict[filt]
        print(f'--{filt}: Realigning')
        myalign(output_file, **realign_kwargs)
        print(f'--{filt}: TweakBack')
        with logger:
            print(tweakback_kwargs)
            tweakback.tweakback(output_file, input=input_files,
                                **tweakback_kwargs)
        input_files = inputs_dict[filt]
        output_file = output_dict[filt]
        print(f'--{filt}: Redrizzling')
        mydrizzle(input_files, output_file, **redrizzle_kwargs)
    
    # dark_final = {filt: '{:s}_dark_drz.fits'.format(filt) for filt in all_filters}

    # # Create temorary dark files
    # print('--Creating temporary dark files')
    # dark_inputs = {}
    # dark_outputs = {}
    # with logger:
    #     for filt in all_filters:
    #         if filt in darks_dict:
    #             dark_inputs[filt] = []
    #             for f in inputs_dict[filt]:
    #                 newfile = _copy_dark_header(f, darks_dict[filt])
    #                 dark_inputs[filt].append(newfile)
    #             dark_outputs[filt] = 'temp_dark_{:s}_drz.fits'.format(filt)
                    
    # Drizzle the images
    # print('--Drizzling Images with {:s} kernel'.format(kernel))
    # with logger:
    #     # Drizzle first filter
    #     first_filt = all_filters[0]
    #     if len(inputs_dict[first_filt]) >= 4:
    #         old_mode = d_kwgs.pop('combine_type', 'minmed')
    #         d_kwgs['combine_type'] = 'median'
    #     if len(inputs_dict[first_filt]) == 4:
    #         old_nhigh = d_kwgs.pop('combine_nhigh', 0)
    #         d_kwgs['combine_nhigh'] = 1
    #     astrodrizzle.AstroDrizzle(
    #         inputs_dict[first_filt], output=output_dict[first_filt], **d_kwgs)
    #     if len(inputs_dict[first_filt]) >= 4:
    #         d_kwgs['combine_type'] = old_mode
    #     if len(inputs_dict[first_filt]) == 4:
    #         d_kwgs['combine_nhigh'] = old_nhigh
    #     for filt in all_filters[1:]:
    #         if len(inputs_dict[filt]) >= 4:
    #             old_mode = d_kwgs.pop('combine_type', 'minmed')
    #             d_kwgs['combine_type'] = 'median'
    #         if len(inputs_dict[filt]) == 4:
    #             old_nhigh = d_kwgs.pop('combine_nhigh', 0)
    #             d_kwgs['combine_nhigh'] = 1
    #         # Align all other filters relative to first
    #         astrodrizzle.AstroDrizzle(
    #             inputs_dict[filt], output=output_dict[filt],
    #             final_refimage=output_dict[first_filt],
    #             **d_kwgs)
    #         if len(inputs_dict[filt]) >= 4:
    #             d_kwgs['combine_type'] = old_mode
    #         if len(inputs_dict[filt]) == 4:
    #             d_kwgs['combine_nhigh'] = old_nhigh
    #         _add_flag(output_dict[filt], exp_ratio=exp_ratio)
        # Align and drizzle the dark images
        # dark_kwgs = d_kwgs.copy()
        # dark_kwgs.update(
        #     {'driz_separate': False,
        #      'median': False,
        #      'blot': False,
        #      'driz_cr': False}
        # )
        # for filt in dark_inputs:
        #     if len(dark_inputs[filt]) < 3:
        #         old_mode = d_kwgs.pop('combine_type', 'median')
        #         d_kwgs['combine_type'] = 'minmed'
        #     astrodrizzle.AstroDrizzle(
        #         dark_inputs[filt], output=dark_outputs[filt],
        #         final_refimage=output_dict[first_filt], **dark_kwgs)
        #     if len(dark_inputs[filt]) < 3:
        #         d_kwgs['combine_type'] = old_mode

    print('*************************')
    print('Accounting for Noise and Errors')
    # Compute and add dark values
    # dark_meds = {}
    # for filt in dark_outputs:
    #     dark_meds[filt] = _add_dark(output_dict[filt], dark_outputs[filt],
    #                                 dark_med_factor=dark_med_factor)

    # Add flag HDUs
    for filt in all_filters:
        _add_flag(output_dict[filt], exp_ratio=exp_ratio)

    # Compute Sky noise
    sky_vals = {}
    exposures = {}
    for filt in all_filters:
        with fits.open(output_dict[filt], mode='update') as h:
            sky_vals[filt] = Table(h['HDRTAB'].data).to_pandas()[['MDRIZSKY']].values.sum() / 2.
            exposures[filt] = h[0].header['EXPTIME']
            h[0].header['SKYNOISE'] = (sky_vals[filt], 'Total sky value (sum of MDRIZSKY)')
            # h[0].header['NOISE'] = (sky_vals[filt] + dark_meds.get(filt, 0),
            #                         'Total noise (SKYNOISE + DRKNOISE)')
            # Track later steps
            h[0].header['BKGDCOMP'] = ('PENDING', 'Has background been computed?')
            h[0].header['SEXTRACT'] = ('PENDING', 'Have SExtractor sources been masked?')
            h[0].header['MANUAL'] = ('PENDING', 'Have manual sources been masked?')
            h[0].header['REGIONS'] = ('PENDING', 'Have PCMD regions been computed?')

    for filt in all_filters:
        # noise = dark_meds.get(filt, 0.) + sky_vals[filt]
        noise = sky_vals[filt]
        exp = exposures[filt]
        print(f'--Sky Noise ({filt}): {noise:.1f} counts ({noise/exp:.3f} cps)')
        # print('--Sky: {:.1f} counts ({:.3f} cps)'.format(sky_vals[filt], sky_vals[filt]/exp))
        # print('--Dark: {:.1f} counts ({:.3f} cps)'.format(dark_meds.get(filt, 0), dark_meds.get(filt, 0)/exp))

    # if clean:
    #     print('--Removing temporary files')
    #     all_files = []
    #     for v in dark_inputs.values():
    #         if isinstance(v, list):
    #             all_files.extend(v)
    #         else:
    #             all_files.append(v)
    #     for v in dark_outputs.values():
    #         if isinstance(v, list):
    #             all_files.extend(v)
    #         else:
    #             all_files.append(v)
    #     for f in all_files:
    #         try:
    #             os.remove(f)
    #         except FileNotFoundError:
    #             pass

    print('--Done')
