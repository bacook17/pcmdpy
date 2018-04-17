# plotting.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_rgb_image(images, extent=None, ax=None,
                   clip_percent=98, r_index=0,
                   g_index=1, b_index=2):
    if ax is None:
        fig, ax = plt.subplots()
    if images.shape[-1] != 3:
        assert images.shape[0] == 3, 'not proper RGB image shape'
        ims_new = np.zeros((images.shape[1], images.shape[2], 3))
        for i in range(3):
            ims_new[:, :, i] = images[i]
        images = np.copy(ims_new)
    else:
        images = np.copy(images)
    for i in range(images.shape[-1]):
        images[:, :, i] /= np.percentile(images[:, :, i], clip_percent)
    images[images <= 0.] = 0.
    images[images >= 1.] = 1.
    ax.imshow(images, origin='lower', aspect='equal',
              extent=extent)
    return ax


def plot_pcmd(pcmd, bins=100, ax=None, norm=None, hist2d_kwargs={}):
    if ax is None:
        fig, ax = plt.subplots()
    if norm is None:
        norm = mpl.colors.LogNorm()
    if 'cmap' not in hist2d_kwargs:
        hist2d_kwargs['cmap'] = 'viridis'
    H, xbins, ybins, _ = ax.hist2d(pcmd[1], pcmd[0], bins=bins, norm=norm,
                                   **hist2d_kwargs)
    return ax, H, [xbins, ybins], norm


def plot_pcmd_residual(pcmd_model, pcmd_compare, bins=100, ax=None, norm=None,
                       im_kwargs={}):
    if ax is None:
        fig, ax = plt.subplots()
    if 'cmap' not in im_kwargs:
        im_kwargs['cmap'] = 'RdBu_r'
    n_compare = pcmd_compare.shape[1]
    counts_compare, xbins, ybins = np.histogram2d(pcmd_compare[1],
                                                  pcmd_compare[0], bins=bins)
    err_compare = np.sqrt(counts_compare)
    err_compare += 2. * np.exp(-err_compare)
    counts_compare /= n_compare
    err_compare /= n_compare
    bins = [xbins, ybins]
    n_model = pcmd_model.shape[1]
    counts_model, _, _ = np.histogram2d(pcmd_model[1], pcmd_model[0],
                                        bins=bins)
    err_model = np.sqrt(counts_model)
    err_model += 2. * np.exp(-err_model)
    counts_model /= n_model
    err_model /= n_model

    denom = np.sqrt(2. * (err_model**2. + err_compare**2.))
    chi = (counts_model - counts_compare) / denom
    chi_max = np.max(np.abs(chi))
    if norm is None:
        norm = mpl.colors.Normalize(vmin=-chi_max, vmax=chi_max)
    ax.imshow(chi.T, norm=norm, origin='lower',
              extent=(xbins[0], xbins[1],
                      ybins[0], ybins[1]),
              **im_kwargs)
    return ax, chi, bins, norm


def plot_isochrone(galaxy, iso_model, axes=None, system='vega', **kwargs):
    if axes is None:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(ncols=(iso_model.num_filters-1), sharey=True)
    names = iso_model.filter_names
    for age, feh, _, d_mod in galaxy.iter_SSPs():
        _, mags = iso_model.get_isochrone(age, feh, system=system)
        mags += d_mod
        if iso_model.num_filters == 2:
            axes.plot(mags[1]-mags[0], mags[0], 'k-',
                      label='age:{0:.1f}, feh:{1:.1f}'.format(age, feh),
                      **kwargs)
            axes.set_xlabel('{0:s} - {1:s}'.format(names[1], names[0]),
                            fontsize='x-large')
            axes.set_ylabel(names[0], fontsize='x-large')
            yl = axes.get_ylim()
            axes.set_ylim([max(yl), min(yl)])
        else:
            for i, ax in enumerate(axes):
                ax.plot(mags[i+1]-mags[i], mags[0], 'k-',
                        label='age:{0:.1f}, feh:{1:.1f}'.format(age, feh),
                        **kwargs)
                ax.set_xlabel('{0:s} - {1:s}'.format(names[i+1], names[i]),
                              fontsize='x-large')
                ax.set_ylabel(names[0], fontsize='x-large')
                yl = ax.get_ylim()
                ax.set_ylim([max(yl), min(yl)])
    return axes



              
