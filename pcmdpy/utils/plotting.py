# plotting.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .utils import make_hess
from ..galaxy.galaxy import SSPSimple


def plot_rgb_image(images, extent=None, ax=None,
                   clip_percent=98, clip_vals=None, r_index=0,
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
    if clip_vals is not None:
        for i in range(3):
            images[:, :, i] /= clip_vals[i]
    else:
        for i in range(3):
            images[:, :, i] /= np.percentile(images[:, :, i], clip_percent)
    images[images <= 0.] = 0.
    images[images >= 1.] = 1.
    ax.imshow(images, origin='lower', aspect='equal',
              extent=extent, interpolation='none')
    return ax


def plot_pcmd(pcmd, bins=None, axes=None, norm=None, hist2d_kwargs={},
              title=None, keep_limits=False):
    n_bands = pcmd.shape[0]
    if bins is None:
        mins = np.min(pcmd, axis=-1)
        maxs = np.max(pcmd, axis=-1)
        bins = [np.arange(mins[i], maxs[i], 0.05) for i in range(n_bands)]
    if axes is None:
        fig, axes = plt.subplots(ncols=n_bands-1)
    if n_bands == 2:
        axes = [axes]
    if norm is None:
        norm = mpl.colors.LogNorm()
    if 'cmap' not in hist2d_kwargs:
        hist2d_kwargs['cmap'] = 'viridis'
    for i, ax in enumerate(axes):
        # record original axis limits, in case overwritten by hist2d
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        H, xbins, ybins, _ = ax.hist2d(pcmd[i+1], pcmd[0],
                                       bins=[bins[i+1], bins[0]], norm=norm,
                                       **hist2d_kwargs)
        xl += ax.get_xlim()
        yl += ax.get_ylim()
        if keep_limits:
            ax.set_xlim([min(xl), max(xl)])
            ax.set_ylim([max(yl), min(yl)])
    if title is not None:
        axes[0].set_title(title)
    return axes, H, bins, norm


def plot_pcmd_residual(pcmd_model, pcmd_compare, like_mode=2, bins=None,
                       axes=None, norm=None, title='', im_kwargs={},
                       cbar_kwargs={}):
    n_bands = pcmd_model.shape[0]
    n_compare = pcmd_compare.shape[1]
    n_model = pcmd_model.shape[1]
    if axes is None:
        fig, axes = plt.subplots(ncols=n_bands-1)
    if n_bands == 2:
        axes = [axes]
    if bins is None:
        combo = np.append(pcmd_model, pcmd_compare, axis=-1)
        mag_bins = [np.min(combo[0]), np.max(combo[0])]
        color_bins = [np.min(combo[1:]), np.max(combo[1:])]
        bins = np.append([mag_bins], [color_bins for _ in range(1, n_bands)])
    counts_model, hess_model, err_model = make_hess(pcmd_model, bins, boundary=False)
    counts_compare, hess_compare, err_compare = make_hess(pcmd_compare, bins, boundary=False)
    
    if like_mode == 1:
        root_nn = np.sqrt(n_model * n_compare)
        term1 = np.log(root_nn + n_compare * counts_model)
        term2 = np.log(root_nn + n_model * counts_compare)
        chi = term1 - term2
    elif like_mode == 2:
        denom = np.sqrt(2. * (err_model**2. + err_compare**2.))
        hess_diff = (hess_model - hess_compare)
        chi = hess_diff / denom
    chi_sign = np.sign(chi)
    chi2 = chi**2
    chi2_max = np.max(chi2)
    if norm is None:
        kwargs = {'linthresh': 1., 'linscale': 0.1}
        kwargs.update(cbar_kwargs)
        norm = mpl.colors.SymLogNorm(vmin=-chi2_max, vmax=chi2_max,
                                     **kwargs)
    for i, ax in enumerate(axes):
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        plt.subplot(ax)
        # record original axis limits, in case overwritten by hist2d
        kwargs = {'cmap': 'bwr_r'}
        kwargs.update(im_kwargs)
        plt.imshow((chi_sign*chi2)[i], norm=norm, origin='lower',
                   aspect='auto', extent=(bins[i+1][0], bins[i+1][-1],
                                          bins[0][0], bins[0][-1]),
                   **kwargs)
        xl += ax.get_xlim()
        yl += ax.get_ylim()
        ax.set_xlim([min(xl), max(xl)])
        ax.set_ylim([max(yl), min(yl)])
        ax.set_title(title + r' $\chi^2= $' + '{:.2e}'.format(np.sum(chi2)))
    return axes, chi, bins, norm


def plot_isochrone(iso_model, dmod=30., gal_model=None, axes=None,
                   mag_system='vega', update_axes=True, **kwargs):
    if axes is None:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(ncols=(iso_model.num_filters-1), sharey=True)
    if gal_model is None:
        gal_model = SSPSimple(np.array([0., -2., 1., 10.]),
                              dmod=dmod)
    names = iso_model.filter_names
    for age, feh, _, d_mod in gal_model.iter_SSPs():
        _, mags = iso_model.get_isochrone(age, feh, mag_system=mag_system)
        mags += d_mod
        if iso_model.num_filters == 2:
            axes.plot(mags[1]-mags[0], mags[0], 'k-',
                      label='age:{0:.1f}, feh:{1:.1f}'.format(age, feh),
                      **kwargs)
            if update_axes:
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
                if update_axes:
                    ax.set_xlabel('{0:s} - {1:s}'.format(names[i+1], names[i]),
                                  fontsize='x-large')
                    ax.set_ylabel(names[0], fontsize='x-large')
                    yl = ax.get_ylim()
                    ax.set_ylim([max(yl), min(yl)])
    return axes


def step_plot(x, y, ax=None, **kwargs):
    assert len(x) == len(y) + 1
    y = np.append(y, y[-1])
    if ax is None:
        ax = plt
    kwargs['linestyle'] = kwargs.pop('ls', '-')
    ax.step(x, y, where='post', **kwargs)


def step_fill(x, y1, y2, ax=None, **kwargs):
    assert len(x) == len(y1) + 1
    assert len(y1) == len(y2)
    x = np.repeat(x, 2)[1:-1]
    y1 = np.repeat(y1, 2)
    y2 = np.repeat(y2, 2)
    if ax is None:
        ax = plt
    kwargs['linestyle'] = kwargs.pop('ls', '-')
    ax.fill_between(x, y1=y1, y2=y2, **kwargs)

    
