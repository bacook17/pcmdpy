# plotting.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ..utils.utils import make_hess
from ..simulation.driver import Driver
from dynesty.plotting import _hist2d as dyhist


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


def plot_pcmd(pcmd, bins=None, ax=None, norm=None, hist2d_kwargs={},
              title=None, keep_limits=False):
    """
    Arguments
    ---------

    Returns
    -------
    ax : List of matplotlib axes objects representing image(s)
    bins : List of bins used to make Hess image
    norm : matplotlib normalization object

    """
    n_bands = pcmd.shape[0]
    if bins is None:
        mins = np.min(pcmd, axis=-1)
        maxs = np.max(pcmd, axis=-1)
        bins = [np.arange(mins[i], maxs[i], 0.05) for i in range(n_bands)]
    if ax is None:
        fig, ax = plt.subplots(ncols=n_bands-1)
    if n_bands == 2:
        ax = [ax]
    if norm is None:
        norm = mpl.colors.LogNorm()
    if 'cmap' not in hist2d_kwargs:
        hist2d_kwargs['cmap'] = 'viridis'
    for i, a in enumerate(ax):
        # record original axis limits, in case overwritten by hist2d
        xl = a.get_xlim()
        yl = a.get_ylim()
        H, xbins, ybins, _ = a.hist2d(pcmd[i+1], pcmd[0],
                                       bins=[bins[i+1], bins[0]], norm=norm,
                                       **hist2d_kwargs)
        xl += a.get_xlim()
        yl += a.get_ylim()
        if keep_limits:
            a.set_xlim([min(xl), max(xl)])
            a.set_ylim([max(yl), min(yl)])
    if title is not None:
        ax[0].set_title(title)
    return ax, bins, norm


def plot_pcmd_contours(pcmd, ax=None, smooth=0.01, sig_levels=[1, 2, 3, 4],
                       title=None, keep_limits=False, color=None, alpha=1.0,
                       fill_contours=False, **hist_kwargs):
    """
    Returns
    -------
    fig, ax
    """
    n_bands = pcmd.shape[0]
    if ax is None:
        fig, ax = plt.subplots(ncols=n_bands-1)
    else:
        fig = ax.get_figure()
    if n_bands == 2:
        ax = [ax]
    levels = 1.0 - np.exp(-0.5 * np.array(sig_levels)**2)
    if color is None:
        color = plt.rcParams.get('lines.color', 'k')
    kwargs = {'ax': ax[0],
              'levels': levels,
              'smooth': smooth,
              'plot_contours': True,
              'plot_density': False,
              'fill_contours': fill_contours,
              'no_fill_contours': True,
              'color': color}
    kwargs['contour_kwargs'] = hist_kwargs.pop('contour_kwargs', {})
    kwargs['contour_kwargs']['alpha'] = alpha
    kwargs.update(hist_kwargs)
    for i, a in enumerate(ax):
        xl = a.get_xlim()
        yl = a.get_ylim()
        kwargs['ax'] = a
        dyhist(pcmd[i+1], pcmd[0], **kwargs)
        xl += a.get_xlim()
        yl += a.get_ylim()
        if keep_limits:
            a.set_xlim([min(xl), max(xl)])
            a.set_ylim([max(yl), min(yl)])
    if title is not None:
        ax[0].set_title(title)
    return (fig, ax)
    

def plot_pcmd_residual(pcmd_model, pcmd_compare, like_mode=2, bins=None,
                       ax=None, norm=None, title='', keep_limits=False, im_kwargs={},
                       cbar_kwargs={}):
    """
    Arguments
    ---------

    Returns
    -------
    ax : List of matplotlib axes objects representing image(s)
    loglike : map of log-likelihood plotted
    bins : List of bins used to make Hess image
    norm : matplotlib normalization object

    """
    driv_temp = Driver(None, gpu=False)
    n_bands = pcmd_model.shape[0]
    driv_temp.n_filters = n_bands
    if ax is None:
        fig, ax = plt.subplots(ncols=n_bands-1)
    if n_bands == 2:
        ax = [ax]
    if bins is None:
        combo = np.append(pcmd_model, pcmd_compare, axis=-1)
        mag_bins = [np.min(combo[0]), np.max(combo[0])]
        color_bins = [np.min(combo[1:]), np.max(combo[1:])]
        bins = np.append([mag_bins], [color_bins for _ in range(1, n_bands)])
    driv_temp.initialize_data(pcmd_model, bins=bins)
    if like_mode in [1, 2, 3]:
        loglike = driv_temp.loglike_map(pcmd_compare, like_mode=like_mode)
    else:
        counts_compare, _, _ = make_hess(pcmd_compare, bins)
        loglike = driv_temp.counts_data - counts_compare
    loglike_max = np.max(loglike)
    if norm is None:
        kwargs = {'linthresh': 10.}
        kwargs.update(cbar_kwargs)
        norm = mpl.colors.SymLogNorm(vmin=-loglike_max, vmax=loglike_max,
                                     **kwargs)
    for i, a in enumerate(ax):
        xl = a.get_xlim()
        yl = a.get_ylim()
        plt.subplot(a)
        # record original axis limits, in case overwritten by hist2d
        kwargs = {'cmap': 'bwr'}
        kwargs.update(im_kwargs)
        plt.imshow(loglike[i], norm=norm, origin='lower',
                   aspect='auto', extent=(bins[i+1][0], bins[i+1][-1],
                                          bins[0][0], bins[0][-1]),
                   **kwargs)
        xl += a.get_xlim()
        yl += a.get_ylim()
        if keep_limits:
            a.set_xlim([min(xl), max(xl)])
            a.set_ylim([max(yl), min(yl)])
        a.set_title(title + r' $\chi^2= $' + '{:.2e}'.format(np.sum(loglike)))
    return ax, loglike, bins, norm


def plot_isochrone(iso_model, dmod=30., gal_model=None, axes=None,
                   mag_system=None, update_axes=True, downsample=5, **kwargs):
    if axes is None:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(ncols=(iso_model.num_filters-1), sharey=True)
    if gal_model is None:
        from ..galaxy.galaxy import SSPSimple
        gal_model = SSPSimple(np.array([0., -2., 1., 10.]),
                              dmod=dmod)
    names = iso_model.filter_names
    for age, feh, _, d_mod in gal_model.iter_SSPs():
        mags, _, _ = iso_model.get_isochrone(age, feh, mag_system=mag_system,
                                             downsample=downsample)
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

