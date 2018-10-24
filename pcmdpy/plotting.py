# plotting.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pcmdpy as ppy
from corner import corner
import pandas as pd
from scipy.misc import logsumexp


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
    counts_model, hess_model, err_model = ppy.utils.make_hess(pcmd_model, bins, boundary=False)
    counts_compare, hess_compare, err_compare = ppy.utils.make_hess(pcmd_compare, bins, boundary=False)
    
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
        gal_model = ppy.galaxy.SSPSimple(np.array([0., -2., 1., 10.]),
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

    
class ResultsPlotter(object):

    param_labels = {'logfeh': '[Fe/H]', 'logfeh_mean': '[Fe/H]',
                    'logfeh_std': r'$\sigma([Fe/H])$',
                    'logzh': '[Z/H]', 'logdust': 'log E(B-V)',
                    'logdust_med': 'log E(B-V)',
                    'dust_sig': r'$\sigma(E(B-V))$', 'tau': r'$\tau$',
                    'tau_rise': r'$\tau$', 'dmod': r'$\mu_{d}$',
                    'logNpix': r'$\log_{10} N_{pix}$'}
    param_labels.update({'logSFH{:d}'.format(i):
                         r'$\log_{10}$'+'SFH{:d}'.format(i) for i in range(7)})

    def __init__(self, df_file, true_model=None, prior=None,
                 run_name=None, params=None,
                 labels=None):
        try:
            self.df = pd.read_csv(df_file)
        except UnicodeDecodeError:
            self.df = pd.read_csv(df_file, compression='gzip')

        self.true_model = true_model
        self.run_name = run_name
        self.num_iters = len(self.df)
        self.true_params = None
        self.prior = prior

        if true_model is not None:
            self.params = list(true_model._param_names)
            self.metal_model = true_model.metal_model
            self.dust_model = true_model.dust_model
            self.age_model = true_model.age_model
            self.distance_model = true_model.distance_model
            self.true_params = list(true_model._params)

        else:
            if params is not None:
                self.params = list(params)
            else:
                self.params = []
                for p in self.param_labels.keys():
                    if p in self.df.columns:
                        self.params.append(p)
            if 'logfeh_std' in self.params:
                self.metal_model = ppy.metalmodels.NormMDF()
            elif 'logfeh_mean' in self.params:
                self.metal_model = ppy.metalmodels.FixedWidthNormMDF(0.3)
            else:
                self.metal_model = ppy.metalmodels.SingleFeH()
            if 'dust_sig' in self.params:
                self.dust_model = ppy.dustmodels.LogNormDust()
            elif 'logdust_med' in self.params:
                self.dust_model = ppy.dustmodels.FixedWidthLogNormDust(0.3)
            else:
                self.dust_model = ppy.dustmodels.SingleDust()
            if 'logSFH0' in self.params:
                self.age_model = ppy.agemodels.NonParam()
            elif 'tau' in self.params:
                self.age_model = ppy.agemodels.TauModel()
            elif 'tau_rise' in self.params:
                self.age_model = ppy.agemodels.RisingTau()
            elif 'logage' in self.params:
                self.age_model = ppy.agemodels.SSPModel()
            else:
                self.age_model = ppy.agemodels.ConstantSFR()
            if 'dmod' in self.params:
                self.distance_model = ppy.distancemodels.VariableDistance()
            else:
                self.distance_model = ppy.distancemodels.FixedDistance(30.)

            # set iso_step to be -1
            self.age_model = self.age_model.as_default()

        if type(self.age_model) is ppy.agemodels.NonParam:
            nbins = self.age_model._num_params
            sfhs = 10.**self.df[['logSFH{:d}'.format(i) for i in range(nbins)]]
            self.df['logNpix'] = np.log10(np.sum(sfhs.values, axis=1))
            self.params.append('logNpix')
            if self.true_params is not None:
                self.true_params += [np.log10(true_model.Npix)]
            
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [self.param_labels[p] for p in self.params]
            
        self.n_params = len(self.params)
            
        self.df['log_weights'] = (self.df.logl.values -
                                  logsumexp(self.df.logl.values))
        self.df['weights'] = np.exp(self.df['log_weights'])
        self.df['time_elapsed'] /= 3600.
        try:
            self.df['logfeh'] = self.df.logzh
        except AttributeError:
            pass

    def plot_chains(self, axes=None, burn=0, title=None, dlogz=0.5,
                    show_prior=True, chains_only=False, plot_kwargs=None):
        nr = self.n_params + 3
        if chains_only:
            nr = self.n_params
        if axes is None:
            fig, axes = plt.subplots(nrows=nr, figsize=(8, 2+nr), sharex=True)
        else:
            assert(len(axes) == nr)
        if title is None:
            title = self.run_name
        if plot_kwargs is None:
            plot_kwargs = {}
        else:
            plot_kwargs = dict(plot_kwargs)
        for i, p in enumerate(self.params):
            axes[i].plot(self.df[p].values, **plot_kwargs)
            axes[i].set_ylabel(self.labels[i])
        if not chains_only:
            axes[-3].plot(np.log10(self.df['delta_logz'].values))
            axes[-3].axhline(y=np.log10(dlogz), ls='--', color='r')
            axes[-3].set_ylabel(r'log $\Delta$ln Z')
            axes[-2].plot(self.df['eff'].values)
            axes[-2].set_ylabel('eff (%)')
            axes[-1].plot(self.df['time_elapsed'].values)
            axes[-1].set_ylabel('run time (hrs)')
        axes[-1].set_xlabel('Iteration')

        if self.true_model is not None:
            for i in range(self.n_params):
                axes[i].axhline(y=self.true_params[i], color='r', ls='--')

        if show_prior and (self.prior is not None):
            for i in range(self.n_params):
                axes[i].axhline(y=self.prior.lower_bounds[i], color='k',
                                ls=':')
                axes[i].axhline(y=self.prior.upper_bounds[i], color='k',
                                ls=':')

        if burn > 0:
            for ax in axes:
                ax.axvline(x=burn, ls=':', color='k')

        if title is not None:
            axes[0].set_title(title)
        return axes

    def plot_sfr(self, width=68., ax=None, title=None,
                 burn=0, stop_after=None, show_prior=False, **plot_kwargs):
        assert (0. <= width <= 100.), "width must be between 0 and 100"
        if type(self.age_model) is ppy.agemodels.SSPModel:
            print('Cannot plot cumulative SFH for SSP')
            return
        cols = self.age_model._param_names
        take = slice(burn, stop_after)
        vals = self.df[cols].values[take]
        edges = self.age_model.default_edges
        lookback_gyr = 10.**(edges - 9.)
        logdt = np.log10(np.diff(lookback_gyr * 1e9))
        sfr = np.array([self.age_model.set_params(v).logSFH - logdt
                        for v in vals])
        if self.true_model is not None:
            p_age = self.age_model._num_params
            p_dist = self.distance_model._num_params
            if p_dist > 0:
                vals_true = self.true_model._params[-p_age-p_dist:-p_dist]
            else:
                vals_true = self.true_model._params[-p_age:]
            true_sfr = self.age_model.set_params(vals_true).logSFH - logdt

        if ax is None:
            fig, ax = plt.subplots()
        med = np.percentile(sfr, 50., axis=0)
        upper = np.percentile(sfr, 50. + 0.5*width, axis=0)
        lower = np.percentile(sfr, 50. - 0.5*width, axis=0)
        color = plot_kwargs.pop('color', 'k')
        alpha = plot_kwargs.pop('alpha', 0.3)
        step_plot(lookback_gyr, med, ax=ax, ls='-', color=color, **plot_kwargs)
        step_fill(lookback_gyr, y1=lower, y2=upper, ax=ax,
                  alpha=alpha, color=color,
                  **plot_kwargs)
        if self.true_model is not None:
            step_plot(lookback_gyr, true_sfr, ax=ax, ls='--',
                      color='r', **plot_kwargs)
        ax.set_yscale('linear')
        if title is None:
            ax.set_title(self.run_name)
        else:
            ax.set_title(title)
        ax.set_xlabel('Lookback Time (Gyr)')
        ax.set_ylabel('Log Instantaneous SFR')
        if show_prior:
            if self.prior is None:
                self.plot_sfr(burn=0, stop_after=500, ax=ax, width=width,
                              color='b', alpha=0.1, zorder=-1,
                              show_prior=False, title=title, **plot_kwargs)
            else:
                if p_dist > 0:
                    lower_p = self.prior.lower_bounds[-p_age-p_dist:-p_dist]
                    upper_p = self.prior.upper_bounds[-p_age-p_dist:-p_dist]
                else:
                    lower_p = self.prior.lower_bounds[-p_age:]
                    upper_p = self.prior.upper_bounds[-p_age:]
                lower = self.age_model.set_params(lower_p).logSFH - logdt
                upper = self.age_model.set_params(upper_p).logSFH - logdt
                step_fill(lookback_gyr, y1=lower, y2=upper, ax=ax, alpha=0.1,
                          color='b', zorder=-1, **plot_kwargs)
        return ax
    
    def plot_cum_sfh(self, width=68., ax=None, title=None,
                     burn=0, stop_after=None, show_prior=False, **plot_kwargs):
        assert (0. <= width <= 100.), "width must be between 0 and 100"
        if type(self.age_model) is ppy.agemodels.SSPModel:
            print('Cannot plot cumulative SFH for SSP')
            return
        cols = self.age_model._param_names
        take = slice(burn, stop_after)
        vals = self.df[cols].values[take]
        cum_sfh = np.array([self.age_model.set_params(v).get_cum_sfh()
                            for v in vals])
        if self.true_model is not None:
            p_age = self.age_model._num_params
            p_dist = self.distance_model._num_params
            if p_dist > 0:
                vals_true = self.true_model._params[-p_age-p_dist:-p_dist]
            else:
                vals_true = self.true_model._params[-p_age:]
            true_cum_sfh = self.age_model.set_params(vals_true).get_cum_sfh()
        edges = self.age_model.default_edges
        # time_gyr = 10.**(edges[-1] - 9.) - 10.**(edges - 9.)
        time_gyr = 10.**(edges - 9.)
        if ax is None:
            fig, ax = plt.subplots()
        med = np.percentile(cum_sfh, 50., axis=0)
        upper = np.percentile(cum_sfh, 50. + 0.5*width, axis=0)
        lower = np.percentile(cum_sfh, 50. - 0.5*width, axis=0)
        color = plot_kwargs.pop('color', 'k')
        alpha = plot_kwargs.pop('alpha', 0.3)
        ax.plot(time_gyr, med, ls='-', color=color, **plot_kwargs)
        ax.fill_between(time_gyr, y1=lower, y2=upper, alpha=alpha,
                        color=color, **plot_kwargs)
        if self.true_model is not None:
            ax.plot(time_gyr, true_cum_sfh, ls='--', color='r',
                    **plot_kwargs)
        ax.set_yscale('linear')
        if title is None:
            ax.set_title(self.run_name)
        else:
            ax.set_title(title)
        ax.set_xlabel('Lookback Time (Gyr)')
        ax.set_ylabel('cumulative SFH')
        if show_prior:
            if self.prior is None:
                self.plot_cum_sfh(burn=0, stop_after=500, ax=ax, width=width,
                                  color='b', alpha=0.1, zorder=-1,
                                  show_prior=False, title=title, **plot_kwargs)
            else:
                if p_dist > 0:
                    lower_p = self.prior.lower_bounds[-p_age-p_dist:-p_dist]
                    upper_p = self.prior.upper_bounds[-p_age-p_dist:-p_dist]
                else:
                    lower_p = self.prior.lower_bounds[-p_age:]
                    upper_p = self.prior.upper_bounds[-p_age:]
                lower = self.age_model.set_params(lower_p).get_cum_sfh()
                upper = self.age_model.set_params(upper_p).get_cum_sfh()
                ax.fill_between(time_gyr, y1=lower, y2=upper, alpha=0.1,
                                color='b', zorder=-1, **plot_kwargs)
        return ax

    def plot_corner(self, fig=None, title=None, burn=0, bins=30,
                    smooth_frac=.01, smooth1d=0.,
                    weight=False, full_range=True,
                    show_prior=True, plot_density=False, fill_contours=True,
                    sig_levels=None,
                    **corner_kwargs):
        df_temp = self.df.iloc[burn:]
        vals = df_temp[self.params].values
        smooth = smooth_frac * bins
        if sig_levels is None:
            sig_levels = np.arange(1, 4)
        # convert from sigma to 2d CDF (equivalent of 68-95-99.7 rule)
        levels = 1. - np.exp(-0.5 * sig_levels**2.)
        if full_range:
            lims = []
            for p in self.params:
                lims += [[self.df[p].min(), self.df[p].max()]]
        else:
            lims = None
        if corner_kwargs is None:
            corner_kwargs = {}
        else:
            corner_kwargs = dict(corner_kwargs)
        if weight:
            corner_kwargs['weights'] = df_temp['weights'].values
        else:
            corner_kwargs['weights'] = None
        corner_kwargs.update({'labels': self.labels,
                              'truths': self.true_params, 'fig': fig,
                              'bins': bins, 'smooth': smooth,
                              'plot_density': plot_density,
                              'fill_contours': fill_contours,
                              'levels': levels,
                              'range': lims,
                              'smooth1d': smooth1d})
        fig = corner(vals, **corner_kwargs)
        axes = np.array(fig.get_axes()).reshape(self.n_params, self.n_params)
        if show_prior:
            for i in range(self.n_params):
                a = axes[i, i]
                lower, upper = a.get_ylim()
                y = len(df_temp) / bins
                if weight:
                    y *= np.mean(corner_kwargs['weights'])
                a.axhline(y=y, ls=':')
        if title is None:
            fig.suptitle(self.run_name)
        else:
            fig.suptitle(title)
        return (fig, axes)

    def plot_everything(self, chain_kwargs=None, cum_sfh_kwargs=None,
                        corner_kwargs=None, **all_kwargs):
        if chain_kwargs is None:
            chain_kwargs = {}
        chain_kwargs.update(all_kwargs)
        if cum_sfh_kwargs is None:
            cum_sfh_kwargs = {}
        cum_sfh_kwargs.update(all_kwargs)
        if corner_kwargs is None:
            corner_kwargs = {}
        corner_kwargs.update(all_kwargs)
        chain_axes = self.plot_chains(**chain_kwargs)
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        axes = axes.flatten()
        self.plot_sfr(ax=axes[0], **cum_sfh_kwargs)
        self.plot_cum_sfh(ax=axes[1], **cum_sfh_kwargs)
        corner_fig, corner_axes = self.plot_corner(**corner_kwargs)
        return (chain_axes, axes, (corner_fig, corner_axes))


    def get_chains(self):
        return self.df[self.params]
