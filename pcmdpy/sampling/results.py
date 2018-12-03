import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import pandas as pd
from scipy.misc import logsumexp
from ..utils.plotting import step_plot, step_fill
from ..galaxy.agemodels import all_age_models, NonParam, SSPModel
from ..galaxy.dustmodels import all_dust_models
from ..galaxy.distancemodels import all_distance_models
from ..galaxy.metalmodels import all_metal_models


class ResultsPlotter(object):
    def __init__(self, df_file, true_model=None, prior=None,
                 run_name=None):
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
            self.metal_model = true_model.metal_model
            self.dust_model = true_model.dust_model
            self.age_model = true_model.age_model
            self.distance_model = true_model.distance_model
            self.true_params = list(true_model._params)

        else:
            cols = self.df.columns
            # Identify the metal model from parameters found
            self.metal_model = None
            for mm in all_metal_models:
                if np.all(np.in1d(mm._param_names, cols)):
                    self.metal_model = mm()
                    break
            if self.metal_model is None:
                raise ValueError(
                    'params found to not match a known metal model:\n'
                    '{}'.format(cols))

            # Identify the dust model from parameters found
            self.dust_model = None
            for dm in all_dust_models:
                if np.all(np.in1d(dm._param_names, cols)):
                    self.dust_model = dm()
                    break
            if self.dust_model is None:
                raise ValueError(
                    'params found to not match a known dust model:\n'
                    '{}'.format(cols))
            
            # Identify the age model from parameters found
            self.age_model = None
            for am in all_age_models:
                params = am._param_names
                if isinstance(params, property):
                    params = am()._param_names
                if np.all(np.in1d(params, cols)):
                    self.age_model = am()
                    break
            if self.age_model is None:
                raise ValueError(
                    'params found to not match a known age model:\n'
                    '{}'.format(cols))
            
            # Identify the distance model from parameters found
            self.distance_model = None
            for dm in all_distance_models:
                if np.all(np.in1d(dm._param_names, cols)):
                    self.distance_model = dm()
                    break
            if self.distance_model is None:
                raise ValueError(
                    'params found to not match a known distance model:\n'
                    '{}'.format(cols))

            # set iso_step to be -1
            # self.age_model = self.age_model.as_default()

        self.params, self.labels = [], []
        for m in [self.metal_model, self.dust_model, self.age_model,
                  self.distance_model]:
            self.params.extend(m._param_names)
            self.labels.extend(m._fancy_names)
            
        if isinstance(self.age_model, NonParam):
            nbins = self.age_model._num_params
            sfhs = 10.**self.df[['logSFH{:d}'.format(i) for i in range(nbins)]]
            self.df['logNpix'] = np.log10(np.sum(sfhs.values, axis=1))
            self.params.append('logNpix')
            self.labels.append(r'$\log_{10} N_\mathrm{pix}$')
            if self.true_params is not None:
                self.true_params += [np.log10(true_model.age_model.Npix)]
            
        self.n_params = len(self.params)
            
        self.df['log_weights'] = (self.df.logl.values -
                                  logsumexp(self.df.logl.values))
        self.df['weights'] = np.exp(self.df['log_weights'])
        self.df['time_elapsed'] /= 3600.
        try:
            self.df['logfeh'] = self.df.logzh
        except AttributeError:
            pass

    @property
    def best_params(self):
        if isinstance(self.age_model, NonParam):
            return self.df.tail(1)[self.params[:-1]].values[0]
        else:
            return self.df.tail(1)[self.params].values[0]

    @property
    def best_model(self):
        from ..galaxy.galaxy import CustomGalaxy
        gal = CustomGalaxy(self.metal_model, self.dust_model, self.age_model,
                           self.distance_model)
        gal.set_params(self.best_params)
        return gal

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
        if isinstance(self.age_model, SSPModel):
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
        if isinstance(self.age_model, SSPModel):
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
