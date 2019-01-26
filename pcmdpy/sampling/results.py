import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import pandas as pd
from scipy.special import logsumexp
from statsmodels.stats.weightstats import DescrStatsW
import sys
from datetime import datetime
import time
from ..plotting.plotting import step_plot, step_fill
from ..galaxy.sfhmodels import all_sfh_models, NonParam, SSPModel
from ..galaxy.dustmodels import all_dust_models
from ..galaxy.distancemodels import all_distance_models
from ..galaxy.metalmodels import all_metal_models
from dynesty import utils as dyfunc
from dynesty.results import Results


class ResultsCollector(object):

    def __init__(self, sampler, out_file, verbose=True, print_loc=sys.stdout,
                 out_df=None, live_file=None,
                 save_every=10, param_names=None):
        ndim = sampler.npdim
        self.sampler = sampler
        self.print_loc = print_loc
        self.last_time = time.time()
        self.start_time = time.time()
        self.out_file = out_file
        if out_df is None:
            self.colnames = ['niter', 'nc', 'eff', 'logl', 'logwt',
                             'logvol', 'logz', 'logzerr', 'h', 'delta_logz',
                             'time_elapsed']
            if param_names is not None:
                self.colnames += list(param_names)
            else:
                self.colnames += ['param{:d}'.format(i) for i in range(ndim)]
                self.out_df = pd.DataFrame(columns=self.colnames)
        else:
            assert np.all(np.in1d(param_names, out_df.columns)), (
                "provided parameter names {} not in output Dataframe".format(param_names))
            self.out_df = out_df
        self.live_file = live_file
        self.verbose = verbose
        self.save_every = save_every
        self.param_names = param_names
        
    def collect(self, results, niter, ncall, nbatch=None, dlogz=None,
                logl_max=None, add_live_it=None, stop_val=None,
                logl_min=-np.inf):
        (worst, ustar, vstar, loglstar, logvol,
         logwt, logz, logzvar, h, nc, worst_it,
         boundidx, bounditer, eff, delta_logz) = results
        if delta_logz > 1e6:
            delta_logz = np.inf
        if logzvar >= 0. and logzvar <= 1e6:
            logzerr = np.sqrt(logzvar)
        else:
            logzerr = np.nan
        if logz <= -1e6:
            logz = -np.inf

        last = self.last_time
        self.last_time = time.time()
        dt = self.last_time - last
        total_time = self.last_time - self.start_time
        ave_t = dt/nc
            
        if self.verbose:
            # constructing output
            print_str = 'iter: {:d}'.format(niter)
            if add_live_it is not None:
                print_str += "+{:d}".format(add_live_it)
            print_str += " | "
            if nbatch is not None:
                print_str += "batch: {:d} | ".format(nbatch)
            print_str += "nc: {:d} | ".format(nc)
            print_str += "ncalls: {:d} | ".format(ncall)
            print_str += "bounds: {:d} | ".format(bounditer)
            print_str += "eff(%): {:6.3f} | ".format(eff)
            print_str += "logz: {:.1e} +/- {:.1e} | ".format(logz, logzerr)
            if dlogz is not None:
                print_str += "dlogz: {:6.3f} > {:6.3f}".format(delta_logz,
                                                               dlogz)
            else:
                print_str += "stop: {:6.3f}".format(stop_val)
            print_str += "\n loglike: {:.1e} | ".format(loglstar)
            print_str += "params: {:s}".format(str(vstar))
            print_str += "\n Average call time: {:.2f} sec | ".format(ave_t)
            print_str += "Current time: {:s}".format(str(datetime.now()))
            print_str += '\n --------------------------'

            print(print_str, file=self.print_loc)
            sys.stdout.flush()

        # Saving results to df
        row = {'niter': niter}
        row['time_elapsed'] = total_time
        row['logl'] = loglstar
        row['logvol'] = logvol
        row['logwt'] = logwt
        row['logz'] = logz
        row['h'] = h
        row['eff'] = eff
        row['nc'] = nc
        row['delta_logz'] = delta_logz
        row['logzerr'] = logzerr
        if self.param_names is not None:
            for i, pname in enumerate(self.param_names):
                row[pname] = vstar[i]
        else:
            for i, v in enumerate(vstar):
                row['param{0:d}'.format(i)] = v
        self.out_df = self.out_df.append(row, ignore_index=True)

        # Save current live points
        if ((niter+1) % self.save_every == 0):
            self.flush_to_csv()

    def flush_to_csv(self):
        self.out_df.to_csv(self.out_file, mode='a', index=False,
                           header=False, float_format='%.4e')
        self.out_df.drop(self.out_df.index, inplace=True)
        # track current live points
        if self.live_file is not None:
            live_df = pd.DataFrame(columns=self.param_names,
                                   data=self.sampler.live_v)
            live_df['logl'] = self.sampler.live_logl
            live_df.to_csv(self.live_file, mode='w', index=False,
                           header=True, float_format='%.4e')


class ResultsPlotter(object):
    def __init__(self, df_file, live_file=None, true_model=None,
                 prior=None, run_name=None, dynesty_weights=True):
        try:
            self.df = pd.read_csv(df_file)
        except UnicodeDecodeError:
            self.df = pd.read_csv(df_file, compression='gzip')
        self.df['live'] = False

        if live_file is not None:
            try:
                live_df = pd.read_csv(live_file)
            except FileNotFoundError:
                print('Unable to find live_file: {}. Continuing without '
                      'live points'.format(live_file))
            else:
                # Check if live points have already been added
                n_live = len(live_df)
                if (self.df.nc.tail(n_live) == 1.0).mean() < 0.9:
                    live_df['live'] = True
                    live_df['nc'] = 1
                    live_df['eff'] = self.df['eff'].values[-1]

                    logvols = self.df['logvol'].values[-1]
                    logvols += np.log(1. - (np.arange(n_live)+1.) / (n_live+1.))
                    logvols_pad = np.concatenate(([self.df['logvol'].values[-1]], logvols))
                    logdvols = logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                                         axis=1, b=np.c_[np.ones(n_live),
                                                         -np.ones(n_live)])
                    logdvols += np.log(0.5)
                    dlvs = logvols_pad[:-1] - logvols_pad[1:]
                    logz = self.df['logz'].values[-1]
                    logzvar = self.df['logzerr'].values[-1]**2
                    h = self.df['h'].values[-1]
                    loglstar = self.df['logl'].values[-1]

                    lsort_idx = np.argsort(live_df.logl.values)
                    loglmax = live_df.logl.max()

                    logwts = []
                    hs = []
                    logzs = []
                    logzerrs = []
                    delta_logzs = []
                    for i in range(n_live):
                        idx = lsort_idx[i]
                        logvol, logdvol, dlv = logvols[i], logdvols[i], dlvs[i]
                        loglstar_new = live_df.logl.values[idx]

                        logwt = np.logaddexp(loglstar_new, loglstar) + logdvol
                        logz_new = np.logaddexp(logz, logwt)
                        lzterm = (np.exp(loglstar - logz_new) * loglstar +
                                  np.exp(loglstar_new - logz_new) * loglstar_new)
                        h_new = (np.exp(logdvol) * lzterm +
                                 np.exp(logz - logz_new) * (h + logz) -
                                 logz_new)
                        dh = h_new - h
                        h = h_new
                        logz = logz_new
                        logzvar += dh * dlv
                        loglstar = loglstar_new
                        logz_remain = loglmax + logvol
                        delta_logz = np.logaddexp(logz, logz_remain) - logz

                        logwts.append(logwt)
                        hs.append(h)
                        logzs.append(logz)
                        logzerrs.append(np.sqrt(logzvar))
                        delta_logzs.append(delta_logz)
                    
                    live_df.sort_values('logl', inplace=True)
                    live_df['logwt'] = logwts
                    live_df['logvol'] = logvols
                    live_df['h'] = hs
                    live_df['logz'] = logzs
                    live_df['logzerr'] = logzerrs
                    live_df['delta_logz'] = delta_logzs
                    live_df['niter'] = np.arange(n_live) + self.df['niter'].max()
                    self.df = self.df.append(live_df, ignore_index=True,
                                             sort=False)

        self.true_model = true_model
        self.run_name = run_name
        self.n_iter = len(self.df)
        self.n_live = self.df['live'].sum()
        self.n_dead = self.n_iter - self.n_live
        self.true_params = None
        self.prior = prior

        if true_model is not None:
            self.metal_model = true_model.metal_model
            self.dust_model = true_model.dust_model
            self.sfh_model = true_model.sfh_model
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
            
            # Identify the SFH model from parameters found
            self.sfh_model = None
            for sfhm in all_sfh_models:
                params = sfhm._param_names
                if isinstance(params, property):
                    params = sfhm()._param_names
                if np.all(np.in1d(params, cols)):
                    self.sfh_model = sfhm()
                    break
            if self.sfh_model is None:
                raise ValueError(
                    'params found to not match a known sfh model:\n'
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
            # self.sfh_model = self.sfh_model.as_default()

        self.params, self.labels = [], []
        for m in [self.metal_model, self.dust_model, self.sfh_model,
                  self.distance_model]:
            self.params.extend(m._param_names)
            self.labels.extend(m._fancy_names)
            
        if isinstance(self.sfh_model, NonParam):
            nbins = self.sfh_model._num_params
            sfhs = 10.**self.df[['logSFH{:d}'.format(i) for i in range(nbins)]]
            self.df['logNpix'] = np.log10(np.sum(sfhs.values, axis=1))
            self.params.append('logNpix')
            self.labels.append(r'$\log_{10} N_\mathrm{pix}$')
            if self.true_params is not None:
                self.true_params += [np.log10(true_model.sfh_model.Npix)]

        self.n_params = len(self.params)

        # weights defined by Dynesty
        self.df['log_weights'] = (self.df.logwt.values -
                                  logsumexp(self.df.logwt.values))
        self.df['dynesty_weights'] = np.exp(self.df['log_weights'])
        # weights purely from log likelihoods
        logl_ws = (self.df.logl.values -
                   logsumexp(self.df.logl.values))
        self.df['likelihood_weights'] = np.exp(logl_ws)

        if dynesty_weights:
            self.df['weights'] = self.df['dynesty_weights']
        else:
            self.df['weights'] = self.df['likelihood_weights']

        self.df['time_elapsed'] /= 3600.
        try:
            self.df['logfeh'] = self.df.logzh
        except AttributeError:
            pass

        results = [
            ('nlive', self.n_live or 500),
            ('niter', self.df.niter.values.max()),
            ('ncall', self.df.nc.values),
            ('eff', self.df.eff.values[-1]),
            ('samples', self.samples),
            ('logwt', self.df.logwt.values),
            ('logl', self.df.logl.values),
            ('logvol', self.df.logvol.values),
            ('logz', self.df.logz.values),
            ('logzerr', self.df.logzerr.values),
            ('information', self.df.h.values)]

        self.dynesty_results = Results(results)

    @property
    def samples(self):
        return self.get_chains().values

    @property
    def weights(self):
        return self.df['weights'].values
    
    def get_chains(self):
        return self.df[self.params]

    def means(self, burn=0, trim=0):
        if trim > 0:
            samples = self.samples[burn:-trim].copy()
            weights = self.weights[burn:-trim].copy()
        else:
            samples = self.samples[burn:].copy()
            weights = self.weights[burn:].copy()
        weights /= weights.sum()
        means, _ = dyfunc.mean_and_cov(samples, weights)
        return means

    def cov(self, burn=0, trim=0):
        if trim > 0:
            samples = self.samples[burn:-trim].copy()
            weights = self.weights[burn:-trim].copy()
        else:
            samples = self.samples[burn:].copy()
            weights = self.weights[burn:].copy()
        weights /= weights.sum()
        _, cov = dyfunc.mean_and_cov(samples, weights)
        return cov

    def stds(self, burn=0, trim=0):
        cov = self.cov(burn=burn, trim=trim)
        return np.sqrt([cov[i, i] for i in range(self.n_params)])
    
    @property
    def best_params(self):
        if isinstance(self.sfh_model, NonParam):
            return self.df.tail(1)[self.params[:-1]].values[0]
        else:
            return self.df.tail(1)[self.params].values[0]

    @property
    def best_model(self):
        from ..galaxy.galaxy import CustomGalaxy
        gal = CustomGalaxy(self.metal_model, self.dust_model, self.sfh_model,
                           self.distance_model)
        gal.set_params(self.best_params)
        return gal

    def plot_chains(self, axes=None, burn=0, title=None, dlogz=0.5,
                    include_live=True, show_prior=True, chains_only=False,
                    plot_kwargs=None):
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
        is_live = self.df['live'].values
        is_dead = ~is_live
        xs_live = np.arange(self.n_live) + self.n_dead
        live_plot_kwargs = plot_kwargs.copy()
        live_plot_kwargs.update({'color': 'c',
                                 'ls': ':'})
        for i, p in enumerate(self.params):
            axes[i].plot(self.df[p].values[is_dead],
                         **plot_kwargs)
            if include_live:
                axes[i].plot(xs_live, self.df[p].values[is_live],
                             **live_plot_kwargs)
            axes[i].set_ylabel(self.labels[i])
        if not chains_only:
            axes[-3].plot(np.log10(self.df['delta_logz'].values[is_dead]),
                          **plot_kwargs)
            axes[-3].axhline(y=np.log10(dlogz), ls='--', color='r')
            axes[-3].set_ylabel(r'log $\Delta$ln Z')
            axes[-2].plot(self.df['eff'].values[is_dead],
                          **plot_kwargs)
            axes[-2].set_ylabel('eff (%)')
            axes[-1].plot(self.df['time_elapsed'].values[is_dead],
                          **plot_kwargs)
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
        if isinstance(self.sfh_model, SSPModel):
            print('Cannot plot cumulative SFH for SSP')
            return
        cols = self.sfh_model._param_names
        take = slice(burn, stop_after)
        vals = self.df[cols].values[take]
        edges = self.sfh_model.default_edges
        lookback_gyr = 10.**(edges - 9.)
        logdt = np.log10(np.diff(lookback_gyr * 1e9))
        sfr = np.array([self.sfh_model.set_params(v).logSFH - logdt
                        for v in vals])
        if self.true_model is not None:
            p_sfh = self.sfh_model._num_params
            p_dist = self.distance_model._num_params
            if p_dist > 0:
                vals_true = self.true_model._params[-p_sfh-p_dist:-p_dist]
            else:
                vals_true = self.true_model._params[-p_sfh:]
            true_sfr = self.sfh_model.set_params(vals_true).logSFH - logdt

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
                    lower_p = self.prior.lower_bounds[-p_sfh-p_dist:-p_dist]
                    upper_p = self.prior.upper_bounds[-p_sfh-p_dist:-p_dist]
                else:
                    lower_p = self.prior.lower_bounds[-p_sfh:]
                    upper_p = self.prior.upper_bounds[-p_sfh:]
                lower = self.sfh_model.set_params(lower_p).logSFH - logdt
                upper = self.sfh_model.set_params(upper_p).logSFH - logdt
                step_fill(lookback_gyr, y1=lower, y2=upper, ax=ax, alpha=0.1,
                          color='b', zorder=-1, **plot_kwargs)
        return ax
    
    def plot_cum_sfh(self, width=68., ax=None, title=None,
                     burn=0, stop_after=None, show_prior=False, **plot_kwargs):
        assert (0. <= width <= 100.), "width must be between 0 and 100"
        if isinstance(self.sfh_model, SSPModel):
            print('Cannot plot cumulative SFH for SSP')
            return
        cols = self.sfh_model._param_names
        take = slice(burn, stop_after)
        vals = self.df[cols].values[take]
        cum_sfh = np.array([self.sfh_model.set_params(v).get_cum_sfh()
                            for v in vals])
        if self.true_model is not None:
            p_sfh = self.sfh_model._num_params
            p_dist = self.distance_model._num_params
            if p_dist > 0:
                vals_true = self.true_model._params[-p_sfh-p_dist:-p_dist]
            else:
                vals_true = self.true_model._params[-p_sfh:]
            true_cum_sfh = self.sfh_model.set_params(vals_true).get_cum_sfh()
        edges = self.sfh_model.default_edges
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
                    lower_p = self.prior.lower_bounds[-p_sfh-p_dist:-p_dist]
                    upper_p = self.prior.upper_bounds[-p_sfh-p_dist:-p_dist]
                else:
                    lower_p = self.prior.lower_bounds[-p_sfh:]
                    upper_p = self.prior.upper_bounds[-p_sfh:]
                lower = self.sfh_model.set_params(lower_p).get_cum_sfh()
                upper = self.sfh_model.set_params(upper_p).get_cum_sfh()
                ax.fill_between(time_gyr, y1=lower, y2=upper, alpha=0.1,
                                color='b', zorder=-1, **plot_kwargs)
        return ax

    def plot_corner(self, fig=None, title=None, burn=0, trim=0, bins=30,
                    include_live=True, smooth_frac=.01, smooth1d=0.,
                    weight=False, full_range=False,
                    show_prior=False, plot_density=False, fill_contours=True,
                    sig_levels=None, plot_datapoints=True, show_truth=True,
                    **corner_kwargs):
        if trim > 0:
            df_temp = self.df.iloc[burn:-trim]
        else:
            df_temp = self.df.iloc[burn:]
        if not include_live:
            df_temp = df_temp[~df_temp['live']]
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
        truths = self.true_params if show_truth else None
        corner_kwargs.update({'labels': self.labels,
                              'truths': truths, 'fig': fig,
                              'bins': bins, 'smooth': smooth,
                              'plot_density': plot_density,
                              'fill_contours': fill_contours,
                              'levels': levels,
                              'range': lims,
                              'smooth1d': smooth1d,
                              'plot_datapoints': plot_datapoints})
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
        
