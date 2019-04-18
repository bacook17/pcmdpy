import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import expanduser
from scipy.special import logsumexp
from ..plotting.plotting import step_plot, step_fill
from ..galaxy.sfhmodels import all_sfh_models, NonParam, SSPModel
from ..galaxy.dustmodels import all_dust_models
from ..galaxy.distancemodels import all_distance_models
from ..galaxy.metalmodels import all_metal_models
from ..galaxy.galaxy import CustomGalaxy
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from dynesty.results import Results
from dynesty.dynamicsampler import DynamicSampler


class ResultsPlotter(object):
    def __init__(self, df_file, max_logl=None, live_file=None, gal_model=None,
                 model_is_truth=False, run_name=None, dmod_true=None):
        self.df = pd.read_csv(df_file, comment='#')
        with open(expanduser(df_file), 'r') as f:
            line = f.readline().strip()
        if 'max_logl' in line:
            self._max_logl = float(line.split(': ')[-1])
        elif max_logl is not None:
            self._max_logl = max_logl
        else:
            self._max_logl = None
        self.df['live'] = False
        self.live_included = False

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
                    live_df['niter'] = np.arange(n_live) + self.df['niter'].max() + 1
                    self.df = self.df.append(live_df, ignore_index=True,
                                             sort=False)
        if self.max_logl is None:
            self._max_logl = self.df.logl.values[-10]

        self.gal_model = gal_model
        self.true_model = None
        self.run_name = run_name
        self.n_iter = len(self.df)
        self.n_live = self.df['live'].sum()
        self.n_dead = self.n_iter - self.n_live
        self.true_params = None
        
        if gal_model is not None:
            if model_is_truth:
                self.true_model = self.gal_model
                self.true_params = np.array(self.true_model._params)
                self.dmod_true = self.true_model.dmod

        else:  # If no model provided, must guess the model used
            cols = self.df.columns
            # Identify the metal model from parameters found
            metal_model = None
            for mm in all_metal_models:
                if np.all(np.in1d(mm._param_names, cols)):
                    metal_model = mm()
                    break
            if metal_model is None:
                raise ValueError(
                    'params found to not match a known metal model:\n'
                    '{}'.format(cols))

            # Identify the dust model from parameters found
            dust_model = None
            for dm in all_dust_models:
                if np.all(np.in1d(dm._param_names, cols)):
                    dust_model = dm()
                    break
            if dust_model is None:
                raise ValueError(
                    'params found to not match a known dust model:\n'
                    '{}'.format(cols))
            
            # Identify the SFH model from parameters found
            sfh_model = None
            for sfhm in all_sfh_models:
                params = sfhm._param_names
                if isinstance(params, property):
                    params = sfhm()._param_names
                if np.all(np.in1d(params, cols)):
                    sfh_model = sfhm()
                    break
            if sfh_model is None:
                raise ValueError(
                    'params found to not match a known sfh model:\n'
                    '{}'.format(cols))
            
            # Identify the distance model from parameters found
            distance_model = None
            for dm in all_distance_models:
                if np.all(np.in1d(dm._param_names, cols)):
                    distance_model = dm()
                    break
            if distance_model is None:
                raise ValueError(
                    'params found to not match a known distance model:\n'
                    '{}'.format(cols))

            self.gal_model = CustomGalaxy(
                metal_model, dust_model, sfh_model, distance_model)

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
            self.labels.append(r'$\log N_\mathrm{pix}$')
            if self.true_params is not None:
                self.true_params = np.append(self.true_params,
                                             self.true_model.sfh_model.logNpix)

        self.n_params = len(self.params)
        if (self.true_params is None) and (dmod_true is not None):
            try:
                dmod_index = self.params.index('dmod')
                self.true_params = [None]*self.n_params
                self.true_params[dmod_index] = dmod_true
            except ValueError:
                pass

        # weights defined by Dynesty
        self.df['log_weights'] = (self.df.logwt.values -
                                  logsumexp(self.df.logwt.values))
        self.df['dynesty_weights'] = np.exp(self.df['log_weights'])
        # weights purely from log likelihoods
        logl_ws = (self.df.logl.values -
                   logsumexp(self.df.logl.values))
        self.df['likelihood_weights'] = np.exp(logl_ws)

        self.df['weights'] = self.df['dynesty_weights']

        self.df['time_elapsed'] /= 3600.
        try:
            self.df['logfeh'] = self.df.logzh
        except AttributeError:
            pass
        self._dynesty = self.as_dynesty()
        self._equal_samples = self.get_equal_samples()

    @property
    def max_logl(self):
        return self._max_logl

    @max_logl.setter
    def max_logl(self, max_logl):
        self._max_logl = max_logl
        self._dynesty = self.as_dynesty()
        self._equal_samples = self.get_equal_samples()
        
    @property
    def metal_model(self):
        return self.gal_model.metal_model

    @property
    def dust_model(self):
        return self.gal_model.dust_model

    @property
    def sfh_model(self):
        return self.gal_model.sfh_model

    @property
    def distance_model(self):
        return self.gal_model.distance_model

    @property
    def dynesty(self):
        return self._dynesty

    def as_dynesty(self, burn=0, trim=0, max_logl=None):
        if max_logl is None:
            max_logl = self.max_logl
        if trim > 0:
            sub_df = self.df.iloc[burn:-trim]
            samples = self.get_chains().values[burn:-trim]
        else:
            sub_df = self.df.iloc[burn:]
            samples = self.get_chains().values[burn:]
        logwt = sub_df.logwt.values
        logwt -= logsumexp(logwt)
        logwt += sub_df.logz.values[-1]
        results = [
            ('nlive', 0),
            ('niter', len(sub_df)),
            ('ncall', sub_df.nc.values.astype(int)),
            ('eff', sub_df.eff.values[-1]),
            ('samples', samples),
            ('logwt', logwt),
            ('logl', sub_df.logl.values),
            ('logvol', sub_df.logvol.values),
            ('logz', sub_df.logz.values),
            ('logzerr', sub_df.logzerr.values),
            ('information', sub_df.h.values),
            ('delta_logz', sub_df.delta_logz.values)]

        results = Results(results)
        if max_logl is not None:
            new_logl = np.array(sub_df.logl.values)
            new_logl[new_logl >= max_logl] = max_logl
            results = dyfunc.reweight_run(results,
                                          logp_new=new_logl)
            logz_remain = results['logvol'] + max_logl
            results['delta_logz'] = np.logaddexp(results['logz'], logz_remain) - results['logz']
        return results

    def get_samples(self, burn=0, trim=0, max_logl=None):
        if max_logl is None:
            max_logl = self.max_logl
        results = self.as_dynesty(burn=burn, trim=trim)
        return results['samples']

    def get_weights(self, burn=0, trim=0, max_logl=None):
        if max_logl is None:
            max_logl = self.max_logl
        results = self.as_dynesty(burn=burn, trim=trim,
                                  max_logl=max_logl)
        return np.exp(results['logwt'] - logsumexp(results['logwt']))

    @property
    def samples(self):
        return self.dynesty['samples']

    @property
    def weights(self):
        results = self.dynesty
        return np.exp(results['logwt'] - logsumexp(results['logwt']))

    def get_equal_samples(self, burn=0, trim=0, max_logl=None):
        if max_logl is None:
            max_logl = self.max_logl
        results = self.as_dynesty(burn=burn, trim=trim,
                                  max_logl=max_logl)
        samples = results['samples']
        weights = np.exp(results['logwt'] - logsumexp(results['logwt']))
        return dyfunc.resample_equal(samples, weights)

    @property
    def equal_samples(self):
        return self._equal_samples

    def get_chains(self):
        return self.df[self.params]

    @property
    def means(self):
        return dyfunc.mean_and_cov(self.samples, self.weights)[0]
    
    def get_means(self, burn=0, trim=0, max_logl=None):
        if max_logl is None:
            max_logl = self.max_logl
        kwargs = {'burn': burn,
                  'trim': trim,
                  'max_logl': max_logl}
        samples = self.get_samples(**kwargs)
        weights = self.get_weights(**kwargs)
        means, _ = dyfunc.mean_and_cov(samples, weights)
        return means

    @property
    def cov(self):
        return dyfunc.mean_and_cov(self.samples, self.weights)[1]

    def get_cov(self, burn=0, trim=0, max_logl=None):
        if max_logl is None:
            max_logl = self.max_logl
        kwargs = {'burn': burn,
                  'trim': trim,
                  'max_logl': max_logl}
        samples = self.get_samples(**kwargs)
        weights = self.get_weights(**kwargs)
        _, cov = dyfunc.mean_and_cov(samples, weights)
        return cov

    @property
    def stds(self):
        cov = self.cov
        return np.sqrt([cov[i, i] for i in range(self.n_params)])
    
    def get_stds(self, burn=0, trim=0, max_logl=None):
        if max_logl is None:
            max_logl = self.max_logl
        cov = self.get_cov(burn=burn, trim=trim, max_logl=max_logl)
        return np.sqrt([cov[i, i] for i in range(self.n_params)])

    @property
    def medians(self):
        samples = self.equal_samples
        return np.percentile(samples, 50., axis=0)
    
    def get_medians(self, burn=0, trim=0, max_logl=None):
        if max_logl is None:
            max_logl = self.max_logl
        kwargs = {'burn': burn,
                  'trim': trim,
                  'max_logl': max_logl}
        samples = self.get_equal_samples(**kwargs)
        return np.percentile(samples, 50., axis=0)

    @property
    def lower_upper(self):
        samples = self.equal_samples
        upper = np.percentile(samples, 84., axis=0)
        lower = np.percentile(samples, 16., axis=0)
        return lower, upper

    def get_lower_upper(self, burn=0, trim=0, max_logl=None):
        if max_logl is None:
            max_logl = self.max_logl
        kwargs = {'burn': burn,
                  'trim': trim,
                  'max_logl': max_logl}
        samples = self.get_equal_samples(**kwargs)
        upper = np.percentile(samples, 84., axis=0)
        lower = np.percentile(samples, 16., axis=0)
        return lower, upper

    @property
    def best_params(self):
        if isinstance(self.sfh_model, NonParam):
            return self.df.tail(1)[self.params[:-1]].values[0]
        else:
            return self.df.tail(1)[self.params].values[0]

    @property
    def best_model(self):
        gal = self.gal_model.copy()
        gal.set_params(self.best_params)
        return gal

    def plot_trace(self, axes=None, burn=0, trim=0, max_logl=None, smooth=0.02,
                   show_truth=True, full_range=False, tight=True,
                   **traceplot_kwargs):
        """
        
        Returns
        -------
        fig, axes
        """
        if max_logl is None:
            max_logl = self.max_logl
        dynesty_kwargs = {'burn': burn,
                          'trim': trim,
                          'max_logl': max_logl}
        results = self.as_dynesty(**dynesty_kwargs)
        kwargs = {'labels': self.labels,
                  'smooth': smooth,
                  'truths': self.true_params if show_truth else None,
                  'fig': None,
                  'span': None,
                  'show_titles': True}
        if full_range:
            kwargs['span'] = [[results['samples'][:, i].min(),
                               results['samples'][:, i].max()] for i in range(self.n_params)]
        else:
            means = self.get_means(**dynesty_kwargs)
            stds = self.get_stds(**dynesty_kwargs)
            kwargs['span'] = [[means[i] - max(5*stds[i], 1e-3),
                               means[i] + max(5*stds[i], 1e-3)] for i in range(self.n_params)]
        kwargs.update(traceplot_kwargs)
        if (axes is not None) and (axes.shape == (self.n_params, 2)):
            kwargs['fig'] = (axes.flatten()[0].get_figure(), axes)
        fig, axes = dyplot.traceplot(results, **kwargs)
        if tight:
            fig.tight_layout();
        return fig, axes

    def plot_chains(self, axes=None, burn=0, title=None, dlogz=0.5,
                    include_live=True, show_prior=False, chains_only=False,
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

    def plot_corner(self, burn=0, trim=0, max_logl=None, axes=None,
                    show_titles=True, smooth=0.03, density_smooth=0.05,
                    show_truth=True, full_range=False, sig_levels=[1, 2, 3],
                    filled=True, plot_many=False, **corner_kwargs):
        """
        
        Returns
        -------
        fig, axes
        """
        if max_logl is None:
            max_logl = self.max_logl
        dynesty_kwargs = {'burn': burn,
                          'trim': trim,
                          'max_logl': max_logl}
        results = self.as_dynesty(**dynesty_kwargs)
        color = plt.rcParams.get('lines.color', 'k')
        kwargs = {'labels': self.labels,
                  'smooth': smooth,
                  'density_smooth': density_smooth,
                  'truths': self.true_params if show_truth else None,
                  'fig': None,
                  'span': None,
                  'show_titles': show_titles,
                  'color': color}
        levels = 1.0 - np.exp(-0.5 * np.array(sig_levels)**2)
        kwargs['hist_kwargs'] = corner_kwargs.pop('hist_kwargs', {})
        kwargs['hist2d_kwargs'] = corner_kwargs.pop('hist2d_kwargs', {})
        kwargs['hist2d_kwargs']['levels'] = levels
        kwargs['hist2d_kwargs']['plot_datapoints'] = kwargs['hist2d_kwargs'].pop('plot_datapoints', False)
        kwargs['hist2d_kwargs']['fill_contours'] = kwargs['hist2d_kwargs'].pop('fill_contours', filled)
        kwargs['hist2d_kwargs']['plot_density'] = kwargs['hist2d_kwargs'].pop('plot_density', not filled)
        kwargs['hist2d_kwargs']['no_fill_contours'] = kwargs['hist2d_kwargs'].pop('no_fill_contours', True)
        if plot_many:
            kwargs['hist2d_kwargs']['fill_contours'] = False
            kwargs['hist2d_kwargs']['plot_density'] = False
            kwargs['hist2d_kwargs']['no_fill_contours'] = True
            kwargs['hist2d_kwargs']['plot_datapoints'] = False
            kwargs['hist_kwargs']['histtype'] = 'step'
        datamins = [results['samples'][:, i].min() for i in range(self.n_params)]
        datamaxs = [results['samples'][:, i].max() for i in range(self.n_params)]
        means = self.get_means(**dynesty_kwargs)
        stds = self.get_stds(**dynesty_kwargs)
        postmins = [means[i] - max(5*stds[i], 1e-1) for i in range(self.n_params)]
        postmaxs = [means[i] + max(5*stds[i], 1e-1) for i in range(self.n_params)]

        if full_range:
            kwargs['span'] = [[a, b] for a,b in zip(datamins, datamaxs)]
        else:
            kwargs['span'] = [[max(a,c), min(b,d)] for a,b,c,d in zip(datamins, datamaxs,
                                                                      postmins, postmaxs)]
        kwargs.update(corner_kwargs)
        if (axes is not None) and (axes.shape == (self.n_params,
                                                  self.n_params)):
            kwargs['fig'] = (axes.flatten()[0].get_figure(), axes)
        fig, axes = dyplot.cornerplot(results, **kwargs)
        return fig, axes

    def plot_sfr(self, width=68., ax=None, title=False, all_ages=False,
                 burn=0, trim=0, max_logl=None, show_truth=True,
                 true_model=None, show_prior=True, legend=True,
                 color=None, show_bars=False, dark=False,
                 line_kwargs={}, error_kwargs={}, fill_kwargs={},
                 truth_kwargs={}, prior_kwargs={}):
        if max_logl is None:
            max_logl = self.max_logl
        assert (0. <= width <= 100.), "width must be between 0 and 100"
        if isinstance(self.sfh_model, SSPModel):
            print('Cannot plot cumulative SFH for SSP')
            return
        sfh_indices = np.array([self.params.index(k) for k in self.sfh_model._param_names])
        if (burn == 0) and (trim == 0) and (max_logl is None):
            samples = self.get_samples()
        elif np.isclose(width, 100.) and (show_prior is False):
            samples = self.get_samples()
        else:
            samples = self.get_equal_samples(burn=burn, trim=trim,
                                             max_logl=max_logl)
        model = self.sfh_model.copy()
        if not all_ages:
            model = model.as_NonParam()
        denom = model.delta_ts * 1e9 / self.gal_model.meanmass
        SFHs = []
        for i in range(len(samples)):
            model.set_params(samples[i, sfh_indices])
            SFHs.append(model.SFH)
        SFRs = np.array(SFHs) / denom
        ages = model.ages
        if ax is None:
            fig, ax = plt.subplots()
        med = np.percentile(SFRs, 50., axis=0)
        upper = np.percentile(SFRs, 50. + 0.5*width, axis=0)
        lower = np.percentile(SFRs, 50. - 0.5*width, axis=0)
        color = color or plt.rcParams.get('lines.color', 'k')
        kwargs = {'color': color,
                  'alpha': 0.6 if dark else 0.3,
                  'linewidth': 0}
        if fill_kwargs.pop('no_fill', False):
            kwargs.pop('color')
        kwargs.update(fill_kwargs)
        fill = ax.fill_between(x=ages, y1=lower, y2=upper, **kwargs)
        kwargs = {'color': color,
                  'capsize': 10,
                  'marker': '',
                  'ms': 8,
                  'alpha': (1.0 if show_bars else 0.0),
                  'ls': ''}
        kwargs.update(error_kwargs)
        error = ax.errorbar(x=ages, y=med, yerr=[med-lower, upper-med], **kwargs)
        kwargs = {'color': color,
                  'ls': '-',
                  'marker': 'o',
                  'ms': 8}
        kwargs.update(line_kwargs)
        line, = ax.plot(ages, med, **kwargs)
        truth = None
        if show_truth:
            model = true_model or self.true_model
            if model is not None:
                if isinstance(model, CustomGalaxy):
                    model = model.sfh_model
                if not all_ages:
                    model = model.as_NonParam()
                ages = model.ages
                true_sfh = model.SFH
                true_sfr = true_sfh / denom
                kwargs = {'color': 'r',
                          'ls': '-',
                          'lw': 2,
                          'zorder': 10}
                kwargs.update(truth_kwargs)
                truth, = ax.plot(ages, true_sfr, **kwargs)
        prior = None
        if show_prior:
            kwargs = {'width': 100.,
                      'all_ages': all_ages,
                      'burn': 0,
                      'trim': 0,
                      'max_logl': None,
                      'show_truth': False,
                      'title': False,
                      'legend': False,
                      'line_kwargs': {'alpha': 0.},
                      'error_kwargs': {'alpha': 0.},
                      'fill_kwargs': {'alpha': 1.0,
                                      'no_fill': True,
                                      'facecolor': 'k' if dark else 'w',
                                      'zorder': -10,
                                      'linestyle': ':',
                                      'linewidth': 2,
                                      'edgecolors': 'w' if dark else 'k'}}
            kwargs.update(prior_kwargs)
            _, lines = self.plot_sfr(show_prior=False, ax=ax, **kwargs)
            prior = lines[2]
        ax.set_yscale('log')
        if title is None:
            ax.set_title(self.run_name)
        elif title is not False:
            ax.set_title(title)
        ax.set_xlabel('Log age (yr)')
        ax.set_ylabel(r'SFR ($\mathrm{M_{\star}\;yr^{-1}\;pix^{-1}}$)')
        if legend:
            ax.legend(((line, fill), prior, truth), ('Posterior', 'Prior', 'Truth'), loc=0)
        return ax, (line, error, fill, truth, prior)
    
    def plot_cum_sfh(self, width=68., ax=None, title=False, all_ages=False,
                     burn=0, trim=0, max_logl=None, legend=True,
                     bulk_norm=False,
                     show_truth=True, true_model=None, show_prior=True,
                     color=None, show_bars=False,
                     dark=False, 
                     line_kwargs={}, error_kwargs={}, fill_kwargs={},
                     truth_kwargs={}, prior_kwargs={}):
        if max_logl is None:
            max_logl = self.max_logl
        assert (0. <= width <= 100.), "width must be between 0 and 100"
        if isinstance(self.sfh_model, SSPModel):
            print('Cannot plot cumulative SFH for SSP')
            return
        sfh_indices = np.array([self.params.index(k) for k in self.sfh_model._param_names])
        if (burn == 0) and (trim == 0) and (max_logl is None):
            samples = self.get_samples()
        elif np.isclose(width, 100.) and (show_prior is False):
            samples = self.get_samples()
        else:
            samples = self.get_equal_samples(burn=burn, trim=trim,
                                             max_logl=max_logl)
        model = self.sfh_model.copy()
        if not all_ages:
            model = model.as_NonParam()
        ages = model.ages
        SFHs = []
        for i in range(len(samples)):
            model.set_params(samples[i, sfh_indices])
            SFHs.append(model.SFH)
        cum_sfhs = np.cumsum(SFHs, axis=1)
        if bulk_norm:
            cum_sfhs /= np.mean(cum_sfhs[:, -1])  # normalize to mean 1
        else:
            cum_sfhs = (cum_sfhs.T / cum_sfhs[:, -1]).T
        if ax is None:
            fig, ax = plt.subplots()
        med = np.percentile(cum_sfhs, 50., axis=0)
        upper = np.percentile(cum_sfhs, 50. + 0.5*width, axis=0)
        lower = np.percentile(cum_sfhs, 50. - 0.5*width, axis=0)
        color = color or plt.rcParams.get('lines.color', 'k')
        kwargs = {'color': color,
                  'alpha': 0.6 if dark else 0.3,
                  'linewidth': 0}
        if fill_kwargs.pop('no_fill', False):
            kwargs.pop('color')
        kwargs.update(fill_kwargs)
        fill = ax.fill_between(x=ages, y1=lower, y2=upper, **kwargs)
        kwargs = {'color': color,
                  'capsize': 10,
                  'marker': '',
                  'alpha': (1.0 if show_bars else 0.0),
                  'ms': 8,
                  'ls': ''}
        kwargs.update(error_kwargs)
        error = ax.errorbar(x=ages, y=med, yerr=[med-lower, upper-med], **kwargs)
        kwargs = {'color': color,
                  'ls': '-',
                  'marker': 'o',
                  'ms': 8}
        kwargs.update(line_kwargs)
        line, = ax.plot(ages, med, **kwargs)
        truth = None
        if show_truth:
            model = true_model or self.true_model
            if model is not None:
                if isinstance(model, CustomGalaxy):
                    model = model.sfh_model
                if not all_ages:
                    model = model.as_NonParam()
                ages = model.ages
                true_sfh = model.SFH
                cum_sfh = np.cumsum(true_sfh)
                cum_sfh /= cum_sfh[-1]
                kwargs = {'color': 'r',
                          'ls': '-',
                          'lw': 2,
                          'zorder': 10}
                kwargs.update(truth_kwargs)
                truth, = ax.plot(ages, cum_sfh, **kwargs)
        prior = None
        if show_prior:
            kwargs = {'width': 100.,
                      'all_ages': all_ages,
                      'burn': 0,
                      'trim': 0,
                      'max_logl': None,
                      'show_truth': False,
                      'title': False,
                      'legend': False,
                      'bulk_norm': bulk_norm,
                      'line_kwargs': {'alpha': 0.},
                      'error_kwargs': {'alpha': 0.},
                      'fill_kwargs': {'alpha': 1.0,
                                      'no_fill': True,
                                      'facecolor': 'k' if dark else 'w',
                                      'zorder': -10,
                                      'linestyle': ':',
                                      'linewidth': 2,
                                      'edgecolors': 'w' if dark else 'k'}}
            kwargs.update(prior_kwargs)
            _, lines = self.plot_cum_sfh(show_prior=False, ax=ax, **kwargs)
            prior = lines[2]
        ax.set_yscale('log')
        if title is None:
            ax.set_title(self.run_name)
        elif title is not False:
            ax.set_title(title)
        ax.set_xlabel('Log age (yr)')
        ax.set_ylabel('Log Cumulative SFH')
        if legend:
            ax.legend(((line, fill), prior, truth), ('Posterior', 'Prior', 'Truth'), loc=0)
        return ax, (line, error, fill, truth, prior)

    def plot_errorbars(self, axes, trim=0, burn=0, max_logl=None,
                       x=0, medians=True, offsets=None, **error_kwargs):
        if len(axes) < self.n_params:
            axes = list(axes) + [None]*(self.n_params - len(axes))
        if offsets is None:
            offsets = np.array([0.]*self.n_params)
        if len(offsets) < self.n_params:
            offsets = np.append(offsets, [0.]*len(self.n_params, len(offsets)))
        kwargs = {'marker': 'o',
                  'ms': 10,
                  'capsize': 6,
                  'ls': ''}
        kwargs.update(error_kwargs)
        if medians:
            ys = self.get_medians(trim=trim, burn=burn, max_logl=max_logl)
            yerrs = list(self.get_lower_upper(trim=trim, burn=burn,
                                              max_logl=max_logl))
            yerrs[0] = ys - yerrs[0]
            yerrs[1] = yerrs[1] - ys
            ys -= offsets
        else:
            ys = self.get_means(trim=trim, burn=burn, max_logl=max_logl) - offsets
            yerrs = [self.get_stds(trim=trim, burn=burn, max_logl=max_logl)]*2
        for i, a in enumerate(axes):
            if a is None:
                continue
            a.errorbar(x=[x], y=[ys[i]], yerr=[[yerrs[0][i]], [yerrs[1][i]]],
                       **kwargs)
        return axes

    def plot_violin(self, axes, trim=0, burn=0, max_logl=None,
                    x=0, offsets=None, color=None, **violin_kwargs):
        if len(axes) < self.n_params:
            axes = list(axes) + [None]*(self.n_params - len(axes))
        if offsets is None:
            offsets = np.array([0.]*self.n_params)
        if len(offsets) < self.n_params:
            offsets = np.append(offsets, [0.]*len(self.n_params, len(offsets)))
        kwargs = {'showmedians': True,
                  'showmeans': False,
                  'showextrema': False,
                  'points': 100,
                  'widths': 0.5}
        kwargs.update(violin_kwargs)
        ys = self.equal_samples
        for i, a in enumerate(axes):
            if a is None:
                continue
            violins = a.violinplot([ys[:, i] - offsets[i]], positions=[x],
                                   **kwargs)
            if color is not None:
                for pc in violins['bodies']:
                    pc.set_color(color)
        return axes

    def plot_corner_sfh(self, burn=0, trim=0, max_logl=None,
                        axes=None, cumulative=True,
                        widths=[68., 95.],
                        corner_kwargs={}, sfh_kwargs={}):
        """
        Plot corner plot with SFR in upper-right corner

        Returns
        -------
        fig
        axes
        axbig (SFH)
        lines (SFH)
        """
        if axes is None:
            fig, axes = plt.subplots(ncols=self.n_params, nrows=self.n_params,
                                     figsize=(2*self.n_params, 2*self.n_params))
        fig, axes = self.plot_corner(axes=axes, burn=burn, trim=trim,
                                     max_logl=max_logl, **corner_kwargs)
        gs = axes[0, 0].get_gridspec()
        for i in range(self.n_params):
            for j in range(self.n_params):
                if j > i:
                    axes[i, j].remove()
        x = (self.n_params // 2) + 1
        y = (self.n_params // 2) - 1
        axbig = fig.add_subplot(gs[:y, x:])
        sfh_func = (self.plot_cum_sfh if cumulative else self.plot_sfr)
        for w in widths:
            kws = sfh_kwargs.copy()
            if w != widths[-1]:
                kws.update({
                    'line_kwargs': {'alpha': 0.},
                    'show_prior': False,
                })
            axbig, lines = sfh_func(ax=axbig, burn=burn, trim=trim,
                                    max_logl=max_logl, width=w, **kws)
        return fig, axes, axbig, lines
    
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
        
    def restart_sampler(self, sampler, pri_inv_trans):
        dynamic = isinstance(sampler, DynamicSampler)
        N = int(np.sum(~self.df.live))
        N_live = int(np.sum(self.df.live))
        sampler.it = N
        sampler.ncall = int(self.df.nc[~self.df.live].sum())
        sampler.ubound_eff = 1e5
        sampler.ubound_ncall = -1
        sampler.since_update = sampler.ncall
        sampler.saved_id = [0 for _ in range(N)]
        N_p = self.gal_model._num_params
        params = self.params[:N_p]
        sampler.saved_v = list(self.df[params][~self.df.live].values)
        sampler.saved_u = [pri_inv_trans(v) for v in sampler.saved_v]
        sampler.saved_logl = list(self.df.logl[~self.df.live])
        sampler.saved_logvol = list(self.df.logvol[~self.df.live])
        sampler.saved_logwt = list(self.df.logwt[~self.df.live])
        sampler.saved_logz = list(self.df.logz[~self.df.live])
        sampler.saved_logzvar = list(self.df.logzerr[~self.df.live]**2)
        sampler.saved_h = list(self.df.h[~self.df.live])
        sampler.saved_nc = list(self.df.nc[~self.df.live].astype(int))
        sampler.saved_boundidx = [0 for _ in range(N)]
        sampler.saved_it = [0 for _ in range(N)]
        sampler.saved_bounditer = [0 for _ in range(N)]
        sampler.saved_scale = [1. for _ in range(N)]
        if dynamic:
            sampler.base = False
            sampler.saved_batch = [0 for _ in range(N)]
            sampler.saved_batch_nlive = [N_live]
            sampler.saved_batch_bounds = [(-np.inf, np.inf)]
            sampler.base_id = sampler.saved_id
            sampler.base_u = sampler.saved_u
            sampler.base_v = sampler.saved_v
            sampler.base_logl = sampler.saved_logl
            sampler.base_logvol = sampler.saved_logvol
            sampler.base_logwt = sampler.saved_logwt
            sampler.base_logz = sampler.saved_logz
            sampler.base_logzvar = sampler.saved_logzvar
            sampler.base_h = sampler.saved_h
            sampler.base_nc = sampler.saved_nc
            sampler.base_boundix = sampler.saved_boundidx
            sampler.base_it = sampler.saved_it
            sampler.base_n = [N_live for _ in range(N)]
            sampler.base_bounditer = sampler.saved_bounditer
            sampler.base_scale = sampler.saved_scale
