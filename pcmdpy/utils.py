# utils.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.misc import logsumexp
from datetime import datetime
import time
from pcmdpy import metalmodels, dustmodels, agemodels
from corner import corner
from astropy.io import fits


# A module to create various utility functions
def my_assert(bool_statement, fail_message=None):
    if bool_statement:
        return
    else:
        if fail_message is not None:
            print(fail_message)
        else:
            print("custom assertion failed")
        sys.exit(1)


def make_pcmd(mags):
    pcmd = np.copy(mags)
    n_filters = pcmd.shape[0]
    if (n_filters < 2):
        raise IndexError("Must be at least 2 images to create a PCMD")
    else:
        for i in range(1, n_filters):
            pcmd[i] = (mags[i] - mags[i-1]).flatten()
    return pcmd
    

def make_hess(pcmd, bins, charlie_err=False, err_min=2.):
    mags = pcmd[0]
    colors = pcmd[1:]
    n_colors = colors.shape[0]
    n = pcmd.shape[1]  # total number of pixels
    counts = []
    for i in range(n_colors):
        c, _, _ = np.histogram2d(mags, colors[i],
                                 bins=[bins[0], bins[i+1]])
        counts.append(c)
    counts = np.array(counts).flatten()
    # add "everything else" bin
    counts = np.append(counts, n - np.sum(counts))
    err = np.sqrt(counts)
    
    if charlie_err:
        # this is Charlie's method for inflating errors
        err[counts < 1.] = 0.1
        err[counts < 2.] *= 10.
    else:
        # inflate small errors, with inflation decreasing
        # exponentially at higher counts
        err += err_min * np.exp(-err)

    # normalize by number of pixels
    hess = counts / n
    err /= n
    
    return counts, hess, err


def wrap_image(image, w_border):
    my_assert(image.ndim == 2,
              "images must be 2-dimensional")
    Nx, Ny = image.shape
    if (w_border >= Nx) or (w_border >= Ny):
        message = "wrap_image is not implemented for cases where border is wider than existing image"
        print(w_border)
        raise NotImplementedError(message)
    w_roll = w_border // 2
    im_temp = np.tile(image, [2,2])
    im_temp = np.roll(np.roll(im_temp, w_roll, axis=0), w_roll, axis=1)

    return im_temp[:Nx+w_border, :Ny+w_border]


def subdivide_image(image, d_sub, w_border=0):
    my_assert(image.ndim == 2,
              "images must be 2-dimensional")
    Nx, Ny = image.shape
    if (Nx != Ny):
        message = "image must be square"
        raise NotImplementedError(message)
    if (Nx % d_sub != 0):
        message = "subdivide_image is only implemented if image can be cleanly subdivided"
        raise NotImplementedError(message)
    Nx_sub, Ny_sub = Nx // d_sub , Ny // d_sub

    if w_border > 0:
        image = wrap_image(image, w_border)

    sub_im_matrix = np.zeros((d_sub, d_sub, Nx_sub + w_border, Ny_sub + w_border))
    for i in range(d_sub):
        for j in range(d_sub):
            x_slice = slice(Nx_sub*i, Nx_sub*(i+1) + w_border)
            y_slice = slice(Ny_sub*j, Ny_sub*(j+1) + w_border)
            sub_im_matrix[i,j] = image[x_slice, y_slice]
    return sub_im_matrix


def subpixel_shift(image, dx, dy):
    assert(np.abs(dx)<= 1.)
    assert(np.abs(dy)<= 1.)
    #roll the image by -1, 0, +1 in x and y
    rolls = np.zeros((3, 3, image.shape[0], image.shape[1]))
    for i, x in enumerate([-1, 0, 1]):
        for j, y in enumerate([-1, 0, 1]):
            rolls[j,i] = np.roll(np.roll(image, x, axis=1), y, axis=0)
    #make the coefficients for each corresponding rolled image
    coeffs = np.ones((3, 3))
    if np.isclose(dx, 0.):
        coeffs[:, 0] = coeffs[:, 2] = 0.
    elif dx < 0.:
        coeffs[:, 2] = 0.
        coeffs[:, 0] *= -dx
        coeffs[:, 1] *= 1 + dx
    else:
        coeffs[:, 0] = 0.
        coeffs[:, 2] = dx
        coeffs[:, 1] *= 1 - dx
        
    if np.isclose(dy, 0.):
        coeffs[0, :] = coeffs[2, :] = 0.
    elif dy < 0.:
        coeffs[2, :] = 0.
        coeffs[0, :] *= -dy
        coeffs[1, :] *= 1 + dy
    else:
        coeffs[0, :] = 0.
        coeffs[2, :] *= dy
        coeffs[1, :] *= 1 - dy
    assert(np.isclose(np.sum(coeffs), 1.))
    result = np.zeros((image.shape[0], image.shape[0]))
    for i in range(3):
        for j in range(3):
            result += coeffs[i, j] * rolls[i, j]
    return result


def generate_image_dithers(image, shifts=[0., 0.25, 0.5, 0.75], norm=False):
    n = len(shifts)
    X = image.shape[0]
    Y = image.shape[1]
    tiles = np.zeros((n, n, X, Y))
    for i, dx in enumerate(shifts):
        for j, dy in enumerate(shifts):
            tiles[j, i] = subpixel_shift(image, dx, dy)
            if norm:
                tiles[j, i] /= np.sum(tiles[j, i])
    return tiles


class ResultsCollector(object):

    def __init__(self, ndim, verbose=True, print_loc=sys.stdout, out_file=None,
                 save_every=10, param_names=None):
        self.print_loc = print_loc
        self.last_time = time.time()
        self.start_time = time.time()
        if out_file is None:
            self.out_df = None
        else:
            self.colnames = ['nlive', 'niter', 'nc', 'eff', 'logl', 'logwt',
                             'logvol', 'logz', 'logzerr', 'h', 'delta_logz',
                             'time_elapsed']
            if param_names is not None:
                self.colnames += list(param_names)
            else:
                self.colnames += ['param{:d}'.format(i) for i in range(ndim)]
            self.out_df = pd.DataFrame(columns=self.colnames)
        self.out_file = out_file
        self.verbose = verbose
        self.save_every = save_every
        self.param_names = param_names
        
    def collect(self, results, niter, ncall, nbatch=None, dlogz=None,
                logl_max=None, add_live_it=None, stop_val=None,
                logl_min=-np.inf):
        if self.verbose:
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
            
            # constructing output
            print_str = 'iter: {:d}'.format(niter)
            if add_live_it is not None:
                print_str += "+{:d}".format(add_live_it)
            print_str += " | "
            if nbatch is not None:
                print_str += "batch: {:d} | ".format(nbatch)
            print_str += "nc: {:d} | ".format(nc)
            print_str += "ncalls: {:d} | ".format(ncall)
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
        if (self.out_df is not None):
            row = {'niter': niter}
            row['time_elapsed'] = total_time
            row['logl'] = loglstar
            row['logvol'] = logvol
            row['logwt'] = logwt
            row['logz'] = logz
            row['h'] = h
            row['eff'] = eff
            row['nc'] = nc
            row['nlive'] = 2000
            row['delta_logz'] = delta_logz
            row['logzerr'] = logzerr
            if self.param_names is not None:
                for i, pname in enumerate(self.param_names):
                    row[pname] = vstar[i]
            else:
                for i, v in enumerate(vstar):
                    row['param{0:d}'.format(i)] = v
            self.out_df = self.out_df.append(row, ignore_index=True)
            if ((niter+1) % self.save_every == 0):
                self.flush_to_csv()

    def flush_to_csv(self):
        self.out_df.to_csv(self.out_file, mode='a', index=False,
                           header=False, float_format='%.4e')
        self.out_df.drop(self.out_df.index, inplace=True)

        
class ResultsPlotter(object):

    param_labels = {'logfeh': '[Fe/H]', 'logfeh_mean': '[Fe/H]',
                    'logfeh_std': r'$\sigma([Fe/H])$',
                    'logzh': '[Z/H]', 'logdust': 'log E(B-V)',
                    'logdust_med': 'log E(B-V)',
                    'dust_sig': r'$\sigma(E(B-V))$', 'tau': r'$\tau$',
                    'tau_rise': r'$\tau$',
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
                self.metal_model = metalmodels.NormMDF
            else:
                self.metal_model = metalmodels.SingleFeH
            if 'dust_sig' in self.params:
                self.dust_model = dustmodels.LogNormDust
            else:
                self.dust_model = dustmodels.SingleDust
            if 'logSFH0' in self.params:
                self.age_model = agemodels.NonParam
            elif 'tau' in self.params:
                self.age_model = agemodels.TauModel
            elif 'tau_rise' in self.params:
                self.age_model = agemodels.RisingTau
            elif 'logage' in self.params:
                self.age_model = agemodels.SSPModel
            else:
                self.age_model = agemodels.ConstantSFR

        if self.age_model == agemodels.NonParam:
            sfhs = 10.**self.df[['logSFH{:d}'.format(i) for i in range(7)]]
            self.df['logNpix'] = np.log10(np.sum(sfhs.values, axis=1))
            self.params.append('logNpix')
            if self.true_params is not None:
                self.true_params += [np.log10(true_model.Npix)]
            
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [self.param_labels[p] for p in self.params]
            
        self.n_params = len(self.params)
            
        self.df['log_weights'] = (self.df.logwt.values -
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

    def plot_sfh(self, width=68., axis=None, title=None,
                 burn=0, show_prior=True, log=False, norm=True, **plot_kwargs):
        if (width > 100.) or (width < 0.):
            print('width must be between 0 and 100')
            return
        if self.age_model == agemodels.SSPModel:
            print('Cannot plot cumulative SFH for SSP')
            return
        n_plot = self.num_iters - burn
        cols = self.age_model._param_names
        vals = self.df[cols].values[-n_plot:]
        edges = self.age_model.default_edges
        ages = 10.**(edges[-1] - 9.) - 10.**(edges - 9.)
        if norm:
            dt = -np.diff(ages * 1e9)
        else:
            dt = np.ones(len(ages) - 1)
        ages = np.repeat(ages, 2)[1:-1]
        edges = np.repeat(edges, 2)[1:-1]
        if log:
            x = edges
        else:
            x = ages
        sfh = np.array([np.log10(self.age_model(v, iso_step=-1.).SFH / dt)
                        for v in vals])
        if self.true_model is not None:
            p_age = self.age_model._num_params
            vals_true = self.true_model._params[-p_age:]
            true_sfh = np.log10(self.age_model(vals_true,
                                               iso_step=-1.).SFH / dt)
        # ages = 0.5*(edges[:-1] + edges[1:])

        if axis is None:
            fig, axis = plt.subplots()
        med = np.repeat(np.percentile(sfh, 50., axis=0), 2)
        upper = np.repeat(np.percentile(sfh, 50. + 0.5*width, axis=0), 2)
        lower = np.repeat(np.percentile(sfh, 50. - 0.5*width, axis=0), 2)
        if 'color' in plot_kwargs:
            color = plot_kwargs.pop('color')
        else:
            color = 'k'
        if 'alpha' in plot_kwargs:
            alpha = plot_kwargs.pop('alpha')
        else:
            alpha = 0.3
        axis.plot(x, med, ls='-', color=color, **plot_kwargs)
        axis.fill_between(x, y1=lower, y2=upper, alpha=alpha, color=color,
                          **plot_kwargs)
        if self.true_model is not None:
            axis.plot(x, np.repeat(true_sfh, 2), ls='--', color=color, **plot_kwargs)
        axis.set_yscale('linear')
        if title is None:
            axis.set_title(self.run_name)
        else:
            axis.set_title(title)
        if log:
            axis.set_xlabel('log age (yr)')
        else:
            axis.set_xlabel('Time (Gyr)')
        if norm:
            axis.set_ylabel('Log Instantaneous SFR')
        else:
            axis.set_ylabel('Log Stars Formed')
        if show_prior:
            if self.prior is None:
                self.plot_sfh(burn=0, axis=axis, width=99.9, color='b',
                                  alpha=0.1, zorder=-1, show_prior=False,
                                  title=title, norm=norm, log=log, 
                                  **plot_kwargs)
            else:
                lower_p = self.prior.lower_bounds[-p_age:]
                upper_p = self.prior.upper_bounds[-p_age:]
                lower = np.repeat(np.log10(self.age_model(lower_p, iso_step=-1.).SFH / dt), 2)
                upper = np.repeat(np.log10(self.age_model(upper_p, iso_step=-1.).SFH / dt), 2)
                axis.fill_between(x, y1=lower, y2=upper, alpha=0.1,
                                  color='b', zorder=-1, **plot_kwargs)
        return axis
    
    def plot_cum_sfh(self, width=68., axis=None, title=None,
                     burn=0, show_prior=True, **plot_kwargs):
        if (width > 100.) or (width < 0.):
            print('width must be between 0 and 100')
            return
        if self.age_model == agemodels.SSPModel:
            print('Cannot plot cumulative SFH for SSP')
            return
        n_plot = self.num_iters - burn
        cols = self.age_model._param_names
        vals = self.df[cols].values[-n_plot:]
        cum_sfh = np.array([self.age_model(v, iso_step=-1.).get_cum_sfh()
                            for v in vals])
        if self.true_model is not None:
            p_age = self.age_model._num_params
            vals_true = self.true_model._params[-p_age:]
            true_cum_sfh = self.age_model(vals_true,
                                          iso_step=-1.).get_cum_sfh()
        edges = self.age_model.default_edges
        ages = 10.**(edges[-1] - 9.) - 10.**(edges - 9.)
        # ages = 0.5*(edges[:-1] + edges[1:])

        if axis is None:
            fig, axis = plt.subplots()
        med = np.percentile(cum_sfh, 50., axis=0)
        upper = np.percentile(cum_sfh, 50. + 0.5*width, axis=0)
        lower = np.percentile(cum_sfh, 50. - 0.5*width, axis=0)
        if 'color' in plot_kwargs:
            color = plot_kwargs.pop('color')
        else:
            color = 'k'
        if 'alpha' in plot_kwargs:
            alpha = plot_kwargs.pop('alpha')
        else:
            alpha = 0.3
        axis.plot(ages, med, ls='-', color=color, **plot_kwargs)
        axis.fill_between(ages, y1=lower, y2=upper, alpha=alpha, color=color,
                          **plot_kwargs)
        if self.true_model is not None:
            axis.plot(ages, true_cum_sfh, ls='--', color=color, **plot_kwargs)
        axis.set_yscale('linear')
        if title is None:
            axis.set_title(self.run_name)
        else:
            axis.set_title(title)
        axis.set_xlabel('Time (Gyr)')
        axis.set_ylabel('cumulative SFH')
        if show_prior:
            if self.prior is None:
                self.plot_cum_sfh(burn=0, axis=axis, width=99.9, color='b',
                                  alpha=0.1, zorder=-1, show_prior=False,
                                  title=title,
                                  **plot_kwargs)
            else:
                lower_p = self.prior.lower_bounds[-p_age:]
                upper_p = self.prior.upper_bounds[-p_age:]
                lower = self.age_model(lower_p, iso_step=-1.).get_cum_sfh()
                upper = self.age_model(upper_p, iso_step=-1.).get_cum_sfh()
                axis.fill_between(ages, y1=lower, y2=upper, alpha=0.1,
                                  color='b', zorder=-1, **plot_kwargs)
        return axis

    def plot_cum_sfh_log(self, width=68., axis=None, title=None,
                         burn=0, show_prior=True, **plot_kwargs):
        if (width > 100.) or (width < 0.):
            print('width must be between 0 and 100')
            return
        if self.age_model == agemodels.SSPModel:
            print('Cannot plot cumulative SFH for SSP')
            return
        n_plot = self.num_iters - burn
        cols = self.age_model._param_names
        vals = self.df[cols].values[-n_plot:]
        cum_sfh = np.array([self.age_model(v, iso_step=-1.).get_cum_sfh(inverted=False)
                            for v in vals])
        if self.true_model is not None:
            p_age = self.age_model._num_params
            vals_true = self.true_model._params[-p_age:]
            true_cum_sfh = self.age_model(vals_true,
                                          iso_step=-1.).get_cum_sfh(inverted=False)
        edges = self.age_model.default_edges
        # ages = 10.**(edges[-1] - 9.) - 10.**(edges - 9.)
        ages = 0.5*(edges[:-1] + edges[1:])

        if axis is None:
            fig, axis = plt.subplots()
        med = np.percentile(cum_sfh, 50., axis=0)
        upper = np.percentile(cum_sfh, 50. + 0.5*width, axis=0)
        lower = np.percentile(cum_sfh, 50. - 0.5*width, axis=0)
        if 'color' in plot_kwargs:
            color = plot_kwargs.pop('color')
        else:
            color = 'k'
        if 'alpha' in plot_kwargs:
            alpha = plot_kwargs.pop('alpha')
        else:
            alpha = 0.3
        axis.plot(ages, med, ls='-', color=color, **plot_kwargs)
        axis.fill_between(ages, y1=lower, y2=upper, alpha=alpha, color=color,
                          **plot_kwargs)
        if self.true_model is not None:
            axis.plot(ages, true_cum_sfh, ls='--', color=color, **plot_kwargs)
        axis.set_yscale('log')
        if title is None:
            axis.set_title(self.run_name)
        else:
            axis.set_title(title)
        axis.set_xlabel('log age (yr)')
        axis.set_ylabel('log cumulative SFH')
        if show_prior:
            if self.prior is None:
                self.plot_cum_sfh_log(burn=0, axis=axis, width=99.9, color='b',
                                  alpha=0.1, zorder=-1, show_prior=False,
                                  title=title,
                                  **plot_kwargs)
            else:
                lower_p = self.prior.lower_bounds[-p_age:]
                upper_p = self.prior.upper_bounds[-p_age:]
                lower = self.age_model(lower_p, iso_step=-1.).get_cum_sfh(inverted=False)
                upper = self.age_model(upper_p, iso_step=-1.).get_cum_sfh(inverted=False)
                axis.fill_between(ages, y1=lower, y2=upper, alpha=0.1,
                                  color='b', zorder=-1, **plot_kwargs)
        return axis

    
    def plot_corner(self, fig=None, title=None, burn=0, bins=30,
                    smooth_frac=.03, smooth1d=0.,
                    weight=True, full_range=True,
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
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 10))
        axes = axes.flatten()
        self.plot_cum_sfh(axis=axes[0], **cum_sfh_kwargs)
        self.plot_cum_sfh_log(axis=axes[1], **cum_sfh_kwargs)
        self.plot_sfh(axis=axes[2], norm=True, log=False, **cum_sfh_kwargs)
        self.plot_sfh(axis=axes[3], norm=False, log=False,  **cum_sfh_kwargs)
        self.plot_sfh(axis=axes[4], norm=True, log=True,  **cum_sfh_kwargs)
        self.plot_sfh(axis=axes[5], norm=False, log=True, **cum_sfh_kwargs)
        corner_fig, corner_axes = self.plot_corner(**corner_kwargs)
        return (chain_axes, axes, (corner_fig, corner_axes))


class DataSet(object):
    
    def __init__(self, file_names, filter_classes):
        assert(len(file_names) == len(filter_classes))
        self.n_bands = len(filter_classes)
        headers = []
        with fits.open(file_names[0]) as hdu:
            self.im_shape = hdu[0].data.shape
            self.images = np.zeros((self.im_shape[0], self.im_shape[1],
                                    self.n_bands))
            headers.append(hdu[0].header)
            self.images[:, :, 0] = hdu[0].data
        for i, f in enumerate(file_names[1:]):
            with fits.open(f) as hdu:
                self.images[:, :, i+1] = hdu[0].data
                headers.append(hdu[0].header)
        self.headers = np.array(headers)
        assert(self.images.ndim == 3)  # else the images weren't matching sizes
        filters = []
        for filt, header in zip(filter_classes, headers):
            filters.append(filt(exposure=header['EXPTIME']))
        self.filters = np.array(filters)
        
    def get_pcmd(self, bool_matrix, bands=None):
        if bands is not None:
            assert(max(bands) < self.n_bands)
            assert(min(bands) <= 0)
            pixels = self.images[bool_matrix, bands]
        else:
            bands = np.arange(self.n_bands)
            pixels = self.images[bool_matrix, :]
        assert(bool_matrix.shape == self.im_shape)
        filts = self.filters[bands]
        mags = np.zeros_like(pixels.T)
        for i in bands:
            flux = pixels[:, i] * filts[i]._exposure  # convert to counts
            mags[i] = filts[i].counts_to_mag(flux)
        pcmd = make_pcmd(mags)
        return pcmd

    def get_image(self, bool_matrix=None, downsample=1, bands=None):
        if bands is None:
            bands = np.arange(self.n_bands)
        if bool_matrix is None:
            images = np.copy(self.images[::downsample, ::downsample, :])
            for i, b in enumerate(bands):
                images[:, :, i] *= self.filters[b]._exposure
            xmin, ymin = 0, 0
            xmax, ymax = self.im_shape
        else:
            assert(bool_matrix.shape == self.images.shape[:2])
            bool_matrix = bool_matrix[::downsample, ::downsample]
            x, y = np.where(bool_matrix)
            xmin, xmax = min(x), max(x)+1
            ymin, ymax = min(y), max(y)+1
            images = np.zeros((xmax-xmin, ymax-ymin, 3))
            for a in [xmin, xmax, ymin, ymax]:
                a *= downsample
            bools = bool_matrix[xmin:xmax, ymin:ymax]
            for i, b in enumerate(bands):
                images[bools, i] = self.images[::downsample,
                                               ::downsample][bool_matrix, b]
                images[:, :, i] *= self.filters[b]._exposure
        return images, (ymin, ymax, xmin, xmax)

    
