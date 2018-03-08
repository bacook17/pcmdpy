# utils.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from scipy.misc import logsumexp
from datetime import datetime
import time
from pcmdpy import agemodels

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


def make_pcmd(data):
    pcmd = np.copy(data)
    n_filters = pcmd.shape[0]
    if (n_filters < 2):
        raise IndexError("Must be at least 2 images to create a PCMD")
    else:
        for i in range(1, n_filters):
            pcmd[i] = (data[i] - data[0]).flatten()
    return pcmd
    
def make_hess(pcmd, bins, charlie_err=False, err_min=2.):
    n_dim = pcmd.shape[0]
    n = pcmd.shape[1] #total number of pixels
    if (n_dim != bins.shape[0]):
        raise IndexError("The first dimensions of pcmd and bins must match")
    counts = np.histogramdd(pcmd.T, bins=bins)[0].astype(float)
    #add "everything else" bin
    counts = counts.flatten()
    counts = np.append(counts, n - np.sum(counts))
    err = np.sqrt(counts)
    
    if charlie_err:
        #this is Charlie's method for inflating errors
        err[counts < 1.] = 0.1
        err[counts < 2.] *= 10.
    else:
        #inflate small errors, with inflation decreasing exponentially at higher counts
        err += err_min * np.exp(-err)

    #normalize by number of pixels
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

    def __init__(self, df_file, truths=None, run_name=None, params=None,
                 param_labels=None):
        try:
            self.df = pd.read_csv(df_file)
        except UnicodeDecodeError:
            self.df = pd.read_csv(df_file, compression='gzip')

        self.truths_dict = truths
        self.param_labels = param_labels
        self.run_name = run_name
        self.num_iters = len(self.df)
        
        self.default_params = {'logfeh': '[Fe/H]', 'logfeh_mean': '[Fe/H]',
                               'logfeh_std': r'$\sigma([Fe/H])$',
                               'logzh': '[Z/H]', 'logdust': 'log E(B-V)',
                               'logdust_med': 'log E(B-V)',
                               'dust_sig': r'$\sigma(E(B-V))$',
                               'tau': r'$\tau$', 'tau_rise': r'$\tau$',
                               'logNpix': 'log Npix'}
        for i in range(7):
            s = 'logSFH{:d}'.format(i)
            self.default_params[s] = s
        if params is None:
            self.params = []
            for p in self.default_params.keys():
                if p in self.df.columns:
                    self.params.append(p)
        else:
            self.params = list(params)

        if ('logSFH0' in self.params) and ('logNpix' not in self.params):
            sfhs = 10.**self.df[['logSFH{:d}'.format(i) for i in range(7)]]
            self.df['logNpix'] = np.log10(np.sum(sfhs.values, axis=1))
            self.params.append('logNpix')

        self.n_params = len(self.params)
            
        self.df['log_weights'] = (self.df.logwt.values -
                                  logsumexp(self.df.logwt.values))
        self.df['weights'] = np.exp(self.df['log_weights'])
        self.df['time_elapsed'] /= 3600.
        try:
            self.df['logfeh'] = self.df.logzh
        except AttributeError:
            pass

    def plot_chains(self, axes=None, title=None, dlogz=0.5):
        nr = self.n_params + 3
        if axes is None:
            fig, axes = plt.subplots(nrows=nr, figsize=(8, 2+nr), sharex=True)
        else:
            assert(len(axes) == nr)
        if title is None:
            title = self.run_name
        for i, p in enumerate(self.params):
            axes[i].plot(self.df[p].values)
            if self.param_labels is None:
                axes[i].set_ylabel(self.default_params[p])
            else:
                axes[i].set_ylabel(self.param_labels[i])
        axes[-3].plot(np.log10(self.df['delta_logz'].values))
        axes[-3].axhline(y=np.log10(dlogz), ls='--', color='r')
        axes[-3].set_ylabel(r'log $\Delta$ln Z')
        axes[-2].plot(self.df['eff'].values)
        axes[-2].set_ylabel('eff (%)')
        axes[-1].plot(self.df['time_elapsed'].values)
        axes[-1].set_ylabel('run time (hrs)')
        axes[-1].set_xlabel('Iteration')

        if self.truths_dict is not None:
            for i, p in enumerate(self.params):
                axes[i].axhline(y=self.truths_dict[p], color='r', ls='--')

        if title is not None:
            axes[0].set_title(title)
        return axes

    def plot_cum_sfh(self, width=68., axis=None, title=None,
                     burn=0, color='k', **plot_kwargs):
        if (width > 100.) or (width < 0.):
            print('width must be between 0 and 100')
            return
        n_plot = self.num_iters - burn
        if ('logSFH0' in self.params):
            cols = ['logSFH{:d}'.format(i) for i in range(7)]
            normed_sfh = 10.**self.df[cols].values[-n_plot:].T
            normed_sfh /= 10.**self.df['logNpix'].values[-n_plot:]
            normed_sfh = normed_sfh.T
            if self.truths_dict is not None:
                truth_sfh = 10.**([self.truths_dict[p] for p in cols])
                truth_sfh /= np.sum(truth_sfh)
                truth_sfh = np.cumsum(truth_sfh)
        elif ('tau' in self.params):
            taus = self.df['tau'].values
            normed_sfh = np.zeros((n_plot, 7))
            for i in range(0, n_plot):
                model = agemodels.TauModel(np.array([0., taus[i+burn]]),
                                           iso_step=-1.)
                normed_sfh[i] = model.SFH
            if self.truths_dict is not None:
                t = self.truths_dict['tau']
                truth_sfh = agemodels.TauModel(np.array([0., t]),
                                               iso_step=-1.).SFH
                truth_sfh = np.cumsum(truth_sfh)
        elif ('tau_rise' in self.params):
            taus = self.df['tau_rise'].values
            normed_sfh = np.zeros((n_plot, 7))
            for i in range(0, n_plot):
                model = agemodels.RisingTau(np.array([0., taus[i+burn]]),
                                            iso_step=-1.)
                normed_sfh[i] = model.SFH
            if self.truths_dict is not None:
                t = self.truths_dict['tau_rise']
                truth_sfh = agemodels.TauModel(np.array([0., t]),
                                               iso_step=-1.).SFH
                truth_sfh = np.cumsum(truth_sfh)
        elif ('logage' in self.params):
            print('Cannot plot cumulative SFH for SSP')
            return
        else:
            sfh_base = agemodels.ConstantSFR(np.array([0.]), iso_step=-1.).SFH
            normed_sfh = np.ones((n_plot, 7)) * sfh_base
            truth_sfh = sfh_base
        if axis is None:
            fig, axis = plt.subplots()
        cum_sfh = np.cumsum(normed_sfh, axis=1)
        upper_lim = 50 + (width / 2.)
        lower_lim = 50 - (width / 2.)
        med = np.percentile(cum_sfh, 50, axis=0)
        upper = np.percentile(cum_sfh, upper_lim, axis=0)
        lower = np.percentile(cum_sfh, lower_lim, axis=0)
        edges = agemodels._AgeModel.default_edges
        ages = 0.5*(edges[:-1] + edges[1:])
        axis.plot(ages, med, 'k-', color=color, **plot_kwargs)
        axis.fill_between(ages, y1=lower, y2=upper, alpha=0.3, color=color,
                          **plot_kwargs)
        if self.truths_dict is not None:
            axis.plot(ages, truth_sfh, 'r--')
        axis.set_yscale('log')
        axis.set_title(title)
        axis.set_xlabel('log age (yr)')
        axis.set_ylabel('log cumulative SFH')
        return axis
