# utils.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from scipy.misc import logsumexp
import sys
import pandas as pd
from datetime import datetime
import time
from astropy.io import fits


# A module to create various utility functions
def make_pcmd(mags):
    pcmd = np.copy(mags)
    n_filters = pcmd.shape[0]
    for i in range(1, n_filters):
        pcmd[i] = mags[i] - mags[i-1]
    return pcmd


def pcmd_to_mags(pcmd):
    mags = np.copy(pcmd)
    n_filters = mags.shape[0]
    for i in range(1, n_filters):
        mags[i] = pcmd[i] + mags[i-1]
    return mags


def mean_mags(pcmd):
    mags = pcmd_to_mags(pcmd)
    mag_factor = -0.4 * np.log(10)  # convert from base 10 to base e
    weights = float(1) / mags.shape[1]  # evenly weight each pixel
    return logsumexp(mag_factor*mags, b=weights, axis=1)


def make_hess(pcmd, bins, err_min=2.):
    mags = pcmd[0]
    colors = pcmd[1:]
    n_colors = colors.shape[0]
    n = pcmd.shape[1]  # total number of pixels
    counts = []
    for i in range(n_colors):
        c, _, _ = np.histogram2d(mags, colors[i],
                                 bins=[bins[0], bins[i+1]])
        if np.sum(c) < n:  # if some pixels fell outside bins, add to corner of Hess grid
            c[0, 0] += (n - np.sum(c))
        counts += [c]
    if n_colors == 0:
        c, _ = np.histogram(mags, bins=bins[0])
        counts += [c]
    counts = np.array(counts)
    counts[counts <= 0.] = 0.
    err = np.sqrt(counts)
    
    # inflate small errors
    err[err <= err_min] = err_min
    # err += err_min * np.exp(-err)
    
    # normalize by number of pixels
    hess = counts / n
    err /= n
    
    return counts, hess, err


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

        
class DataSet(object):
    
    def __init__(self, file_names, filter_classes):
        assert(len(file_names) == len(filter_classes))
        self.n_bands = len(filter_classes)
        headers = []
        with fits.open(file_names[0]) as hdu:
            if len(hdu) > 1:
                data = hdu['SCI'].data
            else:
                data = hdu['PRIMARY'].data
            self.im_shape = data.shape
            self.images = np.zeros((self.im_shape[0], self.im_shape[1],
                                    self.n_bands))
            headers.append(hdu[0].header)
            self.images[:, :, 0] = data
        for i, f in enumerate(file_names[1:]):
            with fits.open(f) as hdu:
                if len(hdu) > 1:
                    data = hdu['SCI'].data
                else:
                    data = hdu['PRIMARY'].data
                self.images[:, :, i+1] = data
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

    
