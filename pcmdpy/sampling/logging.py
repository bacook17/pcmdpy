import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import pandas as pd
from scipy.special import logsumexp
import sys
from datetime import datetime
import time


class ResultsLogger(object):

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

        # Save current live points, only if in initial run
        if (((niter+1) % self.save_every == 0) and nbatch is None):
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
