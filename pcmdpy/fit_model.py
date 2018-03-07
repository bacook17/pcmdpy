# fit_model.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from pcmdpy import isochrones, galaxy, driver, utils, priors
import sys
import dynesty
import time
import pandas as pd
from datetime import datetime


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

        
def lnlike(gal_params, driv, N_im, lnprior_func,
           gal_class=galaxy.DefaultNonParam, **kwargs):
    pri = lnprior_func(gal_params)
    if np.isinf(pri):
        return -np.inf
    gal_model = gal_class(gal_params)
    pcmd, _ = driv.simulate(gal_model, N_im, **kwargs)
    like = driv.loglike(pcmd, **kwargs)

    return like


def lnprob(gal_params, driv, N_im, lnprior_func,
           gal_class=galaxy.DefaultNonParam, **kwargs):
    pri = lnprior_func(gal_params)
    if np.isinf(pri):
        return -np.inf
    like = lnlike(gal_params, driv, N_im, lnprior_func,
                  gal_class=gal_class, **kwargs)
    return pri + like


def nested_integrate(pcmd, filters, N_im, gal_class=galaxy.DefaultNonParam,
                     use_gpu=True, iso_model=None, bins=None, verbose=False,
                     dynamic=False, out_df=None, out_file=None, save_every=10,
                     param_names=None, prior=None, sampler_kwargs={},
                     run_kwargs={}, **ln_kwargs):
    # Default sampler arguments
    run_kwargs['print_progress'] = True
    run_kwargs['save_bounds'] = False
    print('dynamic: ', dynamic)
    print(sampler_kwargs)
    print(run_kwargs)

    print('-initializing models')
    n_filters = len(filters)
    utils.my_assert(pcmd.shape[0] == n_filters,
                    "pcmd shape doesn\'t match number of filters")
    n_dim = gal_class._num_params

    if iso_model is None:
        iso_model = isochrones.Isochrone_Model(filters)
    driv = driver.Driver(iso_model, gpu=use_gpu)
    if bins is None:
        utils.my_assert(n_filters == 2,
                        "Default behavior only defined for 2 filters")
        xbins = np.arange(-1.5, 4.6, 0.05)
        ybins = np.arange(-12, 15.6, 0.05)
        bins = np.array([xbins, ybins])
    driv.initialize_data(pcmd, bins)

    if prior is None:
        try:
            prior = priors.default_prior[gal_class]
        except KeyError:
            print('No prior object given, and no default prior set for this '
                  'galaxy class')
            sys.exit(2)
    
    this_pri_transform = prior.prior_transform
    lnprior_func = prior.lnprior

    def this_lnlike(gal_params):
        return lnlike(gal_params, driv, N_im, lnprior_func,
                      gal_class=gal_class, **ln_kwargs)

    # Initialize the nestle sampler with a different random state than global
    # This is important because the driver resets the global seed
    rstate = np.random.RandomState(1234)

    if (out_df is not None) and (out_file is not None):
        print('-Saving initial results dataframe')
        out_df.to_csv(out_file, index=False, float_format='%.4e')
    if dynamic:
        sampler = dynesty.DynamicNestedSampler(this_lnlike, this_pri_transform,
                                               ndim=n_dim, rstate=rstate,
                                               **sampler_kwargs)
        print('Dynamic Sampler Initialized')
        
    else:
        sampler = dynesty.NestedSampler(this_lnlike, this_pri_transform,
                                        ndim=n_dim, rstate=rstate,
                                        **sampler_kwargs)
        print('Traditional Sampler Initialized')

    collector = ResultsCollector(n_dim, out_file=out_file,
                                 save_every=save_every,
                                 param_names=param_names)
    
    run_kwargs['print_func'] = collector.collect
    sampler.run_nested(**run_kwargs)

    if (collector.out_df is not None):
        print('-Saving final results dataframe')
        collector.flush_to_csv()
