# fit_model.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from pcmdpy import isochrones, galaxy, driver, utils, priors
import sys
import dynesty
import time
from datetime import datetime


def lnlike(gal_params, driv, N_im, lnprior_func,
           gal_class=galaxy.NonParam, **kwargs):
    pri = lnprior_func(gal_params)
    if np.isinf(pri):
        return -np.inf
    gal_model = gal_class(gal_params)
    pcmd, _ = driv.simulate(gal_model, N_im, **kwargs)
    like = driv.loglike(pcmd, **kwargs)

    return like


def lnprob(gal_params, driv, N_im, lnprior_func, gal_class=galaxy.NonParam,
           **kwargs):
    pri = lnprior_func(gal_params)
    if np.isinf(pri):
        return -np.inf
    like = lnlike(gal_params, driv, N_im, lnprior_func,
                  gal_class=gal_class, **kwargs)
    return pri + like


def dynesty_run(func, out_df=None, out_file=None, save_every=10,
                param_names=None, ncall_start=0, tstart=0., **func_kwargs):
    ncall = ncall_start
    if 'dlogz' in list(func_kwargs.keys()):
        dlogz = func_kwargs['dlogz']
    else:
        dlogz = np.nan
    start = time.time()
    for it, results in enumerate(func(**func_kwargs)):
        dt = (time.time() - start) + tstart
        row = {'niter': it}
        row['time_elapsed'] = dt
        (worst, ustar, vstar, row['logl'], row['logvol'], row['logwt'], row['logz'], logzvar, row['h'], nc, worst_it, propidx, propiter, row['eff'], delta_logz) = results
        ncall += nc
        ave_t = float(dt) / ncall
        row['ncall'] =  ncall
        row['nlive'] = 2000
        if delta_logz > 1e6:
            delta_logz = np.inf
        row['delta_logz'] = delta_logz
        if logzvar >= 0.:
            row['logzerr'] = np.sqrt(logzvar)
        else:
            row['logzerr'] = np.nan
        if param_names is not None:
            for i, pname in enumerate(param_names):
                row[pname] = vstar[i]
        else:
            for i, v in enumerate(vstar):
                row['param{0:d}'.format(i)] = v
                    
        if out_df is not None:
            out_df = out_df.append(row, ignore_index=True)
            if ((it+1) % save_every == 0) and (out_file is not None):
                out_df.to_csv(out_file, mode='a', index=False, header=False, float_format='%.4e')
                out_df.drop(out_df.index, inplace=True)
        
        message = 'iteration: {:d} | nc: {:d} | ncalls: {:d} | eff(%): {:3.1f} | logz: {:.1e} +/- {:.1e} | dlogz: {:.1e} > {:6.3f}'.format(it, nc, ncall, row['eff'], row['logz'], row['logzerr'], row['delta_logz'], dlogz)
        message += '\n loglike: {:.1e} | params: {:s}'.format(row['logl'], str(vstar))
        message += '\n Average call time: {:.2f} sec | Current time: {:s}'.format(ave_t, str(datetime.now()))
        message += '\n --------------------------'
        print(message)
        sys.stdout.flush()
    #save remaining lines
    if out_df is not None:
        out_df.to_csv(out_file, mode='a', index=False, header=False, float_format='%.4e')
        out_df.drop(out_df.index, inplace=True)

    return ncall, dt

def nested_integrate(pcmd, filters, N_im, N_live, method='multi', max_call=100000, gal_class=galaxy.NonParam, use_gpu=True, iso_model=None,
                     bins=None, verbose=False, dlogz=None, dynamic=False, N_batch=0, save_live=False,
                     pool=None, out_df=None, out_file=None, save_every=100, param_names=None, prior=None, **kwargs):
    print('-initializing models')
    n_filters = len(filters)
    utils.my_assert(pcmd.shape[0] == n_filters,
                    "pcmd shape doesn\'t match number of filters")
    n_dim = gal_class._num_params
    if pool is None:
        nprocs = 1
    else:
        nprocs = pool._processes

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
                      gal_class=gal_class, **kwargs)

    # Initialize the nestle sampler with a different random state than global
    # This is important because the driver resets the global seed
    rstate = np.random.RandomState(1234)

    if dynamic:
        sampler = dynesty.DynamicNestedSampler(this_lnlike, this_pri_transform, ndim=n_dim, bound=method,
                                               sample='unif', rstate=rstate, pool=pool, nprocs=nprocs)
        print('-Running dynesty dynamic sampler')
        sampler.run_nested(nlive_init=N_live, maxcall=max_call, nlive_batch=N_batch,
                           wt_kwargs={'pfrac':1.0}, print_progress=True,
                           print_to_stderr=False, dlogz_init=dlogz)
    else:
        sampler = dynesty.NestedSampler(this_lnlike, this_pri_transform, ndim=n_dim,
                                        bound=method, sample='unif', nlive=N_live,
                                        update_interval=1, rstate=rstate, pool=pool,
                                        nprocs=nprocs, boostrap=0, enlarge=1.1, first_update={'min_eff':30.})
        if (out_df is not None) and (out_file is not None):
            print('-Saving initial results dataframe')
            out_df.to_csv(out_file, index=False, float_format='%.4e')
        print('-Running dynesty sampler')
        dlogz_final = dlogz
        ncall, dt = dynesty_run(sampler.sample, out_df=out_df, save_every=save_every,
                                param_names=param_names, ncall_start=0,
                                dlogz=dlogz_final, maxcall=max_call, out_file=out_file)
        if save_live:
            print('-Adding live points at end of dynesty samping')
            _, _ = dynesty_run(sampler.add_live_points, out_df=out_df, save_every=save_every,
                               param_names=param_names, ncall_start=ncall, tstart=dt, out_file=out_file)

    results = sampler.results
    if (out_df is not None) and (out_file is not None):
        print('-Saving final results dataframe')
        out_df.to_csv(out_file, mode='a', index=False, header=False, float_format='%.4e')

    if driv.num_calls >= (max_call - 1):
        print('Terminated after surpassing max likelihood calls')
    else:
        print('Reached desired convergence')

    return results
