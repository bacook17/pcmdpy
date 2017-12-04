# fit_model.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
import instrument as ins
import isochrones as iso
import galaxy as gal
import sys
import driver
import utils
#import emcee
#import nestle

import dynesty
import time
from datetime import datetime

def lnprior_ssp(gal_params):
    z, log_dust, log_Npix, age = gal_params
    #Flat priors
    # log (z/z_solar) between -2 and 0.5
    if (z < -2.) or (z>0.5):
        return -np.inf
    # E(B-V) between 1e-3 and 3
    if (log_dust < -3) or (log_dust > 0.5):
        return -np.inf
    #Npix between 0.1 and 1e6
    if (log_Npix < -1) or (log_Npix > 6):
        return -np.inf
    #age between 6 (1 Myr) and 10.3 (50 Gyr)
    if (age < 6.) or (age > 10.3):
        return -np.inf
    return 0.

def lnprior_transform_ssp(normed_params):
    results = np.zeros(len(normed_params))
    #Flat priors
    # log (z/z_solar) between -2 and 0.5
    results[0] = -2 + 2.5*normed_params[0]
    # E(B-V) between 1e-3 and 3
    results[1] = -3 + 3.5*normed_params[1]
    #log Npix between -1 and 6
    results[2] = -1 + 7*normed_params[2]
    #age between 6 (1 Myr) and 10.3 (50 Gyr)
    results[3] = 6 + 4.3*normed_params[3]
    return results

def lnprior_transform_ssp_small(normed_params):
    results = np.zeros(len(normed_params))
    #Flat priors
    # log (z/z_solar) between -0.5 and 0.0
    results[0] = -.5 + .5*normed_params[0]
    # E(B-V) between -2.5 and -1.5
    results[1] = -2.5 + normed_params[1]
    #log Npix between 1.5 and 2.5
    results[2] = 1.5 + normed_params[2]
    #age between 9.5 (3 Gyr) and 10.0 (10 Gyr)
    results[3] = 9.5 + 0.5*normed_params[3]
    return results

def lnprior(gal_params):
    z, log_dust = gal_params[:2]
    log_SFH = gal_params[2:]
    log_Npix = np.log10(np.sum(10.**log_SFH))
    #Flat priors
    # log Npix between -1 and 6
    if (log_Npix < -1.) or (log_Npix > 6.):
        return -np.inf
    # log (z/z_solar) between -2 and 0.5
    if (z < -2.) or (z > 0.5):
        return -np.inf
    # E(B-V) between 1e-3 and 3
    if (log_dust < -3) or (log_dust > 0.5):
        return -np.inf
    # log M_i / Mtot between 1e-6 and 1
    for log_m in log_SFH:
        if (log_m < -10 + log_Npix) or (log_m > 0 + log_Npix):
            return -np.inf
    return 0.

def lnprior_transform(normed_params):
    #HUGE HACK: returns more dimensions than used in model!
    results = np.zeros(len(normed_params))
    #Flat priors
    # log (z/z_solar) between -2 and 0.5
    results[0] = -2. + 2.5*normed_params[0]
    # E(B-V) between 1e-3 and 3
    results[1] = -3 + 3.5*normed_params[1]
    # log M_i between -6 and 0
    for i in range(2, len(normed_params)-1):
        results[i] = -6 + 6*normed_params[i]
    log_total = np.log10(np.sum(10.**results[2:-1]))
    #HACKHACKHACK
    #log Npix between -1 and 6
    log_Npix = -1 + 7*normed_params[2]
    results[-1] = log_Npix
    #Normalize the mass bins to sum to log_Npix
    results[2:-1] += log_Npix - log_total
    return results

def lnprior_transform_small(normed_params):
    results = np.zeros(len(normed_params))
    #Flat priors
    # log (z/z_solar) between -0.75 and -0.25
    results[0] = -0.75 + 0.5*normed_params[0]
    # E(B-V) between 3e-2 and 3e-1
    results[1] = -1.5 + normed_params[1]
    # log M_i between +/- 0.5 of truth
    appx_truth = np.array([ 0.75426991,  1.75426991,  2.13493886,  2.63493886,  3.13493886,
        3.63493886,  3.56710397])
    for i in range(2, len(normed_params)):
        results[i] = appx_truth[i-2] - 0.5 + normed_params[i]
    return results

def lnlike(gal_params, driv, im_scale, lnprior_func=None, gal_class=gal.Galaxy_Model, **kwargs):
    if lnprior_func is None:
        if (gal_class is gal.Galaxy_SSP):
            pri = lnprior_ssp(gal_params)
        else:
            pri = lnprior(gal_params)
    else:
        pri = lnprior_func(gal_params)
    if np.isinf(pri):
        return -np.inf
    gal_model = gal_class(gal_params)
    mags, _ = driv.simulate(gal_model, im_scale, **kwargs)
    pcmd = utils.make_pcmd(mags)
    like = driv.loglike(pcmd, **kwargs)

    return like

def lnprob(gal_params, driv, im_scale, gal_class=gal.Galaxy_Model, **kwargs):
    if (gal_class is gal.Galaxy_SSP):
        pri = lnprior_ssp(gal_params)
    else:
        pri = lnprior(gal_params)
    if np.isinf(pri):
        return -np.inf
    like = lnlike(gal_params, driv, im_scale, gal_class=gal_class, **kwargs)
    return pri + like

def dynesty_run(func, out_df=None, out_file=None, save_every=10, param_names=None, ncall_start=0, tstart=0., **func_kwargs):
    ncall = ncall_start
    if 'dlogz' in func_kwargs.keys():
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
        ave_t = dt / ncall
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
                row['param%d'%i] = v
                    
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

def nested_integrate(pcmd, filters, im_scale, N_points, method='multi', max_call=100000, gal_class=gal.Galaxy_Model, gpu=True, iso_model=None,
                     bins=None, verbose=False, small_prior=False, dlogz=None, dynamic=False, N_batch=0, save_live=True,
                     pool=None, out_df=None, out_file=None, save_every=100, param_names=None, prior_trans=None, lnprior_func=None, **kwargs):
    print('-initializing models')
    n_filters = len(filters)
    assert(pcmd.shape[0] == n_filters)
    n_dim = gal_class._num_params
    if pool is None:
        nprocs = 1
    else:
        nprocs = pool._processes

    if iso_model is None:
        iso_model = iso.Isochrone_Model(filters)
    driv = driver.Driver(iso_model, gpu=gpu)
    if bins is None:
        assert(n_filters == 2)
        xbins = np.arange(-1.5, 4.6, 0.05)
        ybins = np.arange(-12, 15.6, 0.05)
        bins = np.array([xbins,ybins])
    driv.initialize_data(pcmd,bins)

    if prior_trans is not None:
        this_pri_transform = prior_trans
    else:
        if gal_class is gal.Galaxy_Model:
            if small_prior:
                this_pri_transform = lnprior_transform_small
            else:
                ndim += 1 #HACKHACKHACK
                this_pri_transform = lnprior_transform
        else:
            if small_prior:
                this_pri_transform = lnprior_transform_ssp_small
            else:
                this_pri_transform = lnprior_transform_ssp

    def this_lnlike(gal_params):
        #HACKHACKHACK to remove trailing zero
        if (not small_prior) and (gal_class is gal.Galaxy_Model):
            gal_params = gal_params[:-1]
        return lnlike(gal_params, driv, im_scale, gal_class=gal_class, lnprior_func=lnprior_func, **kwargs)

    #Initialize the nestle sampler with a different random state than global
    #This is important because the driver resets the global seed
    rstate = np.random.RandomState(1234)

    if dynamic:
        sampler = dynesty.DynamicNestedSampler(this_lnlike, this_pri_transform, ndim=n_dim, bound=method,
                                               sample='unif', rstate=rstate, pool=pool, nprocs=nprocs)
        print('-Running dynesty dynamic sampler')
        sampler.run_nested(nlive_init=N_points, maxcall=max_call, nlive_batch=N_batch,
                           wt_kwargs={'pfrac':1.0}, print_progress=True,
                           print_to_stderr=False, dlogz_init=dlogz)
    else:
        sampler = dynesty.NestedSampler(this_lnlike, this_pri_transform, ndim=n_dim,
                                        bound=method, sample='unif', nlive=N_points,
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

#def sample_post(pcmd, filters, im_scale, N_walkers, N_burn, N_sample, 
#                p0=None, gal_class=gal.Galaxy_Model, gpu=True, bins=None, threads=1, fixed_seed=True,
#                rare_cut=0., like_mode=0, 
#                **kwargs):
#
#    print('-initializing models')
#    N_filters = len(filters)
#    assert(pcmd.shape[0] == N_filters)
#    N_dim = gal_class._num_params
#    
#    iso_model = iso.Isochrone_Model(filters)
#    driv = driver.Driver(iso_model, gpu=gpu)
#    if bins is None:
#        assert(N_filters == 2)
#        xbins = np.arange(-1.5, 4.6, 0.05)
#        ybins = np.arange(-12, 15.6, 0.05)
#        bins = np.array([xbins,ybins])
#    driv.initialize_data(pcmd,bins)
#
#    print('-Setting up emcee sampler')
#    
#    sampler = emcee.EnsembleSampler(N_walkers, N_dim, lnprob, args=[driv, im_scale], kwargs={'gal_class':gal_class, 'fixed_seed':fixed_seed,'rare_cut':rare_cut, "like_mode":like_mode},
#                                    threads=threads, **kwargs)
#
#    if p0 is None:
#        if (gal_class is gal.Galaxy_Model):
#            np.random.seed(0)
#            z0 = np.random.uniform(-2., 0.5, N_walkers)
#            np.random.seed(0)
#            dust0 = np.random.uniform(-6, 0, N_walkers)
#            np.random.seed(0)
#            npix0 = 10.**np.random.uniform(-1, 6, N_walkers)
#            sfh0 = 10.**np.random.uniform(-10, 0, (7, N_walkers))
#            sfh0 *= npix0 / np.sum(sfh0, axis=0)
#            sfh0 = np.log10(sfh0)
#            p0 = np.array([z0, dust0])
#            p0 = np.concatenate([p0, sfh0]).T
#        else:
#            np.random.seed(0)
#            z0 = np.random.uniform(-2., 0.5, N_walkers)
#            np.random.seed(0)
#            dust0 = np.random.uniform(-6, 0, N_walkers)
#            np.random.seed(0)
#            npix0 = np.random.uniform(-1, 6, N_walkers)
#            np.random.seed(0)
#            age0 = np.random.uniform(6, 10.3, N_walkers)
#            p0 = np.array([z0, dust0, npix0, age0]).T
#    assert(p0.shape == (N_walkers, N_dim))
#
#    if N_burn > 0:
#        print('-emcee burn-in')
#        pos,prob,state = sampler.run_mcmc(p0, N_burn)
#
#        print('-emcee sampling')
#        sampler.reset()
#        sampler.run_mcmc(pos, N_sample)
#
#    else:
#        sampler.run_mcmc(p0, N_sample)
#
#    return sampler
#
