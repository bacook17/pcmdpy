# fit_model.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from ..isochrones import Isochrone_Model
from ..simulation.driver import Driver
from .logging import ResultsLogger
import dynesty


def lnlike(gal_params, driv, Nim, lnprior_func,
           gal_model, downsample=5, mag_system='vega', **kwargs):
    pri = lnprior_func(gal_params)
    if np.isinf(pri):
        return -np.inf
    gal_model.set_params(gal_params)
    pcmd, _ = driv.simulate(gal_model, Nim, downsample=downsample,
                            mag_system=mag_system, **kwargs)
    like = driv.loglike(pcmd, **kwargs)

    return like


def lnprob(gal_params, driv, Nim, lnprior_func,
           gal_model, **kwargs):
    pri = lnprior_func(gal_params)
    if np.isinf(pri):
        return -np.inf
    like = lnlike(gal_params, driv, Nim, lnprior_func,
                  gal_model, **kwargs)
    return pri + like


def nested_integrate(pcmd, filters, Nim, gal_model,
                     use_gpu=True, iso_model=None, bins=None, verbose=False,
                     dynamic=False, out_df=None, out_file=None, live_file=None,
                     save_every=10, param_names=None, prior=None,
                     sampler_seed=1234, sampler_kwargs={}, run_kwargs={},
                     downsample=5, mag_system='vega', **logl_kwargs):
    # Default sampler arguments
    run_kwargs['print_progress'] = True
    run_kwargs['save_bounds'] = False

    print('-initializing models')
    n_filters = len(filters)
    assert pcmd.shape[0] == n_filters, (
        "pcmd shape ({:d}) does not match filter numbers ({:d})".format(
            pcmd.shape[0], n_filters))

    if iso_model is None:
        iso_model = Isochrone_Model(filters)
    driv = Driver(iso_model, gpu=use_gpu)
            
    driv.initialize_data(pcmd, bins=bins)

    if (out_df is not None) and (out_file is not None):
        print('-Saving initial results dataframe')
        out_df.to_csv(out_file, index=False, float_format='%.4e')
    
    sampler = get_sampler(gal_model, driv, Nim,
                          prior=prior, dynamic=dynamic,
                          sampler_seed=sampler_seed,
                          sampler_kwargs=sampler_kwargs,
                          **logl_kwargs)

    logger = ResultsLogger(sampler, out_file, out_df=out_df,
                           live_file=live_file,
                           save_every=save_every,
                           param_names=param_names)
    
    run_kwargs['print_func'] = logger.collect
    sampler.run_nested(**run_kwargs)

    if (logger.out_df is not None):
        print('-Saving final results dataframe')
        logger.flush_to_csv()

    return sampler


def get_sampler(gal_model, driv, Nim, prior=None, dynamic=False,
                sampler_seed=1234, sampler_kwargs={}, **logl_kwargs):
    if prior is None:
        prior = gal_model.get_flat_prior()
    ndim = gal_model._num_params
    logl_args = {'driv': driv,
                 'Nim': Nim,
                 'lnprior_func': prior.lnprior,
                 'gal_model': gal_model}
    logl_kwargs['downsample'] = logl_kwargs.get('downsample', 5)
    logl_kwargs['mag_system'] = logl_kwargs.get('mag_system', 'vega')

    # Initialize the nestle sampler with a different random state than global
    # This is important because the driver resets the global seed
    rstate = np.random.RandomState(sampler_seed)
    if dynamic:
        sampler = dynesty.DynamicNestedSampler
    else:
        sampler = dynesty.NestedSampler
    sampler = sampler(lnlike, prior.prior_transform,
                      ndim=ndim, rstate=rstate,
                      logl_args=logl_args, logl_kwargs=logl_kwargs,
                      **sampler_kwargs)
    return sampler
