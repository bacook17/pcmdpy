# fit_model.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from ..isochrones import Isochrone_Model
from ..simulation.driver import Driver
from .results import ResultsCollector
import dynesty


def lnlike(gal_params, driv, N_im, lnprior_func,
           gal_model, downsample=5, mag_system='vega', **kwargs):
    pri = lnprior_func(gal_params)
    if np.isinf(pri):
        return -np.inf
    gal_model.set_params(gal_params)
    pcmd, _ = driv.simulate(gal_model, N_im, downsample=downsample,
                            mag_system=mag_system, **kwargs)
    like = driv.loglike(pcmd, **kwargs)

    return like


def lnprob(gal_params, driv, N_im, lnprior_func,
           gal_model, **kwargs):
    pri = lnprior_func(gal_params)
    if np.isinf(pri):
        return -np.inf
    like = lnlike(gal_params, driv, N_im, lnprior_func,
                  gal_model, **kwargs)
    return pri + like


def nested_integrate(pcmd, filters, N_im, gal_model,
                     use_gpu=True, iso_model=None, bins=None, verbose=False,
                     dynamic=False, out_df=None, out_file=None, live_file=None,
                     save_every=10, param_names=None, prior=None,
                     sampler_kwargs={}, run_kwargs={}, downsample=5,
                     mag_system='vega', **ln_kwargs):
    # Default sampler arguments
    run_kwargs['print_progress'] = True
    run_kwargs['save_bounds'] = False

    print('-initializing models')
    n_filters = len(filters)
    assert pcmd.shape[0] == n_filters, (
        "pcmd shape ({:d}) does not match filter numbers ({:d})".format(
            pcmd.shape[0], n_filters))
    n_dim = gal_model._num_params

    if iso_model is None:
        iso_model = Isochrone_Model(filters)
    driv = Driver(iso_model, gpu=use_gpu)
            
    driv.initialize_data(pcmd, bins=bins)

    if prior is None:
        prior = gal_model.get_flat_prior()
    
    this_pri_transform = prior.prior_transform
    lnprior_func = prior.lnprior

    def this_lnlike(gal_params):
        return lnlike(gal_params, driv, N_im, lnprior_func, gal_model,
                      downsample=downsample, mag_system=mag_system,
                      **ln_kwargs)

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

    collector = ResultsCollector(sampler, out_file, out_df=out_df,
                                 live_file=live_file,
                                 save_every=save_every,
                                 param_names=param_names)
    
    run_kwargs['print_func'] = collector.collect
    sampler.run_nested(**run_kwargs)

    if (collector.out_df is not None):
        print('-Saving final results dataframe')
        collector.flush_to_csv()

    return sampler
