# fit_model.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
from pcmdpy import isochrones, galaxy, driver, utils, priors
import sys
import dynesty


def lnlike(gal_params, driv, N_im, lnprior_func,
           gal_class=galaxy.DefaultNonParam, **kwargs):
    pri = lnprior_func(gal_params)
    if np.isinf(pri):
        return -np.inf
    gal_model = gal_class.get_model(gal_params)
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
        xbins = np.arange(-12, 15.6, 0.05)
        ybins = np.arange(-1.5, 4.6, 0.05)
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

    collector = utils.ResultsCollector(n_dim, out_file=out_file,
                                       save_every=save_every,
                                       param_names=param_names)
    
    run_kwargs['print_func'] = collector.collect
    sampler.run_nested(**run_kwargs)

    if (collector.out_df is not None):
        print('-Saving final results dataframe')
        collector.flush_to_csv()
