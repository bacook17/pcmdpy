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
                     compute_maxlogl=True, continue_run=False,
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

    if prior is None:
        prior = gal_model.get_flat_prior()

    if (live_file is None) or (out_file is None):
        continue_run = False
        
    if continue_run:
        # Load the most recently saved live points and continue running from there
        try:
            live_df = pd.read_csv(live_file)
            samples_v = live_df[param_names].values
            samples_u = np.array([prior.prior_transform(v) for v in samples_v])
            logls = live_df[['logl']].values
            sampler_kwargs['live_points'] = [samples_u, samples_v, logls]
        except FileNotFoundError:
            continue_run = False

    if (out_df is not None) and (out_file is not None):
        if (not continue_run):
            print('-Saving initial results dataframe')
            out_df.to_csv(out_file, index=False, float_format='%.4e')

    logl_kwargs['downsample'] = downsample
    logl_kwargs['mag_system'] = mag_system
    
    sampler = get_sampler(gal_model, driv, Nim,
                          prior=prior, dynamic=dynamic,
                          sampler_seed=sampler_seed,
                          sampler_kwargs=sampler_kwargs,
                          **logl_kwargs)

    if continue_run:
        from ..results.results import ResultsPlotter
        original_res = ResultsPlotter(out_file, live_file=live_file,
                                      max_logl=-np.inf)
        original_res.restart_sampler(sampler, prior.inverse_prior_transform)

    logger = ResultsLogger(sampler, out_file, out_df=out_df,
                           live_file=live_file,
                           save_every=save_every,
                           param_names=param_names)
    
    run_kwargs['print_func'] = logger.collect
    sampler.run_nested(**run_kwargs)

    if (logger.out_df is not None):
        print('-Saving final results dataframe')
        logger.flush_to_csv()

    if compute_maxlogl:
        print('-Computing Max Logl')
        logl = sampler.results['logl']
        best_params = sampler.results['samples'][logl.argmax()]
        logl_kwargs.pop('fixed_seed', False)
        logls = [lnlike(best_params, driv, Nim, prior.lnprior,
                        gal_model, fixed_seed=False, **logl_kwargs) for _ in range(100)]
        logl_max = np.median(logls)
        print('Max Logl: {:.3e}'.format(logl_max))
        if out_file is not None:
            with open(out_file, 'r') as f:
                text = f.read()
                # If a previous max_logl is saved, delete it
                if 'max_logl' in text:
                    text = text.partition('\n')[-1]
            with open(out_file, 'w') as f:
                f.write('# max_logl : {:.3e}\n'.format(logl_max))
                f.write(text)

    return sampler


def get_sampler(gal_model, driv, Nim, prior=None, dynamic=False,
                sampler_seed=1234, sampler_kwargs={}, **logl_kwargs):
    ndim = gal_model._num_params
    logl_args = (driv, Nim, prior.lnprior, gal_model)
    logl_kwargs['downsample'] = logl_kwargs.get('downsample', 5)
    logl_kwargs['mag_system'] = logl_kwargs.get('mag_system', 'vega')
    logl_kwargs['fixed_seed'] = logl_kwargs.get('fixed_seed', False)

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
