#!/usr/bin/env python
# pcmd_integrate.py
# Ben Cook (bcook@cfa.harvard.edu)
import numpy as np
from pcmdpy import fit_model
import sys
import argparse
import signal
import pandas as pd
from traceback import print_exc
from importlib import util
import warnings


def sigterm_handler(sig, frame):
    # exit on sigint
    print('Exiting due to external signal')
    sys.exit(0)


if __name__ == "__main__":

    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config",
                        help="Name of the configuration file (.py)", type=str,
                        required=True)
    parser.add_argument("--data", help=("Name of the PCMD data file (.dat). "
                                        "If not given, assume a mock run."),
                        type=str, default="")
    parser.add_argument("--results-init",
                        help="Name of the initial run results file (.csv)",
                        type=str, required=True)
    parser.add_argument("--results-final",
                        help="Name of the final run results file (.csv)",
                        type=str, required=True)
    cline_args = parser.parse_args()
    
    # external config file
    try:
        config_file = cline_args.config
        config_mod = config_file.strip('.py').rpartition('/')[-1]
        print(('Loading Setup File: {0}'.format(config_file)))
        spec = util.spec_from_file_location(config_mod, config_file)
        config = util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except ModuleNotFoundError as e:
        print_exc()
        print('Unable to import module')
        print(e)
        print('Check stderr for full traceback')
        sys.exit(1)
    except Exception as e:
        print_exc()
        print('Other error when importing')
        print(e)
        print('Check stderr for full traceback')
        sys.exit(1)

    # external data file
    data_file = cline_args.data
    if len(data_file) == 0:
        try:
            if config.params['data_is_mock']:
                data_pcmd = config.params['data_pcmd']
            else:
                raise ValueError
        except (KeyError, ValueError):
            print(("No --data option provided, but {0} does not have "
                  "data_is_mock set to True".format(config_file)))
            sys.exit(1)
    else:
        data = np.loadtxt(data_file, unpack=True)  # Data columns are Mag-Color
        data_pcmd = data
        # data_pcmd = data[::-1, :]  # swap to Mag-Color order
    
    # where to save results
    init_results_file = cline_args.results_init
    final_results_file = cline_args.results_final
    
    # These are the required arguments. If any remain None after import,
    # this will fail
    args = {}
    args['pcmd'] = data_pcmd

    gal_init = config.params.pop('init_gal_model')
    gal_final = config.params.pop('final_gal_model')

    required_keys = ['filters', 'N_im', 'gal_model',
                     'init_prior']
    
    # Load all parameters from configuration file
    # defaults are set by fit_model.nested_integrate
    args.update(config.params)
    args['gal_model'] = gal_init

    for key in required_keys:
        if key not in args.keys():
            print(("Config file %s doesn\'t set required parameter %s" %
                  (config_file, key)))
            sys.exit(1)

    # The default dynesty result values
    results_cols = ['nlive', 'niter', 'nc', 'eff',
                    'logl', 'logwt', 'logvol', 'logz',
                    'logzerr', 'h', 'delta_logz', 'time_elapsed']

    init_param_names = config.params['init_gal_model']._param_names
    final_param_names = config.params['final_gal_model']._param_names
    
    init_results_cols = results_cols + list(init_param_names)
    final_results_cols = results_cols + list(final_param_names)

    # Setup for initial run
    out_df = pd.DataFrame(columns=init_results_cols)
    args['out_file'] = init_results_file
    args['out_df'] = out_df
    args['save_every'] = 10
    args['param_names'] = init_param_names

    args['sampler_kwargs'] = config.init_sampler_params
    args['run_kwargs'] = config.init_run_params

    print('Running Initial Nested Sampling Model')
    init_sampler = fit_model.nested_integrate(**args)

    # Setup the prior object for the second model
    live_points = init_sampler.live_v
    bound_levels = [2.5, 97.5]  # compute 95% range
    prior_bounds = {}

    # Compute 95% range of metallicity parameters.
    # Set prior to be 25% wider than that range
    prior_bounds['feh_bounds'] = []
    for i in range(gal_init.p_feh):
        low = np.percentile(live_points[:, i], bound_levels[0])
        high = np.percentile(live_points[:, i], bound_levels[1])
        width = high - low
        # inflate by 25%
        low -= 0.125 * width
        high += 0.125 * width
        prior_bounds['feh_bounds'].append([low, high])

    # Compute 95% range of dust parameters.
    # Set prior to be 25% wider than that range
    prior_bounds['dust_bounds'] = []
    for i in range(gal_init.p_feh, gal_init.p_feh + gal_init.p_dust):
        low = np.percentile(live_points[:, i], bound_levels[0])
        high = np.percentile(live_points[:, i], bound_levels[1])
        width = high - low
        # inflate by 25%
        low -= 0.125 * width
        high += 0.125 * width
        prior_bounds['dust_bounds'].append([low, high])
        
    # Compute 95% range of distance parameters.
    # Set prior to be 25% wider than that range
    prior_bounds['dmod_bounds'] = []
    for i in range(-gal_init.p_distance, 0):
        low = np.percentile(live_points[:, i], bound_levels[0])
        high = np.percentile(live_points[:, i], bound_levels[1])
        width = high - low
        # inflate by 25%
        low -= 0.125 * width
        high += 0.125 * width
        prior_bounds['dmod_bounds'].append([low, high])

    age_model = gal_init.age_model
    new_edges = gal_final.age_model.default_edges
    s = slice(gal_init.p_feh + gal_init.p_dust,
              gal_init.p_feh + gal_init.p_dust + gal_init.p_age)
    logSFH = np.array([age_model.set_params(p).as_default().update_edges(new_edges).logSFH
                       for p in live_points[:, s]])

    # Compute 95% range of SFH parameters
    # Set prior to be 2x wider than that range
    prior_bounds['age_bounds'] = []
    for i in range(gal_final.p_age):
        low = np.percentile(logSFH[:, i], bound_levels[0])
        high = np.percentile(logSFH[:, i], bound_levels[1])
        width = high - low
        # inflate by 2x
        low -= 0.5 * width
        high += 0.5 * width
        prior_bounds['age_bounds'].append([low, high])
    
    args['prior'] = gal_final.get_flat_prior(**prior_bounds)
    args['gal_model'] = gal_final
    # Run the second model

    out_df = pd.DataFrame(columns=final_results_cols)
    args['out_file'] = final_results_file
    args['out_df'] = out_df
    args['save_every'] = 10
    args['param_names'] = final_param_names

    args['sampler_kwargs'] = config.final_sampler_params
    args['run_kwargs'] = config.final_run_params

    print('Running Final Nested Sampling Model')
    final_sampler = fit_model.nested_integrate(**args)
