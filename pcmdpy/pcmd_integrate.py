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
    warnings.simplefilter(action='ignore', category=UserWarning)
    
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
    parser.add_argument("--results", help="Name of the results file (.csv)",
                        type=str, required=True)
    args = parser.parse_args()
    
    # external config file
    try:
        config_file = args.config
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
    data_file = args.data
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
    results_file = args.results
    
    # These are the required arguments. If any remain None after import,
    # this will fail
    args = {}
    args['pcmd'] = data_pcmd

    required_keys = ['filters', 'N_im', 'gal_model', 'prior']
    
    # Load all parameters from configuration file
    # defaults are set by fit_model.nested_integrate
    args.update(config.params)

    args['sampler_kwargs'] = config.sampler_params
    args['run_kwargs'] = config.run_params

    all_keys = list(args.keys())
    all_keys += list(config.sampler_params.keys())
    all_keys += list(config.run_params.keys())
    
    for key in required_keys:
        if key not in all_keys:
            print(("Config file %s doesn\'t set required parameter %s" %
                  (config_file, key)))
            sys.exit(1)

    # The default dynesty result values
    results_cols = ['nlive', 'niter', 'nc', 'eff',
                    'logl', 'logwt', 'logvol', 'logz',
                    'logzerr', 'h', 'delta_logz', 'time_elapsed']
    param_names = config.params['gal_model']._param_names
    N_params = len(param_names)

    for pname in param_names:
        results_cols.append(pname)

    out_df = pd.DataFrame(columns=results_cols)
    args['out_file'] = results_file
    args['out_df'] = out_df
    args['save_every'] = 10
    args['param_names'] = param_names

    print('Running Nested Sampling')
    fit_model.nested_integrate(**args)
    
