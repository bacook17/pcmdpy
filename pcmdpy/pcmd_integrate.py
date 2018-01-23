# pcmd_integrate.py
# Ben Cook (bcook@cfa.harvard.edu)
import numpy as np
from pcmdpy import fit_model
import sys
import argparse
import pandas as pd

from importlib import import_module

if __name__ == "__main__":

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
    config_file = args.config
    config_mod = config_file.strip('.py').rpartition('/')[-1]
    print(('Loading Setup File: {0}'.format(config_file)))
    config = import_module('./'+config_mod)

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
        data_pcmd = np.loadtxt(data_file, unpack=True)
    
    # where to save results
    results_file = args.results
    
    # These are the required arguments. If any remain None after import,
    # this will fail
    args = {}
    args['pcmd'] = data_pcmd

    required_keys = ['filters', 'N_im', 'N_live']
    
    # Load all parameters from configuration file
    # defaults are set by fit_model.nested_integrate
    for k, v in config.params.items():
        args[k] = v

    for key in required_keys:
        if key not in list(args.keys()):
            print(("Config file %s doesn\'t set required parameter %s" %
                  (config_file, key)))
            sys.exit(1)

    # The default dynesty result values
    results_cols = ['nlive', 'niter', 'ncall', 'eff',
                    'logl', 'logwt', 'logvol', 'logz',
                    'logzerr', 'h', 'delta_logz', 'time_elapsed']
    param_names = config.gal_class._param_names
    N_params = len(param_names)

    for pname in param_names:
        results_cols.append(pname)

    out_df = pd.DataFrame(columns=results_cols)
    args['out_file'] = results_file
    args['out_df'] = out_df
    args['save_every'] = 100  # fix this later
    args['param_names'] = param_names

    print('Running Nested Sampling')
    results = fit_model.nested_integrate(**args)
    
