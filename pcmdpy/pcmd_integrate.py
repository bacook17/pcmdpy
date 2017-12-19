# pcmd_integrate.py
# Ben Cook (bcook@cfa.harvard.edu)
import numpy as np
from pcmdpy import fit_model
import argparse
import pandas as pd

import imp

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config",
                        help="Name of the configuration file (.py)", type=str,
                        required=True)
    parser.add_argument("--data", help="Name of the PCMD data file (.dat)",
                        type=str, required=True)
    parser.add_argument("--results", help="Name of the results file (.csv)",
                        type=str, required=True)
    args = parser.parse_args()
    
    # external config file
    config_file = args.config
    config_mod = config_file.strip('.py').rpartition('/')[-1]
    print('Loading Setup File: %s' % config_file)
    config = imp.load_source(config_mod, config_file)

    # external data file
    data_file = args.data
    data_pcmd = np.loadtxt(data_file, unpack=True)
    
    # where to save results
    results_file = args.results
    
    # arguments
    args = {}
    args['pcmd'] = data_pcmd
    args['filters'] = config.filters
    args['im_scale'] = config.N_scale
    args['N_points'] = config.N_points
    try:
        args['N_batch'] = config.N_batch
    except AttributeError:
        pass

    # optional key-word arguments
    # (defaults are set by fit_model.nested_integrate)
    args['pool'] = config.pool
    args['max_call'] = config.N_max
    args['gpu'] = config.use_gpu
    args['fixed_seed'] = config.fixed_seed
    args['like_mode'] = config.like_mode
    args['small_prior'] = config.small_prior
    try:
        args['lum_cut'] = config.lum_cut
    except AttributeError:
        pass
    try:
        args['dlogz'] = config.dlogz
    except AttributeError:
        pass
    try:
        args['use_dynesty'] = config.use_dynesty
    except AttributeError:
        args['use_dynesty'] = False

    try:
        args['dynamic'] = config.dynamic
    except AttributeError:
        args['dynamic'] = False

    try:
        args['prior_trans'] = config.prior_trans
    except AttributeError:
        args['prior_trans'] = None

    try:
        args['lnprior_func'] = config.lnprior_func
    except AttributeError:
        args['lnprior_func'] = None

    try:
        args['bound_method'] = config.bound_method
    except AttributeError:
        args['bound_method'] = 'multi'
        
    try:
        args['sample_method'] = config.sample_method
    except AttributeError:
        args['sample_method'] = 'unif'

    try:
        args['save_live'] = config.save_live
    except AttributeError:
        args['save_live'] = False

    args['iso_model'] = config.iso_model
    args['gal_class'] = config.model_class
    args['verbose'] = config.verbose

    results_cols = ['nlive', 'niter', 'ncall', 'eff',
                    'logl', 'logwt', 'logvol', 'logz',
                    'logzerr', 'h', 'delta_logz', 'time_elapsed']
    param_names = config.model_class._param_names
    param_names[0] = 'logzh'
    N_params = len(param_names)
    out_file = config.output_file

    for pname in param_names:
        results_cols.append(pname)

    out_df = pd.DataFrame(columns=results_cols)
    args['out_file'] = results_file
    args['out_df'] = out_df
    args['save_every'] = 100  # fix this later
    args['param_names'] = param_names

    print('Running Nested Sampling')
    results = fit_model.nested_integrate(**args)

    #print('Nested Sampling Complete, saving results')
    #Used for saving output

    ##Save results
    #out_df = pd.DataFrame(columns=results_cols)
    #for d in range(N_params):
    #    out_df[param_names[d]] = results.samples[:,d]
    ########Fix nomenclature
    #for col in results_cols:
    #    try:
    #        out_df[col] = getattr(results, col)
    #    except:
    #        if col not in param_names:
    #            print('%s not found among result keys'%(col))
    #
    #out_df.to_csv(out_file, index=False, float_format='%.3e', compression='gzip')
