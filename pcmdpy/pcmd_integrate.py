# pcmd_integrate.py
# Ben Cook (bcook@cfa.harvard.edu)
import numpy as np
from pcmdpy import fit_model

import pandas as pd
import sys

import imp

if __name__ == "__main__":

    # external config file
    setup_file = sys.argv[1]
    setup_mod = sys.argv[1].strip('.py').rpartition('/')[-1]
    print('Loading Setup File: %s'%setup_file)
    setup = imp.load_source(setup_mod, setup_file)

    # external data file
    data_file = sys.argv[2]
    data_pcmd = np.loadtxt(data_file, unpack=True)
    
    # where to save results
    results_file = sys.argv[3]
    
    #arguments
    args = {}
    args['pcmd'] = data_pcmd
    args['filters'] = setup.filters
    args['im_scale'] = setup.N_scale
    args['N_points'] = setup.N_points
    try:
        args['N_batch'] = setup.N_batch
    except:
        pass

    #optional key-word arguments (defaults are set by fit_model.nested_integrate)
    args['pool'] = setup.pool 
    args['max_call'] = setup.N_max
    args['gpu'] = setup.use_gpu
    args['fixed_seed'] = setup.fixed_seed
    args['like_mode'] = setup.like_mode
    args['small_prior'] = setup.small_prior
    try:
        args['lum_cut'] = setup.lum_cut
    except:
        pass
    try:
        args['dlogz'] = setup.dlogz
    except:
        pass
    try:
        args['use_dynesty'] = setup.use_dynesty
    except:
        args['use_dynesty'] = False

    try:
        args['dynamic'] = setup.dynamic
    except:
        args['dynamic'] = False

    try:
        args['prior_trans'] = setup.prior_trans
    except:
        args['prior_trans'] = None

    try:
        args['lnprior_func'] = setup.lnprior_func
    except:
        args['lnprior_func'] = None

    try:
        args['bound_method'] = setup.bound_method
    except:
        args['bound_method'] = 'multi'
        
    try:
        args['sample_method'] = setup.sample_method
    except:
        args['sample_method'] = 'unif'

    args['iso_model'] = setup.iso_model
    args['gal_class'] = setup.model_class
    args['verbose'] = setup.verbose

    results_cols = ['nlive', 'niter', 'ncall', 'eff',
                     'logl', 'logwt','logvol', 'logz',
                     'logzerr', 'h', 'delta_logz', 'time_elapsed']
    param_names = setup.model_class._param_names
    param_names[0] = 'logzh'
    N_params = len(param_names)
    out_file = setup.output_file

    for pname in param_names:
        results_cols.append(pname)

    out_df = pd.DataFrame(columns=results_cols)
    args['out_file'] = results_file
    args['out_df'] = out_df
    args['save_every'] = 100 #fix this later
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
