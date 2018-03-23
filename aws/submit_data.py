# submit_data.py
# Ben Cook (bcook@cfa.harvard.edu)
# Submits a pcmdpy data run to an AWS Batch job queue

import argparse
from batch_utils import submitJob


def submit_data(run_name, job_queue, job_definition, config_file, data_file,
                run_version='', region='us-east-1', verbose=True):
    if run_version:
        job_name = run_name + '_v' + run_version
    else:
        job_name = run_name
    parameters = {'config_file': config_file,
                  'run_name': run_name,
                  'data_file': data_file}
    kwargs = {'verbose': verbose, 'region': region, 'parameters': parameters}
    return submitJob(job_name, job_queue, job_definition, **kwargs)


if __name__ == '__main__':
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument("--run-name", help="name of the run", type=str,
                        required=True)
    parser.add_argument("--run-version",
                        help="which version of this run is it", type=str,
                        default='')
    parser.add_argument("--job-queue",
                        help="name of the job queue to submit this job",
                        type=str, default="pcmdpy_queue_p2")
    parser.add_argument("--job-definition", help="name of the job definition",
                        type=str, default="pcmdpy_mock")
    parser.add_argument("--config-file", help="the configuration file",
                        type=str, required=True)
    parser.add_argument("--data-file", help="the input data file",
                        type=str, required=True)
    parser.add_argument("--region", help="AWS region to submit job to",
                        type=str, default='us-east-1')
    parser.add_argument("--array-job",
                        help="submit an array of jobs, defined by files",
                        action='store_true')
    parser.add_argument("--quiet",
                        help="silence printing of job metadata to STDOUT",
                        action='store_true')
    args = parser.parse_args()

    verbose = ~args.quiet

    if args.array_job:
        if verbose:
            print('Submitting array job')
        with open(args.run_name, 'r') as f:
            run_names = f.readlines()
        with open(args.config_file, 'r') as f:
            config_files = f.readlines()
        with open(args.data_file, 'r') as f:
            data_files = f.readlines()
        n_rn = len(run_names)
        n_cf = len(config_files)
        n_df = len(data_files)
        if (n_rn != n_cf) or (n_rn != n_df):
            print(('number of run_names in {:s}, config_files in {:s}, and'
                   'data_files in {:s} do not all match.'
                   'Exiting.'.format(args.run_name, args.config_file,
                                     args.data_file)))
            return
        for rn, cf, df in zip(run_names, config_files, data_files):
            submit_data(rn, args.job_queue, args.job_definition, cf, df,
                        run_version=args.run_version, region=args.region,
                        verbose=verbose)
    else:
        if verbose:
            print('Submitting single job')
        submit_data(args.run_name, args.job_queue, args.job_definition,
                    args.config_file, args.data_file,
                    run_version=args.run_version, region=args.region,
                    verbose=verbose)
