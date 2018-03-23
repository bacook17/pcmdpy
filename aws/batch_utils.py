#
# Copyright 2013-2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the
# License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions and
# limitations under the License.
#
# Submits an image classification training job to an AWS Batch job queue, and tails the CloudWatch log output.
#

import boto3


def submitJob(job_name, job_queue, job_definition, region='us-east-1',
              verbose=True, parameters={}):
    endpoint_url = 'https://batch.{:s}.amazonaws.com'.format(region)
    batch_client = boto3.client(service_name='batch',
                                region_name=region,
                                endpoint_url=endpoint_url)
    response = batch_client.submit_job(jobName=job_name, jobQueue=job_queue,
                                       jobDefinition=job_definition,
                                       parameters=parameters)
    job_id = response['jobId']
    if verbose:
        print('Submitted job [{:s}] to job queue [{:s}]'.format(job_name,
                                                                job_queue))
        print('Assigned jobId: [{:s}]'.format(job_id))
    return batch_client, response, job_id


def getLogURL(job_name, job_id, region='us-east-1', verbose=True):
    endpoint_url = 'https://logs.{:s}.amazonaws.com'.format(region)
    cw_client = boto3.client(service_name='logs',
                             region_name=region,
                             endpoint_url=endpoint_url)
    prefix = job_name + '/' + job_id
    logStreams = cw_client.describe_log_streams(logGroupName='/aws/batch/job',
                                                logStreamNamePrefix=prefix)['logStreams']
    if len(logStreams) > 0:
        stream_name = logStreams[0]['logStreamName']
        log_url = 'https://console.aws.amazon.com/cloudwatch/home?region='
        log_url += region
        log_url += '#logEventViewer:group=/aws/batch/job;stream='
        log_url += stream_name
        if verbose:
            print('URL to view logs:')
            print(log_url)
        return log_url
    else:
        if verbose:
            print('unable to find log stream')
        return ''

    
def getJobStatus(job_id, region='us-east-1', verbose=True):
    endpoint_url = 'https://batch.{:s}.amazonaws.com'.format(region)
    batch_client = boto3.client(service_name='batch',
                                region_name=region,
                                endpoint_url=endpoint_url)
    response = batch_client.describe_jobs(jobs=[job_id])
    if len(response['jobs']) > 0:
        status = response['jobs'][0]['status']
        if verbose:
            print('The job status is: {:s}'.format(status))
        return status
    else:
        print('unable to find job matching jobId: {:s}'.format(job_id))
        return 'NOT_FOUND'
        

def queryJob(job_name, job_id, region='us-east-1', verbose=True):
    kwargs = {'region': region, 'verbose': verbose}
    status = getJobStatus(job_id, **kwargs)
    log_url = getLogURL(job_name, job_id, **kwargs)
    return status, log_url
