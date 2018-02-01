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
# Creates the minimal set of AWS Batch entities (compute environment, job queue, job definition)
# to be able to submit a GPU job.
#

import boto3
import argparse
import time
import sys

batch = boto3.client(
    service_name='batch',
    region_name='us-east-2',
    endpoint_url='https://batch.us-east-2.amazonaws.com')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--compute-environment", help="name of the compute environment", type=str, required=True)
parser.add_argument("--jobdef-name", help="name of the job definition", type=str, required=True)
parser.add_argument("--subnets", help="comma delimited list of subnets", type=str, default='subnet-3e880473,subnet-823978f9,subnet-126e6f7b')
parser.add_argument("--security-groups", help="comma delimited list of security group ids", type=str, default='sg-2e00be46')
parser.add_argument("--instance-role", help="instance role", type=str, default='arn:aws:iam::845819056159:instance-profile/ecsInstanceRole')
parser.add_argument("--service-role", help="service role", type=str, default='arn:aws:iam::845819056159:role/AWSBatchServiceRole')
parser.add_argument("--image-id", help="image id", type=str, default='ami-a62f07c3')
parser.add_argument("--key-pair", help="ec2 key pair", type=str, default='Cook_Anaconda')
parser.add_argument("--max-nodes", help='max number of nodes', type=int, default=4)
args = parser.parse_args()

spin = ['-', '/', '|', '\\', '-', '/', '|', '\\']

def create_compute_environment(computeEnvironmentName, instanceType, maxVCpus, imageId, serviceRole, instanceRole,
                               subnets, securityGroups, keyPair):
    response = batch.create_compute_environment(
        computeEnvironmentName=computeEnvironmentName,
        type='MANAGED',
        serviceRole=serviceRole,
        computeResources={
            'type': 'EC2',
            'imageId': imageId,
            'minvCpus': 0,
            'maxvCpus': maxVCpus,
            'desiredvCpus': 4,
            'instanceTypes': [instanceType],
            'subnets': subnets,
            'securityGroupIds': securityGroups,
            'ec2KeyPair': keyPair,
            'instanceRole': instanceRole
        }
    )

    spinner = 0
    while True:
        describe = batch.describe_compute_environments(computeEnvironments=[computeEnvironmentName])
        computeEnvironment = describe['computeEnvironments'][0]
        status = computeEnvironment['status']
        if status == 'VALID':
            print('\rSuccessfully created compute environment %s' % (computeEnvironmentName))
            break
        elif status == 'INVALID':
            reason = computeEnvironment['statusReason']
            raise Exception('Failed to create compute environment: %s' % (reason))
        print '\rCreating compute environment... %s' % (spin[spinner % len(spin)]),
        sys.stdout.flush()
        spinner += 1
        time.sleep(1)

    return response


def create_job_queue(computeEnvironmentName):
    jobQueueName = computeEnvironmentName + '_queue'
    response = batch.create_job_queue(jobQueueName=jobQueueName,
                                      priority=0,
                                      computeEnvironmentOrder=[
                                          {
                                              'order': 0,
                                              'computeEnvironment': computeEnvironmentName
                                          }
                                      ])

    spinner = 0
    while True:
        describe = batch.describe_job_queues(jobQueues=[jobQueueName])
        jobQueue = describe['jobQueues'][0]
        status = jobQueue['status']
        if status == 'VALID':
            print('\rSuccessfully created job queue %s' % (jobQueueName))
            break
        elif status == 'INVALID':
            reason = jobQueue['statusReason']
            raise Exception('Failed to create job queue: %s' % reason)
        print '\rCreating job queue... %s' % (spin[spinner % len(spin)]),
        sys.stdout.flush()
        spinner += 1
        time.sleep(1)

    return response


def register_job_definition(jobDefName, image, unitVCpus, unitMemory):
    response = batch.register_job_definition(jobDefinitionName=jobDefName,
                                             type='container',
                                             containerProperties={
                                                 'image': image,
                                                 'vcpus': 1,
                                                 'memory': 2048,
                                                 'privileged': True,
                                                 'volumes': [
                                                     {
                                                         'host': {
                                                             'sourcePath': '/var/lib/nvidia-docker/volumes/nvidia_driver/latest'
                                                         },
                                                         'name': 'nvidia'
                                                     }
                                                 ],
                                                 'mountPoints': [
                                                     {
                                                         'containerPath': '/usr/local/nvidia',
                                                         'readOnly': False,
                                                         'sourceVolume': 'nvidia'
                                                     }
                                                 ],
                                                 "command": ["./pcmdpy_run.sh",
                                                             "Ref::config_file",
                                                             "Ref::data_file",
                                                             "Ref::results_file",
                                                             "Ref::stdout_file",
                                                             "Ref::stderr_file"
                                                         ]
                                             })
    print 'Created job definition %s' % response['jobDefinitionName']
    return response


def main():
    computeEnvironmentName = args.compute_environment
    jobDefName = args.jobdef_name
    imageId = args.image_id
    serviceRole = args.service_role
    instanceRole = args.instance_role
    subnets = args.subnets.split(",")
    securityGroups = args.security_groups.split(",")
    keyPair = args.key_pair

    # vcpus and memory in a p2.xlarge
    unitMemory = 61000
    unitVCpus = 4
    
    maxMemory = args.max_nodes * unitMemory
    maxVCpus = args.max_nodes * unitVCpus

    create_compute_environment(computeEnvironmentName=computeEnvironmentName,
                               instanceType='p2.xlarge',
                               maxVCpus=maxVCpus,
                               imageId=imageId,
                               serviceRole=serviceRole,
                               instanceRole=instanceRole,
                               subnets=subnets,
                               securityGroups=securityGroups,
                               keyPair=keyPair)

    create_job_queue(computeEnvironmentName)
    register_job_definition(jobDefName=jobDefName, image='bacook17/pcmdpy_gpu', unitVCpus=unitVCpus, unitMemory=unitMemory)

    print 'Successfully created batch entities for %s (compute environment, job queue, job definition)'%computeEnvironmentName


if __name__ == "__main__":
    main()
