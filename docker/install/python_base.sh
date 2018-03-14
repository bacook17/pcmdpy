#!/usr/bin/env bash

# get miniconda as conda source
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
/bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
rm Miniconda3-latest-Linux-x86_64.sh
export PATH="$HOME/miniconda/bin:$PATH"
export PYTHONPATH="$HOME/miniconda/lib/python3.6/site-packages"

# install base requirements
conda install -y pip

# install awscli
pip install awscli --upgrade --user

export PATH="${HOME}/.local/bin/:${PATH}"
