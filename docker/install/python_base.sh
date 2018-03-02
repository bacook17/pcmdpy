#!/usr/bin/env bash

# get miniconda as conda source
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
/bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
rm Miniconda3-latest-Linux-x86_64.sh
export PATH="$HOME/miniconda/bin:$PATH"
export PYTHONPATH="$HOME/miniconda/lib/python3.6/site-packages"

# install base requirements
conda install -y pip numpy scipy matplotlib pandas astropy ipython

# install pycuda (not listed on conda)
# conda install lukepfister pycuda
pip install pycuda --upgrade --user

# install awscli
pip install awscli --upgrade --user

# install dynesty and pcmdpy
git clone ssh://git@github.com/joshspeagle/dynesty.git
cd dynesty && python setup.py install && cd ..

export PATH="${HOME}/.local/bin/:${PATH}"
