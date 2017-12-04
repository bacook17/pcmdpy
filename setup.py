#!/usr/bin/env python
# -*- coding: utf-8 -*-

#To use: python setup.py install

import os
import sys

try:
    from setuptools import setup, Command
    setup
except ImportError:
    from distutils.core import setup
    setup

class CleanCommand(Command):
    """Custom clean command to tidy up the project root.
    From https://stackoverflow.com/questions/3779915/why-does-python-setup-py-sdist-create-unwanted-project-egg-info-in-project-r"""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')
    
setup(
    name='pcmdpy',
    version='0.0.1',
    author='Ben Cook',
    author_email='bcook@cfa.harvard.edu',
    packages=['pcmdpy'],
    url='https://github.com/bacook17/pcmdpy',
    license='LICENSE',
    description='Tools for modelling crowded-field photometry using the Pixel Color-Magnitude Diagram technique',
    package_data={'pcmdpy':['isoc_MIST_v1.1/*.iso.cmd', 'psf/*.psf']},
    include_package_data=True,
    cmdclass={'clean' : CleanCommand ,},
    #install_requires=['numpy','scipy','pandas','matplotlib','emcee'],
)
