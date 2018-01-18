#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To use: python setup.py install
#        python setup.py clean

import os

from setuptools import setup, Command
from setuptools.command.install import install


class CustomInstall(install):
    """
    Updates the given installer to create the __init__.py file and
    store the path to the package.
    """
    def run(self):
        install.run(self)
        # create_init(self.install_lib)


class CleanCommand(Command):
    
    """Custom clean command to tidy up the project root.
    From https://stackoverflow.com/questions/3779915"""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


toplevelstr = ("""__all__ = ['priors', 'gpu_utils', 'utils', 'driver',
           'fit_model', 'galaxy', 'instrument','isochrones']

import gpu_utils
import utils
import driver
import fit_model
import galaxy
import instrument
import isochrones
import priors
""")


def create_init(path):
    
    init_file = '{0}pcmdpy/__init__.py'.format(path)
    print(init_file)
    with open(init_file, 'w') as ff:
        ff.write("def install_path():\n")
        ff.write("    return '{0}pcmdpy/'\n".format(path))
        ff.write('\n\n')
        ff.write(toplevelstr)
        

setup(
    name='pcmdpy',
    version='0.0.1',
    author='Ben Cook',
    author_email='bcook@cfa.harvard.edu',
    packages=['pcmdpy'],
    url='https://github.com/bacook17/pcmdpy',
    license='LICENSE',
    description="""Tools for modelling crowded-field photometry using the
       Pixel Color-Magnitude Diagram technique""",
    package_data={'pcmdpy': ['isoc_MIST_v1.1/*', 'psf/*.fits']},
    include_package_data=True,
    cmdclass={'clean': CleanCommand, 'install': CustomInstall},
    install_requires=[
        'astropy==2.0.2', 'dynesty', 'scipy==0.19.1',
        'pandas==0.20.3', 'matplotlib==2.0.2', 'numpy==1.13.1',
    ],
    dependency_links=[
        'git+https://github.com/joshspeagle/dynesty.git@master#egg=dynesty-0'
    ],
    extras_require={"GPU": ['pycuda']},
)
