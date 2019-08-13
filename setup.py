#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To use: python setup.py install
#        python setup.py clean

import os
import re
import codecs
from setuptools import setup, Command, find_packages
from setuptools.command.install import install

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


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

        
setup(
    name='pcmdpy',
    version=find_version('pcmdpy', '__init__.py'),
    author='Ben Cook',
    author_email='bcook@cfa.harvard.edu',
    packages=find_packages(),
    url='https://github.com/bacook17/pcmdpy',
    license='LICENSE',
    description="""Tools for modelling crowded-field photometry using the
       Pixel Color-Magnitude Diagram technique""",
    package_data={'pcmdpy': ['isochrones/MIST_v1.2/*',
                             'isochrones/MIST_v1.2_rot/*',
                             'instrument/PSFs/*.fits', 'simulation/*.c']},
    scripts=['bin/run_pcmdpy', 'bin/pcmd_integrate'],
    include_package_data=True,
    cmdclass={'clean': CleanCommand},
    install_requires=[
        'numpy', 'scipy', 'pandas', 'matplotlib', 'astropy', 'dynesty',
        'corner', 'sklearn', 'memory_profiler', 'pyregion', 'drizzlepac',
        'tqdm', 'sep'
    ],
    python_requires='>=3',
    extras_require={"GPU": ['pycuda']},
    classifiers=[
        "Programming Language :: Python :: 3",
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ]
)
