#!/usr/bin/env python3

import os

# Third party
from setuptools import find_packages, setup
from apu.setup import setversion, Module

project_name="craterdata"

setversion(os.path.abspath(os.path.dirname(__file__)),
           f'{project_name}/__init__.py')

from craterdata import __author__, __version__, __email__

setup(
    name=project_name,
    version='.'.join([str(v) for v in __version__]),
    author=__author__,
    author_email=__email__,
    py_modules=[project_name],
    packages=find_packages(),
    include_package_data=True,
    project_urls={
        'Documentations':
        'https://github.com/afeldman/CraterData',
        'Source': 'https://github.com/afeldman/CraterData.git',
        'Tracker': 'https://github.com/afeldman/CraterData/issues'
    },
    install_requires=Module.load_requirements("requirements.txt"),
)
