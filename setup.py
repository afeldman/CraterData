#!/usr/bin/env python3

import os

# Third party
from setuptools import find_packages, setup

project_name="craterdata"

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
    install_requires=[
        "torchvision",
        "h5py",
        "numpy",
        "Pillow",
        "coloredlogs",
        "fire"
    ],
)
