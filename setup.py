#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   setup.py
"""

import numpy as np
#import versioneer
from setuptools import setup, find_packages, Extension
import os
import sys

setup(
    name = "MPITools",
    package_data = {'': ['ChangeLog.md']},
    include_package_data = True,
    packages = find_packages(),
    python_requires='>3.5.0',
    install_requires=['runtests',
                      'numpy',
                      'pytest',
                      ],
    # metadata for upload to PyPI
    author = "Antoine Sanner",
    author_email = "antoine.sanner@imtek.uni-freiburg.de",
    description = "numerical tools for mpi parallelized code",
    license = "MIT",
    test_suite = 'tests',
)