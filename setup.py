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
    name = "PyLBGFS",
    package_data = {'': ['ChangeLog.md']},
    include_package_data = True,
    packages = find_packages(),
    # metadata for upload to PyPI
    author = "Antoine Sanner",
    author_email = "antoine.sanner@imtek.uni-freiburg.de",
    description = "Parallel implementation of LBGFS",
    license = "MIT",
    test_suite = 'tests',
    python_requires='>3.5.0'
)