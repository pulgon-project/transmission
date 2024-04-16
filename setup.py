# Copyright 2018 Jesús Carrete Montaña <jesus.carrete.montana@tuwien.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import sys
import os
import os.path

import setuptools
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

try:
    import numpy as np
except ImportError:
    print(
        """Error: numpy is not installed.
Please install it using your package manager or with "pip install numpy".
""",
        file=sys.stderr,
        end="")
    sys.exit(1)

# Extra header and library dirs for compiling and linking the C++ source files.
INCLUDE_DIRS = []
LIBRARY_DIRS = []

eigen_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "external", "eigen-eigen-3215c06819b9"))

extensions = cythonize(
    [
        Extension(
            "decimation.frontend",
            ["decimation/frontend.pyx", "decimation/backend.cpp"],
            language="c++",
            include_dirs=INCLUDE_DIRS + [np.get_include(), eigen_dir],
            library_dirs=LIBRARY_DIRS,
            runtime_library_dirs=LIBRARY_DIRS)
    ],
    annotate=True)

setup(
    name="decimation",
    description="C++ implementation of the decimation routines",
    version="0.0.1",
    author="Jesús Carrete Montaña",
    author_email="jesus.carrete.montaña@tuwien.ac.at",
    license="Apache v2",
    packages=["decimation"],
    ext_modules=extensions,
    install_requires=["numpy", "scipy", "matplotlib", "ase"],
    python_requires=">=3.6")
