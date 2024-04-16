#!/usr/bin/env python
# Copyright 2018 Jesús Carrete Montaña <jesus.carrete.montana@tuwien.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path
import sys
import argparse
import math
import subprocess

import numpy as np

parser = argparse.ArgumentParser(description="Sample the whole BZ")
parser.add_argument("poscar_file", help="position file")
parser.add_argument("fc_file", help="force constant file")
parser.add_argument("na", help="divisions along the first axis", type=int)
parser.add_argument("nb", help="divisions along the second axis", type=int)
parser.add_argument("odir", help="output directory")
args = parser.parse_args()

if min(args.na, args.nb) < 1:
    sys.exit("Error: na and nb must be positive.")
if not os.path.isdir(args.odir):
    sys.exit("Error: {} is not a directory".format(args.odit))

for i in range(args.na):
    qa = 2. * np.pi * i / float(args.na)
    for j in range(args.nb):
        qb = 2. * np.pi * j / float(args.nb)
        command = [
            "./transmission.py", args.poscar_file, args.fc_file,
            f"{qa:12.10g}", f"{qb:12.10g}", args.odir
        ]
        subprocess.run(command)
