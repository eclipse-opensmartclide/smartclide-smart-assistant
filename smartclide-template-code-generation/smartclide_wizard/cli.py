#!usr/bin/python
#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************


import argparse
from . import Wizard


# build argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-file', required=True, dest='input_file', type=str, help='Path to the file containing the XML diagram.')

def main():
	# retrieve args
	args = parser.parse_args()

	# parse
	input_file = args.input_file

	# generate code
	Wizard.generate(
		 input_file=input_file
	)
