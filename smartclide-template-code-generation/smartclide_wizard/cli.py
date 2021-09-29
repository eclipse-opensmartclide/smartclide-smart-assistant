#!usr/bin/python

# Copyright 2021 AIR Institute
# See LICENSE for details.

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