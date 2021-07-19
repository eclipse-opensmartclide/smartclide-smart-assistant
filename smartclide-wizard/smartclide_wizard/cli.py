#!usr/bin/python

# Copyright 2021 AIR Institute
# See LICENSE for details.

import argparse
from . import Wizard


# build argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-file', required=True, dest='input_file', type=str, help='Path to the file containing the XML diagram.')
parser.add_argument('-o','--output-folder', required=False, default='.', dest='output_folder', type=str, help='Path to the folder where final results will be dumped. If not provided the current directory will be used for that purpose.')
parser.add_argument('-b','--kafka-broker', required=False, default='localhost:9092', dest='broker', type=str, help='Host and port of the Apache Kafka broker.')
parser.add_argument('-a','--author', required=False, default='unknown', dest='author', type=str, help='Author of the package.')
parser.add_argument('-m','--mail', required=False, default='unknown@unknown.unknown', dest='email', type=str, help='Email of the author of the package.')
parser.add_argument('-r','--kafka-num-retries', default=3, required=False, dest='num_tries', type=str, help='Number of retries on receive and send on Kafka.')
parser.add_argument('-t','--kafka-timeout', default=3000, required=False, dest='delivery_timeout', type=str, help='Timeout on send and receive in Kafka.')


def main():
	# retrieve args
	args = parser.parse_args()

	# parse
	email = args.email
	broker = args.broker
	author = args.author
	num_tries = args.num_tries
	file_location = args.input_file
	final_location = args.output_folder
	delivery_timeout = args.delivery_timeout

	# generate code
	Wizard.generate(
		 diagram_file=file_location
		,final_location=final_location
		,broker=broker
		,author=author
		,email=email
		,num_tries=num_tries
		,delivery_timeout=delivery_timeout
	)

