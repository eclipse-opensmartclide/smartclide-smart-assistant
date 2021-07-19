#!usr/bin/python

# Copyright 2021 AIR Institute
# See LICENSE for details.

import os
import shutil
from smartclide_wizard import Wizard


def test_wizard():
	FINAL_LOCATION = './packages'

	# check if exists folder
	os.mkdir(FINAL_LOCATION)
	if os.path.exists(FINAL_LOCATION) and os.path.isdir(FINAL_LOCATION):
		shutil.rmtree(FINAL_LOCATION)

	# generate
	Wizard.generate('test.xml', 
		broker='localhost:9092', 
		author='example', 
		email='example@air+institute.org', 
		final_location=FINAL_LOCATION
	)
	
	# delete
	shutil.rmtree(FINAL_LOCATION)


def test_command_cli():
	FINAL_LOCATION = './packages_cli'

	# check if exists folder
	os.mkdir(FINAL_LOCATION)
	if os.path.exists(FINAL_LOCATION) and os.path.isdir(FINAL_LOCATION):
		shutil.rmtree(FINAL_LOCATION)

	# generate
	os.system(f'smartclide_wizard -i test.xml -o {FINAL_LOCATION} -a example -m example@air+institute.org -b localhost:9092')

	# delete
	shutil.rmtree(FINAL_LOCATION)
