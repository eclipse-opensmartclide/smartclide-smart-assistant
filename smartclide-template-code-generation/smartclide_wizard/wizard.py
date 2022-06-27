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



import os
from . import BPMNParser, JavaCodeGenerator


class Wizard:

	@classmethod
	def generate(cls, input_file):

		# read input file
		with open(input_file,'r') as f:
			file_content = f.read()

		# parse diagram
		parser = BPMNParser()
		workflow = parser.parse(file_content)

		# build components
		code_generator = JavaCodeGenerator()
		class_generated = list(code_generator.generate(workflow))
