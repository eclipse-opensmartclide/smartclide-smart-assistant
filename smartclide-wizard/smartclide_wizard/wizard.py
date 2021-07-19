#!usr/bin/python

# Copyright 2021 AIR Institute
# See LICENSE for details.


import os
from . import BPMNParser, PackageBuilder


class Wizard:

	@classmethod
	def generate(cls, diagram_file, final_location, **kwargs):

		# create final location if not exists
		if not os.path.exists(final_location):
		    os.makedirs(final_location)

		# parse diagram
		parser = BPMNParser()
		components = parser.parse(diagram_file, **kwargs)

		# build components
		package_builder = PackageBuilder(**kwargs)
		package_builder.build_multiple(components, final_location=final_location, **kwargs)