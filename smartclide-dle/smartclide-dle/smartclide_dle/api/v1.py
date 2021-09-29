#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.

from flask_restx import Api

api = Api(version='1.0',
		  title='SmartCLIDE DLE API',
		  description="Flask restx API for serving SmartCLIDE DLE models")