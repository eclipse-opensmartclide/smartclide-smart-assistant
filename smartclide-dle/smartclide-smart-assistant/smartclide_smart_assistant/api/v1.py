#!/usr/bin/python3
#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************


from flask_restx import Api

api = Api(version='1.0',
		  title='SmartCLIDE Smart Assistant API',
		  description="Flask restx API for serving SmartCLIDE Smart Assistant models")
