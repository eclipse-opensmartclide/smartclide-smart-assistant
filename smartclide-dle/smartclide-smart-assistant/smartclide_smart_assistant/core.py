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


from flask_caching import Cache
from flask_limiter import Limiter	
from flask_limiter.util import get_remote_address


limiter = Limiter(	
	key_func=get_remote_address,	
	default_limits=["1000 per hour"]	
)

cache = Cache(
	config={
		'CACHE_TYPE': 'simple',
		'CACHE_DEFAULT_TIMEOUT': 1
	}
)
