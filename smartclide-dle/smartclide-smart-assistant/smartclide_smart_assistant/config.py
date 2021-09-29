#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.

import os

# api config
PORT = 5000
HOST = '0.0.0.0'
URL_PREFIX = '/smartclide/v1'
DEBUG_MODE = True
DLE_BASE_URL = 'http://smartclide.ddns.net:5001/smartclide/v1/dle'

# mongodb configuration
MONGO_DB = 'smartclide-smart-assistant'
MONGO_IP = 'localhost'
MONGO_PORT = 27017
MONGO_URI = f'mongodb://{MONGO_IP}:{MONGO_PORT}/{MONGO_DB}'