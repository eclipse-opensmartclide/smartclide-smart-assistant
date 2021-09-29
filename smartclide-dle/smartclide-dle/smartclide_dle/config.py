#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.

import os

# api config
PORT = 5001
HOST = '0.0.0.0'
URL_PREFIX = '/smartclide/v1'
DEBUG_MODE = True

# mongodb configuration
MONGO_DB = 'smartclide-smart-assistant'
MONGO_IP = 'localhost' 
MONGO_PORT = 27017
MONGO_URI = f'mongodb://{MONGO_IP}:{MONGO_PORT}/{MONGO_DB}'