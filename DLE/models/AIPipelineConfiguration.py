#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from configparser import ConfigParser

# Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")
# Get the _PATH_ROOT_
_PATH_ROOT_ = config_object["Path"]["root"]



class AIPipelineConfiguration:
        epoch = 23
        defaultDatasetsFolder = "DLE/data/";
        defaultTrainedModelPath = "DLE/trained_models/";

