#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************

'''Set up the package configuration, including logging'''

import tempfile
import logging
import os

import coloredlogs
import yaml
import yaconfig

temp_dir = tempfile.gettempdir()

# Configuration variables
metaconfig = yaconfig.MetaConfig(
    yaconfig.Variable("store", type=str, default=temp_dir, help="Path to the local storage of the models."),
    yaconfig.Variable("celery", type=str, default="",
                      help="If not empty, celery is set as real queue system. Otherwise, eager mode is used."),
    yaconfig.Variable("celery_broker", type=str, default="amqp://", help="The celery broker"),
    yaconfig.Variable("auth", type=str, default="",
                      help="Authentication token for the server. Client request must set X-IAMODELER-AUTH to this token in their headers."),
    yaconfig.Variable("log", type=str, default="logging.yaml", help="Path to a yaml file with a logging configuration"),
)

# Get a default configuration, which will be overridden next
config = yaconfig.Config(metaconfig)

# Load from environment variables
config.load_environment("IAMODELER_")


# Setup logging
def setup_logging(default_path='logging.yaml', default_level=logging.INFO):
    """

    Args:
        default_path (str): A path to a yaml file with the logging configuration.
                            Will be overriden by the log config if available.
        default_level (int): A level of logging (e.g., logging.INFO) used in case an error occurs.

    Returns:

    """
    global config
    value = os.getenv(config["log"], None)
    if value:
        path = value
    else:
        path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                logging.config.dictConfig(yaml.safe_load(f.read()))
                coloredlogs.install()
            except Exception as e:
                print(e)
                print('Error in logging configuration file. Using the default settings.')
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        print('Unable to load a logging configuration file. Using the default settings.')


logger = logging.getLogger(__name__)
setup_logging()
if config["store"] == temp_dir:
    logger.warning("IAMODELER_STORE is set to the temporal directory %s. "
                   "This might result in MODEL LOOSE." % config["store"])

if config["auth"]:
    logger.info("Token authentication is ON")  # Not exposing the value in logs seems a good practice
else:
    logger.info("Token authentication is OFF")
