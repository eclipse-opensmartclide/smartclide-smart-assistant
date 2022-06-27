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
import logging

from .config import config

logger = logging.getLogger(__name__)


def ensure_dir_exists(path):
    """
    Ensure a directory exists, creating it if needed.

    Args:
        path (str): The path to the directory.

    """
    if path:  # Empty dir (cwd) always exists
        try:
            # Will fail either if exists or unable to create it
            os.makedirs(path)
        except OSError:
            # Also raised if the path exists
            pass

        if not os.path.exists(path):
            # There was an error on creation, so make sure we know about it
            raise OSError("Unable to create directory " + path)


# Ensure the structure exists
ensure_dir_exists(config["store"])
for folder in ["sources", "models", "unsupervised"]:
    ensure_dir_exists(os.path.join(config["store"], folder))


def get_file_path(name, path=None):
    """
    Get the path of a file in the local storage

    Args:
        name (str): Name of the file
        path (list of str): List of intermediate folders.

    Returns:
        str: Path to the file

    """
    if path is None:
        path = []
    if isinstance(path, str):
        # Prevent (name, "models") -> m/o/d/e/l/s/name
        logger.warning("Path %s implicitly converted to a list with one string" % path)
        path = [path]
    return os.path.join(*([config["store"]] + path + [name]))
