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



def handle400error(ns, message):
    """
    Function to handle a 400 (bad arguments code) error.
    """

    return ns.abort(400, status=message, statusCode="400")


def handle404error(ns, message):
    """
    Function to handle a 404 (not found) error.
    """

    return ns.abort(404, status=message, statusCode="404")


def handle500error(ns, message=None):
    """
    Function to handle a 500 (unknown) error.
    """

    if message is None:
        message = "Unknown error, please contact administrator."

    return ns.abort(500, status=message, statusCode="500")
