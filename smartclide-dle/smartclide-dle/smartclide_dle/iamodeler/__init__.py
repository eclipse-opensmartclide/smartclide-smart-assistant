#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************

__author__ = """Dih5"""
__copyright__ = "Copyright 2021, AIR-Institute"
__version__ = '0.1.0'

from .server import iamodeler_ns, sources_namespace, plot_namespace, supervised_namespace, unsupervised_namespace

iamodeler_ns_v1 = [iamodeler_ns, sources_namespace, plot_namespace, supervised_namespace, unsupervised_namespace]
