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


import random
from typing import List
from smartclide_wizard import BPMNParser, JavaCodeGenerator


class CodeTemplateGeneration:

    def generate(self, bpmn_file:str) -> str:
        
        parser = BPMNParser()
        code_generator = JavaCodeGenerator()

        # parse diagram
        workflow = parser.parse(bpmn_file)

        # generate code
        code = code_generator.generate(workflow)

        return code
