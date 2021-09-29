#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.


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