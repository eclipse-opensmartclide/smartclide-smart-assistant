#!/usr/bin/env python3

import pycbr
import pickle
from pycbr.models import *
from pathlib import Path
import pandas as pd
import logging

# Define a case base from all BPMN files
data = list()

with open('gherkins/example.feature', 'r') as file:
    gherkin_text = file.read()

# for f in Path("bpmn").rglob('*.bpmn'):
#     process = pycbr.builder.Process()
#     try:
#         with open(f, 'r') as file:
#             bpmn_text = file.read().replace('\n', '')
#         flow = process.generate_text_from_bpmn(bpmn_text)
#         if flow == "":
#             continue
#         data.append(
#             {"name":f.name, 
#             "text": flow,
#             "gherkins": [gherkin_text]})
#     except Exception as e:
#         # Don not save invalid flows
#         pass

# Create a CBR instance
cbr = pycbr.CBR([],"ghrkn_recommendator","smartclide.ddns.net")

with open('bpmn/user.bpmn', 'r') as file:
    bpmn_text = file.read()

# PREDICT EXAMPLE
print(cbr.recommend(bpmn_text))
