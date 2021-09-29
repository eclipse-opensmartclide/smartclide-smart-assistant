#!/usr/bin/env python3

import pycbr
import pickle
from pycbr.models import *
from pathlib import Path
import pandas as pd
import logging
import warnings

warnings.simplefilter("ignore")


# Define a case base from all BPMN files
data = list()

with open('demos/bpmn/gherkins/example.feature', 'r') as file:
    gherkin_text = file.read()

for f in Path("bpmn").rglob('*.bpmn'):
    process = pycbr.builder.Process()
    try:
        with open(f, 'r') as file:
            bpmn_text = file.read().replace('\n', '')
        flow = process.generate_text_from_bpmn(bpmn_text)
        if flow == "":
            continue
        data.append(
            {"name":f.name, 
            "text": flow,
            "gherkins": [gherkin_text]})
    except Exception as e:
        # Don not save invalid flows
        pass
try:
    case_base = pycbr.casebase.MongoCaseBase([],"ghrkn_recommendator" , "localhost" )
except Exception as e:
    print(e)

# Define the set of similarity functions
recovery = pycbr.recovery.Recovery([
    ("text", TextAttribute())
],
    algorithm="brute")
# Define the aggregation method
aggregation = pycbr.aggregate.MajorityAggregate("gherkins")

# Create a CBR instance
cbr = pycbr.CBR(case_base, recovery, aggregation)

with open('demos/bpmn/bpmn/user.bpmn', 'r') as file:
    bpmn_text = file.read()

# PREDICT EXAMPLE
# print(cbr.recommend(bpmn_text))

print('done')