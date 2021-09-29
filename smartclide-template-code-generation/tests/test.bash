#!/bin/bash

# Copyright 2021 AIR Institute
# See LICENSE for details.

cd ..
sudo python3 -m pip install . --upgrade
cd tests
sudo smartclide_wizard -i test.bpmn
sudo smartclide_wizard -i test2.bpmn