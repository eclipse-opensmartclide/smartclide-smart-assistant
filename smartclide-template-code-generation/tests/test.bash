#!/bin/bash

# Copyright 2021 AIR Institute
# See LICENSE for details.

cd ..
sudo python3 setup.py clean
sudo python3 -m pip install . --upgrade
cd tests
sudo smartclide_wizard -i test.bpmn
sudo smartclide_wizard -i test2.bpmn
sudo smartclide_wizard -i AdHocProcess.bpmn
sudo smartclide_wizard -i BPMN2-MultiThreadServiceProcess.bpmn
sudo smartclide_wizard -i contactCustomer.bpmn #problems
sudo smartclide_wizard -i Evaluation.bpmn
sudo smartclide_wizard -i Evaluation2.bpmn
sudo smartclide_wizard -i expenses.bpmn
sudo smartclide_wizard -i HumanTask.bpmn
sudo smartclide_wizard -i HumanTaskDeadline.bpmn
sudo smartclide_wizard -i Looping.bpmn
sudo smartclide_wizard -i multipleinstance.bpmn
sudo smartclide_wizard -i requestHandling.bpmn
sudo smartclide_wizard -i SampleChecklistProcess.bpmn
sudo smartclide_wizard -i ScriptTask.bpmn
sudo smartclide_wizard -i travel.bpmn