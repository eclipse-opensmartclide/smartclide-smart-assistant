#!/bin/bash

# Copyright 2021 AIR Institute
# See LICENSE for details.

sudo python3 -m pip install pip spacy --upgrade
sudo python3 -m pip install . --upgrade
sudo python3 -m spacy install en_core_web_md
sudo smartclide-dle