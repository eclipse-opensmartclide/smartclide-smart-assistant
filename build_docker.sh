#!/bin/bash

# Copyright 2021 AIR Institute
# See LICENSE for details.

docker build --tag ghcr.io/eclipse-researchlabs/smartclide/smart-assistant:$(date +'%Y-%m-%d') .
docker push ghcr.io/eclipse-researchlabs/smartclide/smart-assistant:$(date +'%Y-%m-%d')