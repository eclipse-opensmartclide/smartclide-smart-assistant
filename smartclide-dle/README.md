<!--
   Copyright (C) 2021-2022 AIR Institute
   
   This program and the accompanying materials are made
   available under the terms of the Eclipse Public License 2.0
   which is available at https://www.eclipse.org/legal/epl-2.0/
   
   SPDX-License-Identifier: EPL-2.0
-->
## SmartCLIDE DLE and Smart Asssistant Components

Flask-restx APIs developed to serve SmartCLIDE DLE (Deep Learning Engine) and Smart Assistant.

## Install prerequisites

```bash
sudo apt update
sudo apt install python3 python3-pip nodejs npm -y
sudo npm install -g pm2
```

## Architecture

The models served by the DLE are defined into the following repositories:
- **recommendation of next node in a BPMN file**: https://github.com/AIRInstitute/smartclide-bpnm-poc
- **code skeleton generation via BPMN file**: https://github.com/AIRInstitute/smartclide-template-code-generation
- **various models, beign the most importants the CBR based gherking reccomendation with BPMN file, service classsification and code autocomplete generation with markov chains**: https://github.com/AIRInstitute/smartclide-dle-models

<div align="center">
  <img src="https://github.com/AIRInstitute/smartclide-dle/blob/main/_static/architecture.jpg" hspace="20">
</div>

