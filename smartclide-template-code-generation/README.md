<!--
   Copyright (C) 2021-2022 AIR Institute
   
   This program and the accompanying materials are made
   available under the terms of the Eclipse Public License 2.0
   which is available at https://www.eclipse.org/legal/epl-2.0/
   
   SPDX-License-Identifier: EPL-2.0
-->
# SmartCLIDE wizard

Component flow generation tool.

## Installation

First install prerequisites with

```bash
sudo apt update
sudo apt install python3 python3-pip nodejs npm git -y
sudo npm install -g -y postman-code-generators postman-collection
```

Then requirements will be installed and pplication can be launched with the launch script:
```
sudo bash launch.bash
```
If the script `launch.bash` give you problems, then use in the same way the `launch2.bash` script.


## Usage 
1. Install package with `python -m pip install . --upgrade`.
2. Generate a BPMN diagram.
3. Generate the components with `smartclide_wizard -i bpmn.xml`.
