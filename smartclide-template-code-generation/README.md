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
