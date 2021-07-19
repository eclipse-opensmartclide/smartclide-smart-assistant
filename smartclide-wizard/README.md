# SmartCLIDE wizard

<div align="center">
  <img src="https://github.com/AIRInstitute/smartclide-wizard/blob/main/docs/static/logo.png?raw=true" height="300px" hspace="20" />
</div>

Component flow generation tool.

## Usage 
1. Install package with `python -m pip install . --upgrade`.
2. Generate a BPMN diagram.
3. Generate the components with `kafkawizard -i bpmn.xml -o output_folder -a the_author -m the_author_mail@mail.mail -b kafka_broker_address:9092`.

## How to create diagrams supported by the wizard
1. create the components with blocks.
2. join the components with arrow (ensure arrows are attached).
3. give name to the components.

**Note**: when multiple components read from one topic, if all are joined with normal arrows, all messages are received in each one. But in the ones joined with dotted arrows, the messages are distributed among all.

<p align="center">
  <img src="https://github.com/AIRInstitute/smartclide-wizard/blob/main/tests/test.png">
</p>

####  Additional information
The code generation of this tool, is based in the [kafka-cookie](https://github.com/GandalFran/kafka-cookie) template for [cookiecutter](https://github.com/cookiecutter/cookiecutter).
