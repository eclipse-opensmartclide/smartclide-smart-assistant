# [smartclide-dle](https://github.com/eclipse-researchlabs/smartclide-smart-assistant/tree/main/smartclide-dle/smartclide-dle)
Flask-restx API developed to serve SmartCLIDE DLE (Deep Learning Engine), port 5001.

## SmartCLIDE DLE

Flask-restx API developed to serve SmartCLIDE DLE (Deep Learning Engine).

## Install requirements

```bash
sudo apt update
sudo apt install python3 python3-pip nodejs -y
sudo npm install -g pm2
```

Also is needed to install some python dependencies manually:

```bash
sudo python3 -m pip install tensorflow nlpaug sentence_transformers
sudo python3 -m pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
sudo pyton3 -m pip install git+https://github.com/Dih5/zadeh
```

## Configuration

The application configuration is set via enviroment variables:

- `DLE_API_PORT`: Port to bind to (default: `5001`). 
- `DLE_API_BIND`: Address to bind to (default: `0.0.0.0`).
- `DLE_MONGODB_PORT`: MongoDB database to connect to (default:`27017`).
- `DLE_MONGODB_HOST`: MongoDB database to connect to (default: `localhost`).
- `DLE_MONGODB_DB`: MongoDB database to connect to (default: `smartclide-smart-assistant`).


## Run application

Application can be launched with the launch script:

```bash
sudo bash launch.bash
```

Or using PM2:

```bash
sudo pm2 start pm2.json
```

Note: if the script `launch.bash` doesn't works, you can use `launch2.bash` instead.

# [smartclide-smart-assistant](https://github.com/eclipse-researchlabs/smartclide-smart-assistant/tree/main/smartclide-dle/smartclide-smart-assistant)
Flask-restx API developed to serve SmartCLIDE Smart Assistant, port 5000.
## Install requirements

```bash
sudo apt update
sudo apt install python3 python3-pip nodejs -y
sudo npm install -g pm2
```

Also is needed to install some python dependencies manually:

```bash
sudo python3 -m pip install tensorflow nlpaug sentence_transformers
sudo python3 -m pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
sudo pyton3 -m pip install git+https://github.com/Dih5/zadeh
```


## Configuration

The application configuration is set via enviroment variables:

- `SA_API_PORT`: Port to bind to (default: `5001`). 
- `SA_API_BIND`: Address to bind to (default: `0.0.0.0`).
- `SA_MONGODB_PORT`: MongoDB database to connect to (default:`27017`).
- `SA_MONGODB_HOST`: MongoDB database to connect to (default: `localhost`).
- `SA_MONGODB_DB`: MongoDB database to connect to (default: `smartclide-smart-assistant`).
- `DLE_BASE_URL`: Base URL for DLE connection (default: `http://smartclide.ddns.net:5001/smartclide/v1/dle`).
- `RABBITMQ_HOST`: RabbitMQ host to connect to (default: `localhost`).
- `RABBITMQ_USER`: RabbitMQ user to connect to (default: ``).
- `RABBITMQ_PASSWORD`: RabbitMQ password to connect to (default: ``).
- `RABBITMQ_MAPPINGS`: RabbitMQ mappings between channels and API's endpoint namespaces to connect to (default: `{
	    'acceptance_tests_queue': 'http://smartclide.ddns.net:5000/smartclide/v1/smartassistant/acceptance',
	    'bpmn_item_recommendation_queue': 'http://smartclide.ddns.net:5000/smartclide/v1/smartassistant/bpmnitemrecommendation',
	    'code_generation_queue': 'http://smartclide.ddns.net:5000/smartclide/v1/smartassistant/codegen',
	    'code_repo_recommendation_queue': 'http://smartclide.ddns.net:5000/smartclide/v1/smartassistant/coderepo',
	    'enviroment_queue': 'http://smartclide.ddns.net:5000/smartclide/v1/smartassistant/enviroment'
	}`).

## Run application

Application can be launched with the launch script:

```bash
sudo bash launch.bash
```

Or using PM2:

```bash
sudo pm2 start pm2.json
```

Note: if the script `launch.bash` doesn't works, you can use `launch2.bash` instead.

# [smartclide-dle-models](https://github.com/eclipse-researchlabs/smartclide-smart-assistant/tree/main/smartclide-dle-models)
Repository with the models to be served by the DLE. It has automatic deployment so that being structured as packages, each change can be deployed automatically in the server where the DLE and SmartAssistant are deployed.

- **cbr-gherkin-recommendation**: gherkin recommendation from BPMN files using case-based reasoning.
- **smartclide-service-classification-autocomplete**: service classification and code autocomplete based in markov-chains.

# [smartclide-template-code-generation](https://github.com/eclipse-researchlabs/smartclide-smart-assistant/tree/main/smartclide-template-code-generation)
Component flow generation tool.


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


# Deploy
This docker package contains two REST services that need to be executed as daemons or in a similar way, for this purpose you can use the attached docker-compose.yml file.

