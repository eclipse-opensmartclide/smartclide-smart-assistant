## SmartCLIDE DLE

Flask-restx API developed to serve SmartCLIDE Smart Assistant.

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