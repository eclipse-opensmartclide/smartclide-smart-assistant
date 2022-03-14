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
- `SMART_ASSISTANT_BASE_URL`: Base URL for Smart Assistant RabbitMQ connection (default: `http://smartclide.ddns.net:5000/smartclide/v1/smartassistant`).
- `RABBITMQ_HOST`: RabbitMQ connection string host (default: `localhost`).
- `RABBITMQ_PORT`: RabbitMQ connection string port (default: `5672`).
- `RABBITMQ_USER`: RabbitMQ connection string user (default: `user`).
- `RABBITMQ_PASSWORD`: RabbitMQ connection string password (default: `password`).
- `RABBITMQ_MAPPINGS`: RabbitMQ mappings between queue and API's endpoint to connect to. (default: `{
	    'acceptance_tests_queue': '{SMART_ASSISTANT_BASE_URL}/acceptance',
	    'bpmn_item_recommendation_queue': '{SMART_ASSISTANT_BASE_URL}/bpmnitemrecommendation',
	    'code_generation_queue': '{SMART_ASSISTANT_BASE_URL}/codegen',
	    'code_repo_recommendation_queue': '{SMART_ASSISTANT_BASE_URL}/coderepo',
	    'enviroment_queue': '{SMART_ASSISTANT_BASE_URL}/enviroment'
	}`). Note: All of them are prefixed with `{SMART_ASSISTANT_BASE_URL}/` before start the connection.

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