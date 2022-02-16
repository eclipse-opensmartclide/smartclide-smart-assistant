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