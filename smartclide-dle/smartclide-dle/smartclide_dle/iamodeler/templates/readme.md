# SmartCLIDE - Exported model instructions
This folder includes the files needed to deploy your model as a web service.

The recommended procedure to run the model is to install the dependencies in a clean virtualenv as follows:
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

You can run it with the Flask development server executing ```python server.py```, which will make it available at [localhost:5000](http://localhost:5000/).

Production deployments can be easily done using the standard procedure. Please check the official Flask guide [here](https://flask.palletsprojects.com/deploying).
