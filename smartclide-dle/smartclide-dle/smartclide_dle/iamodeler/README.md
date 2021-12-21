# iamodeler Python package

A Python package to provide an API for building and using machine learning models for the SmartCLIDE platform.

## Installation

You probably to set up and use a virtualenv:

```
# Prepare a clean virtualenv and activate it
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
```

Remember to activate it whenever you are working with the package.

To install a **development** version clone the repo, cd to the directory and:

```
pip install -e .
```

Once installed, the *development* flask server might be started with the command:

```
iamodeler
```

For real deployment, gunicorn might be used instead:

```
pip install gunicorn
unicorn --workers 4 --bind 0.0.0.0:5000 --timeout 600 iamodeler.server:app
```

To use a celery queue system (see configuration below), a celery broker
like [RabbitMQ](https://www.rabbitmq.com/download.html) must also be installed.

With RabbitMQ installed and running, start the queue system by running:

```
celery -A iamodeler.tasks.celery worker
```

Note the gunicorn timeout parameter does not affect the celery queues.

In **Windows**, the default celery pool might not work. You might try to add `--pool=eventlet` to run it.

For instructions to set up a **cluster** to scale the number of workers, see [this file](docs/cluster.md).

## Configuration

Configuration is done with environment variables.

| Variable                 | Description                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| IAMODELER\_STORE         | Path to the local storage of the models. Defaults to a temporal directory.                                    |
| IAMODELER\_CELERY        | If set and not empty, use a local Celery queue system.                                                        |
| IAMODELER\_CELERY\_BROKER| Address of the Celery broker.                                                                                 |
| IAMODELER\_AUTH          | Authentication token for the server. Client request must set X-IAMODELER-AUTH to this token in their headers. |
| IAMODELER\_LOG           | A path to a yaml logging configuration file. Defaults to logging.yaml                                         |

The paths are *relative to the CWD*, provide full paths when needed.

Pro-tip: A .env file can be used installing the python-dotenv package.

An example of logging configuration file is provided in the root of the repo.

## Documentation

The REST API documentation can be seen in Swagger with a deployed system.



