#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.

import ssl
from flask_cors import CORS
from flask import Flask, Blueprint, redirect, request

from smartclide_smart_assistant import config
from smartclide_smart_assistant.api.v1 import api
from smartclide_smart_assistant.api import namespaces_v1
from smartclide_smart_assistant.core import cache, limiter
from smartclide_smart_assistant.mom_connector.rabbitmq_connector import BackgroundAPIRabbitMQConsumer


app = Flask(__name__)

VERSION = (1, 0)
AUTHOR = 'AIR Institute'


def get_version():
    """
    This function returns the API version that is being used.
    """

    return '.'.join(map(str, VERSION))


def get_authors():
    """
    This function returns the API's author name.
    """

    return str(AUTHOR)


__version__ = get_version()
__author__ = get_authors()
    

@app.route('/')
def register_redirection():
    """
    Redirects to dcoumentation page.
    """

    return redirect(f'{request.url_root}/{config.URL_PREFIX}', code=302)


def initialize_app(flask_app):
    """
    This function initializes the Flask Application, adds the namespace and registers the blueprint.
    """

    CORS(flask_app)

    v1 = Blueprint('api', __name__, url_prefix=config.URL_PREFIX)
    api.init_app(v1)

    limiter.exempt(v1)
    cache.init_app(flask_app)

    flask_app.register_blueprint(v1)
    flask_app.config.from_object(config)

    for ns in namespaces_v1:
        api.add_namespace(ns)


def initialize_mom():

    c = BackgroundAPIRabbitMQConsumer(
            host=config.rabbitmq_host
            ,channel_endpoint_mappings=config.channel_endpoint_mappings
        )
    try:
        c.start()
    except Exception as e:
        print(f'Unable to connect to MoM: {e}')


def main():
    separator_str = ''.join(map(str, ["=" for i in range(80)]))
    print(separator_str)
    initialize_mom()
    print(separator_str)
    initialize_app(app)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG_MODE)


if __name__ == '__main__':
    main()