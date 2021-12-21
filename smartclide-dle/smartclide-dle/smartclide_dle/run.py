#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.

import ssl
from flask_cors import CORS
from flask import Flask, Blueprint, redirect, request

from smartclide_dle import config
from smartclide_dle.api.v1 import api
from smartclide_dle.api import namespaces_v1
from smartclide_dle.iamodeler import iamodeler_ns_v1
from smartclide_dle.core import cache, limiter

app = Flask(__name__)

VERSION = (1, 0)
AUTHOR = 'AIR Institute'


# =======================
# IAmodeler configuration
# =======================

app.config.update(
    ERROR_404_HELP=False,  # No "but did you mean" messages
    RESTX_MASK_SWAGGER=False,  # No fields mask
)

# Log POST bodies
@app.before_request
def log_request_info():
    # app.logger.debug('Headers: %s', request.headers)
    data = request.get_data()
    if data:
        try:
            app.logger.info('Body: %s', data.decode())
        except UnicodeDecodeError:
            app.logger.info('Body: <Non UTF8 content>')

# =============================
# IAmodeler configuration (end)
# =============================

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
    
    for ns in iamodeler_ns_v1:
        api.add_namespace(ns)


def main():
    initialize_app(app)
    separator_str = ''.join(map(str, ["=" for i in range(175)]))
    print(separator_str)
    print(f'Debug mode: {config.DEBUG_MODE}')
    print(f'Authors: {get_authors()}')
    print(f'Version: {get_version()}')
    print(f'Base URL: http://localhost:{config.PORT}{config.URL_PREFIX}')
    print(separator_str)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG_MODE)



if __name__ == '__main__':
    main()