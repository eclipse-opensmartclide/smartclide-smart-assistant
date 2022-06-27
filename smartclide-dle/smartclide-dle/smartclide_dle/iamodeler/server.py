#!/usr/bin/env python
#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************

# -*- coding: UTF-8 -*-
import logging
import os
import shutil
import tempfile
import zipfile
from functools import wraps

import numpy as np
import werkzeug

from flask import Flask, jsonify, request, send_file
from flask import __version__ as _flask_version
from flask_cors import CORS
from flask_cors import __version__ as _flask_cors_version
from flask_restx import Api, Resource, fields, reqparse
from flask_restx import __version__ as _flask_restx_version

from . import __version__
from .config import config
from .files import get_file_path
from . import sources, models, unsupervised, server_definitions, common
from .tasks import celery, train_persistable

from smartclide_dle.api.v1 import api

logger = logging.getLogger(__name__)

MAX_PO_POINTS = 1000
"""Maximum amount of points in the returned po-diagram"""

HARD_TIMEOUT_EXTENSION = 10
"""Additional amount of seconds for a celery task to stop after a timeout"""

_package_versions = {"flask": _flask_version, "flask-cors": _flask_cors_version,
                     "flask-restx": _flask_restx_version}

_package_versions = {**_package_versions, **common._package_versions}


iamodeler_ns = api.namespace("iamodeler", description="General methods intended for programmers/maintainers")
sources_namespace = api.namespace("iamodeler/sources", description="Source-related methods")
plot_namespace = api.namespace("iamodeler/plot", description="Plot-related methods")
supervised_namespace = api.namespace("iamodeler/supervised",
                                     description="Create and use supervised learning methods "
                                                 "(classification and regression)")

unsupervised_namespace = api.namespace("iamodeler/unsupervised",
                                       description="Create and use unsupervised learning methods "
                                                   "(clustering, association rules, outlier detection, ...)")


version_return_model = iamodeler_ns.model('Deployment configuration', {
    'version': fields.String(description="Version code", example=__version__),
    'data_path': fields.String(description="Path to the folder where persistent products are stored", example="/tmp"),
    'auth': fields.Boolean(description="Whether header authentication is set", example=True),
    'packages': fields.Raw(description="Installed packages versions", example=_package_versions),
    'storage': fields.String(description="Address of the storage module", example="http://localhost:8080"),
    'storage-auth': fields.Boolean(description="Whether a storage header authentication is set", example=True),
    'celery': fields.Boolean(description="Whether a celery queue is used for training", example=True),
})


def authorize(f):
    """Require header token authentication to call the method"""

    @wraps(f)
    def decorated_function(*args, **kws):
        #if request.headers.get('X-IAMODELER-AUTH', "") != config["auth"]:
        #    iamodeler_ns.abort(401)
        return f(*args, **kws)

    return decorated_function


@iamodeler_ns.route('/')
class Version(Resource):
    @iamodeler_ns.marshal_with(version_return_model, code=200, description='OK')
    @authorize
    def get(self):
        """Check the server status, returning its version and configuration"""
        return {"version": __version__, "data_path": config["store"],
                "auth": bool(config["auth"]), "packages": _package_versions,
                "storage": config["storage_url"], "storage-auth": bool(config["storage_auth"]),
                "celery": bool(config["celery"])}


db_sources_output_model = iamodeler_ns.model('Database sources output model', {
    'source-id': fields.String('Unique identifier of the source', example="2d09274d-c813c8ee-a6c833a5-ac13c021"),
    'name': fields.String('Name of the source', example="Iris"),
    'instances': fields.Integer('Number of instances of the data source', example=150),
    'features': fields.Integer('Number of features of the data source', example=5),
})

EXAMPLE_SOURCE = {"type": "json", "id": "https://raw.githubusercontent.com/domoritz/maps/master/data/iris.json"}

source_model = iamodeler_ns.model('Source model', {
    'type': fields.String(
        description="Type of source. Must be one of the following:\n"
                    "- json: A JSON object.\n",
        enum=["json", "csv"],
        example="json"
    ),
    'id': fields.String(
        description="A unique identifier of the source, like an ID, a path or an URI. The meaning depends on the type."
    ),
})

attributes_model = iamodeler_ns.model('Attribute analysis definition', {
    'source': fields.Nested(source_model, description="Definition of the data source",
                            required=True, example=EXAMPLE_SOURCE),
})


@sources_namespace.route('/attributes')
class Attributes(Resource):
    # @iamodeler_ns.marshal_with(version_return_model, code=200, description='OK')
    @iamodeler_ns.expect(attributes_model)
    @authorize
    def get(self):
        """Get the list of attributes and suggested types from a source. TODO: Implementation pending"""
        return {"status": "TODO"}


train_test_model = iamodeler_ns.model('Test-Train configuration', {
    'test_size': fields.Float(description="Fraction of examples in the test set.", required=False, example=0.3),
    'shuffle': fields.Boolean(description="Whether to shuffle samples before splitting.", required=False, example=True),
    'random_state': fields.Integer(description="An integer to set the seed for random splitting.", required=False,
                                   example=5),
    'full_refit': fields.Boolean(description="Whether to refit with the whole dataset after evaluation.",
                                 required=False,
                                 example=True),
})

scaling_model = iamodeler_ns.model('Scaling configuration', server_definitions.scaling_methods)

# List all available supervised models
_supervised_keys = list(set(server_definitions.definitions["regression"].keys()) | set(
    server_definitions.definitions["classification"].keys()))

new_supervised_model = iamodeler_ns.model('Model definition', {
    'model-id': fields.String(description="A unique ID identifying the model",
                              required=True, example="c2099d83-1c55ac5b-aa080469-a44603ef"),
    'source': fields.Nested(source_model, description="Definition of the data source",
                            required=True, example=EXAMPLE_SOURCE),
    'target': fields.String(description="Name of the attribute to predict",
                            required=True, example="species"),
    'method': fields.String(description="Name of the method to use to build the model",
                            required=True, example="tree", enum=_supervised_keys),
    'scaling': fields.Nested(scaling_model, description="Object describing the scaling of the numeric features."),
    'model-config': fields.Raw(description="Set of key-value pairs configuring the estimator.\n"
                                           "Valid configurations are model-dependent. See specific API points for more "
                                           "information",
                               required=False),
    'train-test-config': fields.Nested(train_test_model,
                                       description="Object describing the train/test split configuration."),
    'timeout': fields.Float(description="A time limit in milliseconds for the training tasks. The process tries to "
                                        "stop after that. A hard stop will be forced 10 seconds later."
                                        "No time limit will apply if 0.",
                            example=0),
})

new_regression_model = iamodeler_ns.inherit('Regression definition', new_supervised_model,
                                   {'method': fields.String(description="Name of the method to use to build the model",
                                                            required=True, example="tree",
                                                            enum=list(models._regressors.keys())), })

new_classification_model = iamodeler_ns.inherit('Classification definition', new_supervised_model,
                                       {'method': fields.String(
                                           description="Name of the method to use to build the model",
                                           required=True, example="tree",
                                           enum=list(models._classifiers.keys())), })

regression_parameters = {}
regression_models = {}

for key, value in server_definitions.definitions["regression"].items():
    regression_parameters[key] = iamodeler_ns.model("Regression %s parameters" % key, value)

    regression_models[key] = iamodeler_ns.inherit('Regression %s definition' % key, new_supervised_model,
                                         {'method': fields.String(
                                             description="Name of the method to use to build the model",
                                             required=True, example=server_definitions.ui_to_inner.get(key, key),
                                             enum=[server_definitions.ui_to_inner.get(key, key)]),
                                             'model-config': fields.Nested(regression_parameters[key],
                                                                           description="Set of key-value pairs "
                                                                                       "configuring the estimator.")})

classification_parameters = {}
classification_models = {}

for key, value in server_definitions.definitions["classification"].items():
    classification_parameters[key] = iamodeler_ns.model("Classification %s parameters" % key, value)

    classification_models[key] = iamodeler_ns.inherit('Classification %s definition' % key, new_supervised_model,
                                             {'method': fields.String(
                                                 description="Name of the method to use to build the model",
                                                 required=True, example=server_definitions.ui_to_inner.get(key, key),
                                                 enum=[server_definitions.ui_to_inner.get(key, key)]),
                                                 'model-config': fields.Nested(classification_parameters[key],
                                                                               description="Set of key-value pairs "
                                                                                           "configuring the estimator."
                                                                               )})

model_id_model = iamodeler_ns.model('Model identifier', {
    'model-id': fields.String(description="A unique ID identifying the model",
                              required=True, example="c2099d83-1c55ac5b-aa080469-a44603ef"),
})

predict_model = iamodeler_ns.model('Prediction request', {
    'data': fields.List(fields.List(fields.Raw), description="Data",
                        example=[[1.2, 1.3, 1.4, 1.5], [2.2, 2.3, 2.4, 2.5]])
})

batch_predict_model = iamodeler_ns.model('Batch prediction request', {
    'source': fields.Nested(source_model, description="Definition of the data source",
                            required=True, example=EXAMPLE_SOURCE),
    'skip': fields.String(description="Name of a field to ignore. This option is ignored if feature-mapping is given.",
                          example="species"),
    'feature-mapping': fields.List(fields.Integer, description="Ordered list of positions in the predicting source "
                                                               "which correspond to the model input attributes.\n"
                                                               "Indexing is ZERO-BASED. E.g.: for a 3-attribute model "
                                                               "where the first one is in the second attribute of the "
                                                               "source (1 when zero-based) and the other two are both "
                                                               "extracted from the first attribute of the source, the "
                                                               "mapping is [1, 0, 0].",
                                   example=[0, 1, 3, 2])
})

predict_1d_model = iamodeler_ns.model('1D prediction request', {
    'data': fields.List(fields.Raw, description="A point which sets the value of all the parameters. A value for the "
                                                "one that is varied must be supplied, although it is not used.",
                        example=[5.8, 3.0, 3.7, 1.2]),
    'pos': fields.Integer(description="Position of the coordinate which has to vary, STARTING WITH ZERO.",
                          example=3),
    'values': fields.List(fields.Raw, description="List of values to use for prediction in the varying coordinate.",
                          example=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
})

predict_result_model = iamodeler_ns.model('Prediction results', {
    'predictions': fields.List(fields.Raw, description="List of values predicted", example=[["setosa", "versicolor"]])
})

model_id_arguments = reqparse.RequestParser()
model_id_arguments.add_argument('model-id', type=str, required=True)

op_model = iamodeler_ns.model("Observed-predicted diagram", {
    'observed': fields.List(fields.Raw, description="Real values in the dataset", example=[1.0, 2.0, 3.0]),
    'predicted': fields.List(fields.Raw, description="Values predicted by the model", example=[1.1, 1.9, 3.1])
})

evaluation_model = iamodeler_ns.model('Model evaluation', {
    'status': fields.String(description="Status of the model. Possible values are:\n"
                                        "-'ok': The model was successfully trained.\n"
                                        "-'training': The training is in process.\n"
                                        "-'error': An error occurred training the model.\n",
                            enum=["ok", "training", "error"]),
    'status-description': fields.String(description="Additional information about the status.",
                                        example="This is the description of an error."),
    'evaluation': fields.Raw(description="Key-value pairs with the metrics of the model.\n"
                                         "Some metrics might be target-class dependent. In that case, the value is "
                                         "replaced by a dict with some average criteria (micro, macro, weighted), as "
                                         "well as a 'target' entry mapping to a list of individual values for each of "
                                         "the classes.",
                             example={"Accuracy": 0.92, "Cohen kappa": 0.88, "Matthews Phi": 0.89,
                                      "precission": {'micro': 0.98,
                                                     'macro': 0.97,
                                                     'weighted': 0.98,
                                                     'target': [1.0, 1.0, 0.92]}}),
    'confusion': fields.List(fields.List(fields.Integer),
                             description="Confusion matrix. Only makes sense in classification problems.\n"
                                         "The number of observations known to be in class i which where assigned class "
                                         "j are given in (i,j).",
                             example=[[30, 3, 0], [3, 58, 2], [1, 5, 55]]),
    'op-diagram': fields.Nested(op_model,
                                description="Observed vs Predicted diagram. Usually makes sense only in "
                                            "regression problems",
                                ),
    'classes': fields.List(fields.String,
                           description="Ordered list of the classes. Use this to understand the confusion matrix",
                           example=["setosa", "virginica", "versicolor"]),
    'importances': fields.Raw(
        description="Key-value pairs with the importance of the featuers in the model. Only available for some predictors.\n",
        example={"feature_1": 0.92, "feature_b": 0.08}),
})

new_clustering_model = iamodeler_ns.model('Clustering definition', {
    'model-id': fields.String(description="A unique ID identifying the clustering",
                              required=True, example="c2099d83-1c55ac5b-aa080469-a44603ef"),
    'source': fields.Nested(source_model, description="Definition of the data source",
                            required=True, example=EXAMPLE_SOURCE),
    'method': fields.String(
        description="Name of the method to use to make the clustering. Available methods include:\n",
        required=True, example="kmeans",
        enum=list(unsupervised.clustering.clustering_methods.keys())),
    'config': fields.Raw(description="Set of key-value pairs configuring the clustering.\n",
                         required=False),
    'knn-pars': fields.Nested(
        iamodeler_ns.model("KNN-Clustering", server_definitions.definitions["classification"]["neighbors"]),
        description="Configuration of the kNN generalization process for algorithms which do not partition the input space",
        required=False),
})

clustering_parameters = {}
clustering_models = {}

for key, value in server_definitions.definitions["clustering"].items():
    clustering_parameters[key] = iamodeler_ns.model("Clustering %s parameters" % key, value)

    clustering_models[key] = iamodeler_ns.inherit('Clustering %s definition' % key, new_clustering_model,
                                         {'method': fields.String(
                                             description="Name of the method to use to build the model",
                                             required=True, example=server_definitions.ui_to_inner.get(key, key),
                                             enum=[server_definitions.ui_to_inner.get(key, key)]),
                                             'config': fields.Nested(clustering_parameters[key],
                                                                     description="Set of key-value pairs configuring "
                                                                                 "the clustering.")})

clustering_labels_model = iamodeler_ns.model('Clustering labels', {
    'status': fields.String(description="Status of the clustering. Possible values are:\n"
                                        "-'ok': The model was successfully trained.\n"
                                        "-'training': The training is in process.\n"
                                        "-'error': An error occurred training the model.\n",
                            enum=["ok", "training", "error"]),
    'labels': fields.List(fields.Integer,
                          description="Ordered list of the clusters each instance belongs, numbered starting from 0.\n"
                                      "Outliers will be tagged with -1 for methods allowing for their detection.",
                          example=[0, 1, 1, 0, 2, 0, 0])
})


def source_to_internal(json_data):
    """Transform a source configuration from the model API into the internal representation"""
    _type = json_data["type"].lower()
    if _type == "json":
        return {
            "class": "JSON",
            "path": json_data["id"]
        }
    elif _type == "csv":
        return {
            "class": "CSV",
            "path": json_data["id"]
        }
    else:
        raise ValueError("Invalid source type: %s" % json_data["type"])


def nlp_model_to_internal(json_data):
    if json_data is None:
        return None
    d = {
        "text_attribute": json_data.get("text_attribute"),
        "lemmatizer": json_data.get("lemmatizer"),
        "language": json_data.get("language"),
        "tf": json_data.get("tf"),
        "idf": json_data.get("idf"),
        "ignore_numbers": json_data.get("ignore-numbers")
    }
    # Skip Nones
    return {k: v for k, v in d.items() if v is not None}


def timedate_to_internal(json_data):
    if json_data is None:
        return None
    strategy = json_data.get("strategy")
    if strategy == "custom":
        return json_data.get("custom", [])
    return strategy


def supervised_model_to_internal(json_data):
    """Transform data from the model API into the internal representation"""
    data = {
        "filename": json_data["model-id"],
        "source": source_to_internal(json_data["source"]),
        "target": json_data["target"],
        "estimator_config": {
            "method": json_data["method"],
            "scaler": json_data["scaling"] if 'scaling' in json_data else {},
            "model_config": json_data["model-config"] if 'model-config' in json_data else {}
        },
        "train_test_config": json_data["train-test-config"] if 'train-test-config' in json_data else {},
        "nlp_config": nlp_model_to_internal(json_data.get("nlp-config")),
        "imbalanced": json_data.get("imbalanced"),
        "date_encoding": timedate_to_internal(json_data.get("timedate_features")),
    }

    return data


def clustering_model_to_internal(json_data):
    """Transform data from the clustering API into the internal representation"""
    data = {
        "filename": json_data["model-id"],
        "source": source_to_internal(json_data["source"]),
        "method": json_data["method"],
        "parameters": json_data["config"],
        "nlp_config": nlp_model_to_internal(json_data.get("nlp-config"))
    }
    return data


def _new_persistable(data, category, timeout=None, task_id=None):
    """Run a training trask, handling the return and timeout"""
    try:
        if timeout:
            train_persistable.apply_async((data, category),
                                          task_id=task_id,
                                          soft_time_limit=timeout / 1000.0,
                                          time_limit=timeout / 1000.0 + HARD_TIMEOUT_EXTENSION)
        else:
            train_persistable.apply_async((data, category),
                                          task_id=task_id
                                          )
    except Exception as e:
        return jsonify({"status": "error", "description": str(e)})

    return jsonify({"status": "ok"})


def _new_model(model_name=None):
    """Create a new model.

    MUST BE USED INSIDE A REQUEST
    """
    data = supervised_model_to_internal(request.json)
    timeout = request.json.get("timeout")

    if model_name is not None:
        data["estimator_config"]["method"] = server_definitions.ui_to_inner.get(model_name, model_name)

    filename = data["filename"]
    p = models.Predictor.from_dict(data)
    p.save(filename, status="training")

    task_id = "predictor-" + filename

    return _new_persistable(data, "predictor", timeout=timeout, task_id=task_id)


@supervised_namespace.route('')
class NewModel(Resource):
    @iamodeler_ns.expect(new_supervised_model)
    @iamodeler_ns.response(404, 'Source not found')
    @iamodeler_ns.response(409, 'Model is already being trained')
    @authorize
    def post(self):
        """Create a new supervised model (regression or classification)

        See nested methods for a better description of the available parameters.
        """
        return _new_model()


@supervised_namespace.route('/regression')
class NewRegressionModel(Resource):
    @iamodeler_ns.expect(new_regression_model)
    @iamodeler_ns.response(404, 'Source not found')
    @iamodeler_ns.response(409, 'Model is already being trained')
    @authorize
    def post(self):
        """Create a new regression model

        See nested methods for a better description of the available parameters.
        """
        return _new_model()


@supervised_namespace.route('/classification')
class NewClassificationModel(Resource):
    @iamodeler_ns.expect(new_classification_model)
    @iamodeler_ns.response(404, 'Source not found')
    @iamodeler_ns.response(409, 'Model is already being trained')
    @authorize
    def post(self):
        """Create a new classification model

        See nested methods for a better description of the available parameters.
        """
        return _new_model()


def prepare_regression_point(key, value):
    if server_definitions.ui_to_inner.get(key, key) not in models._regressors.keys():
        logger.info("Skipping unavailable regressor: %s" % key)
        return

    @supervised_namespace.route('/regression/%s' % key)
    class NewRegressionACertainModel(Resource):
        """Create a new regressor using the chosen method"""

        @iamodeler_ns.expect(value)
        @iamodeler_ns.response(404, 'Source not found')
        @iamodeler_ns.response(409, 'Model is already being trained')
        @iamodeler_ns.doc(id='post_new_regression_%s' % key)
        @authorize
        def post(self):
            return _new_model(model_name=key)

    NewRegressionACertainModel.post.__doc__ = "Create a new regressor using a %s" % server_definitions.ui_to_friendly[
        key]

    return NewRegressionACertainModel


_regression_points = [prepare_regression_point(key, value) for key, value in regression_models.items()]


def prepare_classification_point(key, value):
    if server_definitions.ui_to_inner.get(key, key) not in models._classifiers.keys():
        logger.info("Skipping unavailable classifier: %s" % key)
        return

    @supervised_namespace.route('/classification/%s' % key)
    class NewClassificationACertainModel(Resource):
        """Create a new classifier using the chosen method"""

        @iamodeler_ns.expect(value)
        @iamodeler_ns.response(404, 'Source not found')
        @iamodeler_ns.response(409, 'Model is already being trained')
        @iamodeler_ns.doc(id='post_new_classification_%s' % key)
        @authorize
        def post(self):
            return _new_model(model_name=key)

    NewClassificationACertainModel.post.__doc__ = "Create a new classifier using a %s" % \
                                                  server_definitions.ui_to_friendly[
                                                      key]


_classification_points = [prepare_classification_point(key, value) for key, value in classification_models.items()]


def _remove_data(id, path, extensions):
    """Remove the stored data of a persistable object"""
    file_removed = False
    for extension in extensions:
        try:
            os.remove(get_file_path(id + extension, path))
            file_removed = True
        except FileNotFoundError:
            pass
    return file_removed


def _get_size(id, path, extensions):
    """Find the size of the stored data of a persistable object"""
    size = 0
    for extension in extensions:
        try:
            size += os.path.getsize(get_file_path(id + extension, path))
        except OSError:
            pass
    return size


@supervised_namespace.route('/<id>')
@supervised_namespace.param('id', "The unique identifier of the model")
class SupervisedModel(Resource):
    @authorize
    def delete(self, id):
        """Delete an existing model"""
        if config["celery"]:
            task_id = "predictor-" + id
            celery.control.revoke(task_id, terminate=True)
        removed = _remove_data(id, ["models"], [".json", ".pkl"])
        if removed:
            return jsonify({"status": "ok"})
        else:
            supervised_namespace.abort(404, "Objects not found", status="Objects not found", statusCode="404")

    @authorize
    def get(self, id):
        """Get information of an existing model"""
        size = _get_size(id, ["models"], [".json", ".pkl"])
        try:
            description = models.Predictor.load_description(id)
            return jsonify({"size": size, "status": description.get("_status", "ok"),
                            "status-description": description.get("_status_description", "")})
        except FileNotFoundError:
            supervised_namespace.abort(404, "Objects not found", status="Objects not found", statusCode="404")


def _load_model(name):
    """
    Load a model

    Args:
        name (str): Unique identifier of the model.

    Returns:
        (str, models.Predictor or Exception): 2-tuple with the status (training/error/ok) and the model or an exception
                                              describing an error.

    """
    try:
        model = models.Predictor.load(name)
        return ["ok", model]
    except FileNotFoundError:
        return ["not-found", None]
    except ValueError:
        return ["error", None]
    except common.TrainingError as e:
        return ["error", e]
    except common.TrainingNotFinished as e:
        return ["training", e]


@supervised_namespace.route('/<id>/evaluate')
@supervised_namespace.param('id', "The unique identifier of the model")
class EvaluateModel(Resource):
    @iamodeler_ns.marshal_with(evaluation_model, code=200, description='OK')
    @authorize
    def get(self, id):
        """
        Get the metrics evaluating a trained model
        """
        status, model = _load_model(id)
        if status != "ok":
            return {"status": status, "status-description": str(model)}, 404

        # Otherwise ok.

        # Output metrics skipping invalid values
        pars = {par: value for par, value in model.evaluation.metrics.items() if
                (not isinstance(value, float) or np.isfinite(value))}

        output = {"status": "ok", "evaluation": pars}
        if model.regression:
            if len(model.evaluation.y) > MAX_PO_POINTS:
                output["op-diagram"] = {"observed": model.evaluation.y[:MAX_PO_POINTS].tolist(),
                                        "predicted": model.evaluation.predictions[:MAX_PO_POINTS].tolist()}

            else:
                output["op-diagram"] = {"observed": model.evaluation.y.tolist(),
                                        "predicted": model.evaluation.predictions.tolist()}

        else:
            output["confusion"] = model.evaluation.get_confusion_matrix().tolist()
            output["classes"] = model.evaluation.classes

        # Add feature importances if available
        try:
            output["importances"] = model.get_feature_importance(aggregate_categorical=True)
        except (AttributeError, TypeError):
            pass

        return output


@supervised_namespace.route('/<id>/predict')
@supervised_namespace.param('id', "The unique identifier of the model")
class Predict(Resource):
    @iamodeler_ns.expect(predict_model)
    @iamodeler_ns.response(404, 'Model not found')
    @iamodeler_ns.marshal_with(predict_result_model, code=200, description='OK')
    @authorize
    def post(self, id):
        """Make predictions using an existing model"""
        data = request.json['data']

        status, model = _load_model(id)
        if status != "ok":
            # Note training also returns an error.
            supervised_namespace.abort(404, "Model not found", status="Model not available", statusCode="404")

        # Otherwise ok.
        predictions = model.predict(data)

        return {"predictions": predictions.tolist()}


@supervised_namespace.route('/<id>/batch-predict')
@supervised_namespace.param('id', "The unique identifier of the model")
class BatchPredict(Resource):
    @iamodeler_ns.expect(batch_predict_model)
    @iamodeler_ns.response(404, 'Model or source not found')
    @iamodeler_ns.marshal_with(predict_result_model, code=200, description='OK')
    @authorize
    def post(self, id):
        """Make predictions using an existing model"""
        status, model = _load_model(id)
        if status != "ok":
            # Note training also returns an error.
            supervised_namespace.abort(404, "Model not found", status="Model not available", statusCode="404")

        source = sources.Source.from_dict(source_to_internal(request.json["source"]))

        predictions = model.batch_predict(source, skip=request.json.get("skip"),
                                          feature_mapping=request.json.get("feature-mapping"))

        return {"predictions": predictions.tolist()}


roc_request_model = iamodeler_ns.model('Request ROC model', {
    'positive': fields.String(description='Name of the class treated as "positive"',
                              example="setosa")
})

roc_return_model = iamodeler_ns.model('Returned ROC model', {
    'fpr': fields.List(fields.Float, description='Increasing false positive rates which define the ROC curve.',
                       example=[0, 0, 0.03, 0.1, 1]),
    'tpr': fields.List(fields.Float, description='Increasing true positive rates which define the ROC curve.',
                       example=[0, 1, 1, 1, 1]),
    'thresholds': fields.List(fields.Float, description='Decreasing thresholds used to compute the curve.',
                              example=[2, 1, 0.05, 0.01, 0])
})


@supervised_namespace.route('/<id>/roc')
@supervised_namespace.param('id', "The unique identifier of the model")
class GetROC(Resource):
    @iamodeler_ns.expect(roc_request_model)
    @iamodeler_ns.response(404, 'Model not found')
    @iamodeler_ns.marshal_with(roc_return_model, code=200, description='OK')
    @authorize
    def post(self, id):
        """Return the Receiver Operating Characteristic curve (ROC)"""
        status, model = _load_model(id)
        if status != "ok":
            # Note training also returns an error.
            supervised_namespace.abort(404, "Model not found", status="Model not available", statusCode="404")

        positive = request.json["positive"]
        fpr, tpr, thresholds = model.evaluation.get_roc_curve(positive)
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist(), }


zip_upload = reqparse.RequestParser()
zip_upload.add_argument('file',
                        type=werkzeug.datastructures.FileStorage,
                        location='files',
                        required=True,
                        help='ZIP file')
zip_upload.add_argument('id', type=str, required=True, help='The id of the imported object')


@supervised_namespace.route('/<id>/export')
@supervised_namespace.param('id', "The unique identifier of the model")
class ExportIndependentSupervised(Resource):
    @iamodeler_ns.response(404, 'Model not found')
    @authorize
    def get(self, id):
        """Download a zip with files to independently serve the model"""
        status, model = _load_model(id)
        if status != "ok":
            # Note training also returns an error.
            supervised_namespace.abort(404, "Model not found", status="Model not available", statusCode="404")

        with tempfile.NamedTemporaryFile(prefix="export-", suffix=".zip") as f:
            model.export_server(f.name)
            return send_file(f.name, as_attachment=True, mimetype="application/zip")


def _new_cluster(cluster_method=None):
    """Create a new clustering.

    MUST BE USED INSIDE A REQUEST
    """
    data = clustering_model_to_internal(request.json)
    timeout = request.json.get("timeout")

    if cluster_method is not None:
        data["method"] = server_definitions.ui_to_inner.get(cluster_method, cluster_method)

    filename = data["filename"]
    p = unsupervised.Clustering.from_dict(data)
    p.save(filename, status="training")

    task_id = "clustering-" + filename

    return _new_persistable(data, "clustering", timeout=timeout, task_id=task_id)


@unsupervised_namespace.route('/clustering')
class NewClustering(Resource):
    @iamodeler_ns.expect(new_clustering_model)
    @iamodeler_ns.response(404, 'Source not found')
    @iamodeler_ns.response(409, 'Clustering is already being trained')
    @authorize
    def post(self):
        """Create a new clustering"""
        return _new_cluster()


def _load_cluster(name):
    try:
        model = unsupervised.Clustering.load(name)
        return ["ok", model]
    except (FileNotFoundError, ValueError):
        return ["error", None]
    except common.TrainingError as e:
        return ["error", e]
    except common.TrainingNotFinished as e:
        return ["training", e]


@unsupervised_namespace.route('/clustering/<id>')
@unsupervised_namespace.param('id', "The unique identifier of the clustering")
class SavedClustering(Resource):
    @authorize
    def delete(self, id):
        """Delete an existing clustering"""
        if config["celery"]:
            task_id = "clustering-" + id
            celery.control.revoke(task_id, terminate=True)
        removed = _remove_data(id, ["unsupervised"], [".json", ".pkl"])
        if removed:
            return jsonify({"status": "ok"})
        else:
            supervised_namespace.abort(404, "Objects not found", status="Objects not found", statusCode="404")

    @authorize
    def get(self, id):
        """Get information of an existing clustering"""
        size = _get_size(id, ["unsupervised"], [".json", ".pkl"])
        try:
            description = unsupervised.Clustering.load_description(id)
            return jsonify({"size": size, "status": description.get("_status", "ok"),
                            "status-description": description.get("_status_description", "")})
        except FileNotFoundError:
            unsupervised_namespace.abort(404, "Objects not found", status="Objects not found", statusCode="404")


@unsupervised_namespace.route('/clustering/<id>/labels')
@unsupervised_namespace.param('id', "The unique identifier of the model")
class ClusteringLabelsModel(Resource):
    @iamodeler_ns.marshal_with(clustering_labels_model, code=200, description='OK')
    @authorize
    def get(self, id):
        """
        Get the labels of a clustering

        """
        status, cluster = _load_cluster(id)
        if status == "training":
            return {"status": "training"}, 404
        if status == "error":
            return {"status": "error"}, 404
        return {"status": "ok", "labels": cluster.get_labels().tolist()}


def prepare_clustering_point(key, value):
    @unsupervised_namespace.route('/clustering/%s' % key)
    class NewClusteringACertainModel(Resource):
        """Create a new clustering using the chosen method"""

        @iamodeler_ns.expect(value)
        @iamodeler_ns.response(404, 'Source not found')
        @iamodeler_ns.response(409, 'Clustering is already being trained')
        @iamodeler_ns.doc(id='post_new_clustering_%s' % key)
        @authorize
        def post(self):
            return _new_cluster(cluster_method=key)

    NewClusteringACertainModel.post.__doc__ = "Create a new clustering using %s" % server_definitions.ui_to_friendly[
        key]


_clustering_points = [prepare_clustering_point(key, value) for key, value in clustering_models.items()]

cluster_predict_model = iamodeler_ns.model('Clustering labelling request', {
    'data': fields.List(fields.List(fields.Raw), description="Data",
                        example=[[1.2, 1.3, 1.4, 1.5, "setosa"], [2.2, 2.3, 2.4, 2.5, "virginica"]])
})

cluster_batch_predict_model = iamodeler_ns.model('Clustering batch labelling request', {
    'source': fields.Nested(source_model, description="Definition of the data source",
                            required=True, example=EXAMPLE_SOURCE),
    'feature-mapping': fields.List(fields.Integer, description="Ordered list of positions in the predicting source "
                                                               "which correspond to the model input attributes.\n"
                                                               "Indexing is ZERO-BASED. E.g.: for a 3-attribute model "
                                                               "where the first one is in the second attribute of the "
                                                               "source (1 when zero-based) and the other two are both "
                                                               "extracted from the first attribute of the source, the "
                                                               "mapping is [1, 0, 0].",
                                   example=[0, 1, 3, 2, 4])
})

cluster_predict_result_model = iamodeler_ns.model('Clustering labelling results', {
    'predictions': fields.List(fields.Raw, description="List of assigned labels", example=[[2, 3]])
})


@unsupervised_namespace.route('/clustering/<id>/predict')
@unsupervised_namespace.param('id', "The unique identifier of the clustering")
class PredictCluster(Resource):
    @iamodeler_ns.expect(cluster_predict_model)
    @iamodeler_ns.response(404, 'Model not found')
    @iamodeler_ns.marshal_with(cluster_predict_result_model, code=200, description='OK')
    @authorize
    def post(self, id):
        """Make predictions using an existing clustering"""
        data = request.json['data']

        status, cluster = _load_cluster(id)
        if status != "ok":
            # Note training also returns an error.
            iamodeler_ns.abort(404, "Clustering not found", status="Clustering not available", statusCode="404")

        # Otherwise ok.
        predictions = cluster.predict(data)

        return {"predictions": predictions.tolist()}


@unsupervised_namespace.route('/clustering/<id>/batch-predict')
@unsupervised_namespace.param('id', "The unique identifier of the clustering")
class BatchPredictCluster(Resource):
    @iamodeler_ns.expect(cluster_batch_predict_model)
    @iamodeler_ns.response(404, 'Model or source not found')
    @iamodeler_ns.marshal_with(cluster_predict_result_model, code=200, description='OK')
    @authorize
    def post(self, id):
        """Make batch predictions using an existing clustering"""
        status, cluster = _load_cluster(id)
        if status != "ok":
            # Note training also returns an error.
            iamodeler_ns.abort(404, "Clustering not found", status="Clustering not available", statusCode="404")

        source = sources.Source.from_dict(source_to_internal(request.json["source"]))

        predictions = cluster.batch_predict(source, skip=request.json.get("skip"),
                                            feature_mapping=request.json.get("feature-mapping"))

        return {"predictions": predictions.tolist()}


scatter_model = iamodeler_ns.model('Scatterplot configuration', {
    'source': fields.Nested(source_model, description="Definition of the data source",
                            required=True, example=EXAMPLE_SOURCE),
    'x': fields.String(description="Attribute to use as x coordinate", example="petalLength"),
    'y': fields.String(description="Attribute to use as y coordinate", example="petalWidth"),
})


@plot_namespace.route('/scatter')
class ScatterPlot(Resource):
    # @iamodeler_ns.marshal_with(version_return_model, code=200, description='OK')
    @iamodeler_ns.expect(scatter_model)
    @authorize
    def get(self):
        """Get a scatter plot. TODO: Implementation pending"""
        return {"status": "TODO"}


distribution_model = iamodeler_ns.model('Distribution plot configuration', {
    'source': fields.Nested(source_model, description="Definition of the data source",
                            required=True, example=EXAMPLE_SOURCE),
    'x': fields.String(description="Attribute whose distribution will be plot", example="petalLength"),
})


@plot_namespace.route('/distribution')
class DistributionPlot(Resource):
    # @iamodeler_ns.marshal_with(version_return_model, code=200, description='OK')
    @iamodeler_ns.expect(distribution_model)
    @authorize
    def get(self):
        """Get a unidimensional distribution plot. TODO: Implementation pending"""
        return {"status": "TODO"}
