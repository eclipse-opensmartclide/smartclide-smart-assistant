#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************

import collections
import logging
import os
import pickle
import tempfile
import warnings
import zipfile

import jinja2

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

import numpy as np
import pandas as pd


from sklearn import  model_selection, pipeline
from sklearn import linear_model, svm, neighbors, tree, ensemble, neural_network, naive_bayes, dummy
from sklearn import preprocessing, metrics, impute, compose

from . import sources
from .common import Storable, Persistable, _package_versions

logger = logging.getLogger(__name__)

CATEGORICAL_MAX_CARDINAL = 500
"""Maximum number of values allowed for a categorical variable. Prevents memory explodes due to one-hot-encoding"""


def _set_kwarg(f, fixed_kwargs):
    """Closure of a function fixing a kwarg"""

    def f2(*args, **kwargs):
        fixed_kwargs2 = {k: v for k, v in fixed_kwargs.items() if k not in kwargs}
        return f(*args, **fixed_kwargs2, **kwargs)

    return f2


# Available regressors
_regressors = {"linear": linear_model.LinearRegression,
               "sv": _set_kwarg(svm.SVR, {"gamma": "scale"}),
               "neighbors": neighbors.KNeighborsRegressor,
               "tree": tree.DecisionTreeRegressor,
               "random-forest": _set_kwarg(ensemble.RandomForestRegressor, {"n_estimators": 100}),
               "extra-trees": _set_kwarg(ensemble.ExtraTreesRegressor, {"n_estimators": 100}),
               "gradient-boosting": ensemble.GradientBoostingRegressor,
               "mlp": neural_network.MLPRegressor,
               "dummy": dummy.DummyRegressor}

_regressors = {key: item for key, item in _regressors.items() if item is not None}

# Available classifiers
_classifiers = {"logistic": _set_kwarg(linear_model.LogisticRegression, {"solver": "lbfgs", "multi_class": "auto"}),
                "bayes": naive_bayes.GaussianNB,
                "sv": _set_kwarg(svm.SVC, {"gamma": "scale"}),
                "neighbors": neighbors.KNeighborsClassifier,
                "tree": tree.DecisionTreeClassifier,
                "random-forest": _set_kwarg(ensemble.RandomForestClassifier, {"n_estimators": 100}),
                "extra-trees": _set_kwarg(ensemble.ExtraTreesClassifier, {"n_estimators": 100}),
                "gradient-boosting": ensemble.GradientBoostingClassifier,
                "mlp": neural_network.MLPClassifier,
                "dummy": _set_kwarg(dummy.DummyClassifier, {"strategy": "stratified"})}

_classifiers = {key: item for key, item in _classifiers.items() if item is not None}

# Available scalers
scalers = {"standard": preprocessing.StandardScaler,
           "maxabs": preprocessing.MaxAbsScaler,
           "minmax": preprocessing.MinMaxScaler,
           "robust": preprocessing.RobustScaler,
           "quantile": preprocessing.QuantileTransformer,
           }


def mean_relative_error(y_true, y_pred, weights=None):
    """Get the mean relative error with respect to the prediction"""
    if weights is not None:
        return np.mean(abs((1 - y_true / y_pred * weights)))
    else:
        return np.mean(abs((1 - y_true / y_pred)))


def root_mean_squared_error(y_true, y_pred, weights=None):
    """Get the RSME"""
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred, sample_weight=weights))


# Available metrics for regressors
_regression_metrics = collections.OrderedDict([
    ("evar", metrics.explained_variance_score),
    ("mae", metrics.mean_absolute_error),
    ("mse", metrics.mean_squared_error),
    ("rmse", root_mean_squared_error),
    ("msle", metrics.mean_squared_log_error),
    ("med_ae", metrics.median_absolute_error),
    ("r2", metrics.r2_score),
    ("mre", mean_relative_error)
])

# Available metrics for binary classifiers, not depending on a target class
_classifier_metrics = collections.OrderedDict([
    ("Accuracy", metrics.accuracy_score),
    ("Cohen kappa", metrics.cohen_kappa_score),
    ("Matthews Phi", metrics.matthews_corrcoef),
])

# ROC uses probability score instead of values
_classifier_score_metrics = collections.OrderedDict([
    ("ROC AUC", metrics.roc_auc_score),
])

_classifier_metrics_averaged = collections.OrderedDict([
    ("Precision", lambda criterium: _set_kwarg(metrics.precision_score, {'average': criterium})),
    ("Recall", lambda criterium: _set_kwarg(metrics.recall_score, {'average': criterium})),
    ("F1", lambda criterium: _set_kwarg(metrics.f1_score, {'average': criterium}))
])

_average_criteria = ['micro', 'macro', 'weighted']
"""Criteria to average a target-dependent metric. Note None can be used instead to retrieve the list of
target-dependent values"""


def _df_to_xy(df, target):
    """Convenience function to split a df in X, y matrices"""
    return df.drop(target, axis=1), df[target]


class TrainTestConfig(Storable):
    """Define how a train/test split is performed"""
    class_kwargs = ["test_size", "shuffle", "random_state", "full_refit"]
    kwarg_mapping = {}

    def __init__(self, test_size=0.25, shuffle=False, random_state=5, full_refit=False):
        """

        Args:
            test_size (int or float): Fraction of elements in the test set (float) or number of elements in the test
                                      set (int).
            shuffle (bool): Whether the data is shuffled before partitioning.
            random_state (int): A random state used if shuffle is True.
            full_refit (bool): Whether to refit with the whole dataset after evaluation.

        """
        super().__init__()
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.full_refit = full_refit

    def get_splitter(self):
        """Get a function of signature df->df_train, df_test"""
        if not self.test_size:
            # If no test set was stated, return an empty dataframe with the same columns
            if self.shuffle:
                logger.warning("Ignoring shuffle option with no test set.")

            def splitter(*args):
                return args[0], pd.DataFrame(columns=args[0].columns)
        else:
            def splitter(*args):
                return model_selection.train_test_split(*args, test_size=self.test_size, shuffle=self.shuffle,
                                                        random_state=self.random_state if self.shuffle else None)

        return splitter


class ScalerConfig(Storable):
    """Define a scaling method"""
    class_kwargs = ["method", "pars"]
    kwarg_mapping = {}

    def __init__(self, method=None, pars=None):
        super().__init__()
        self.method = method
        self.pars = pars if pars is not None else {}

    def get_scaler(self):
        return scalers[self.method](**self.pars)

    def __bool__(self):
        return bool(self.method)


class EstimatorConfig(Storable):
    """Define a sklearn estimator"""
    class_kwargs = ["method", "model_config", "scaler"]
    kwarg_mapping = {"scaler": ScalerConfig}

    def __init__(self, method, model_config=None, scaler=None):
        """

        Args:
            method (str): Method used to build an estimator. Available values are _regressors.keys() and
                          _classifiers.keys().
            model_config (dict of str): Hyperparameters of the the estimator.
            scaler (ScalerConfig): An object describing automatic data scaling.
            selector (str): Method used for feature selection. Only 'mi' is available at the moment.
            efs_config(dict): Deprecated, kept for compatibility purposes.

        """
        super().__init__()
        self.method = method
        self.model_config = model_config if model_config is not None else {}
        if isinstance(scaler, ScalerConfig):
            self.scaler = scaler
        elif isinstance(scaler, dict):
            self.scaler = ScalerConfig.from_dict(scaler)
        elif scaler is None:
            self.scaler = None
        else:
            raise TypeError("The scaler argument must be a ScalerConfig or a dict describing it.")

    def get_estimator(self, regression):
        """
        Get an sklearn estimator for the description.

        Args:
            regression (bool): Whether the estimator is used for a regression or classification problem.

        Returns:
            sklearn.pipeline.Pipeline: A pipeline with the sklearn estimator.

        """
        if regression:
            if self.method not in _regressors:
                raise ValueError("Invalid regressor: %s" % self.method)
            e = _regressors[self.method](**self.model_config)
        else:
            if self.method not in _classifiers:
                raise ValueError("Invalid classifier: %s" % self.method)
            e = _classifiers[self.method](**self.model_config)

        p = []

        p.append(("model", e))
        return pipeline.Pipeline(p)


class PredictorEvaluation:
    """Define the evaluation of a predictor"""

    def __init__(self, estimator, df, target, weights=None, target_encoder=None):
        """

        Args:
            estimator (sklearn.base.BaseEstimator): A sklearn estimator to evaluate.
            df (pandas.DataFrame): A dataframe with the information.
            target (str): The name of the attribute in the target.
            weights (list of float): Weights for each instance in df.
            target_encoder (sklearn.preprocessing.LabelEncoder): A encoder for the target as used in the estimator.
                                                                 Useful in classification problems.

        """
        self.target = target

        self.x, self.y = _df_to_xy(df, target)
        self.predictions = estimator.predict(self.x)
        self.target_encoder = target_encoder
        if target_encoder is not None:
            self.predictions = target_encoder.inverse_transform(self.predictions)

        # Cast to Series
        self.predictions = pd.Series(self.predictions, index=self.y.index)

        self.probabilities = None
        try:
            self.probabilities = estimator.predict_proba(self.x)
            self.proba_classes = list(target_encoder.inverse_transform(estimator.classes_))
            # BEWARE: These proba_classes are not ordered like self.classes, but by the internal representation.
        except AttributeError:
            pass

        self.weights = weights


class ClassifierEvaluation(PredictorEvaluation):
    """Define the evaluation of a classifier"""

    def __init__(self, *args, **kwargs):
        # Additional kwarg: classes
        super().__init__(*args, **{key: kwargs[key] for key in ["weights", "target_encoder"]})
        if "classes" in kwargs and kwargs["classes"]:
            self.classes = kwargs["classes"]
        else:
            self.classes = None
            logger.warning("Number of classes not provided, guessed from test set.")
            self.classes = list(set(self.y))

        self.multiclass = bool(len(self.classes) > 2)
        # Find out the majority class for roc_auc_score
        self.majority_class = self.y.value_counts().index[0]

        self.metrics = self._get_metrics()

    def _get_metrics(self):
        """Calculate the evaluation metrics"""
        par_list = []
        # Fixed pars
        for metric, f in _classifier_metrics.items():
            try:
                par_list.append((metric, f(self.y, self.predictions, sample_weight=self.weights)))
            except (ValueError, TypeError) as e:
                logger.warning("Unable to calculate metric %s: %s" % (metric, e))
                par_list.append((metric, np.nan))

        # Parameters depending on target/averaged SCORES
        if self.probabilities is not None:
            if self.multiclass:
                # BEWARE: For target order, cf. comments in the constructor
                logger.warning("Multiclass ROC not implemented yet")
            else:
                for metric, f in _classifier_score_metrics.items():
                    try:
                        # Prefer the majority class, as stated in the docs
                        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
                        i = self.proba_classes.index(self.majority_class)
                        par_list.append((metric, f(self.y == self.proba_classes[i], self.probabilities[:, i],
                                                   sample_weight=self.weights)))
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.warning("Unable to calculate metric %s: %s" % (metric, e))
                        par_list.append((metric, np.nan))
        else:
            for metric, f in _classifier_score_metrics.items():
                logger.warning("Unable to calculate metric %s: %s" % (metric, "Score values are not available."))
                par_list.append((metric, np.nan))

        # Parameters depending on target/averaged
        for metric, f in _classifier_metrics_averaged.items():
            try:
                par_list.append((metric,
                                 {**{criterium: f(criterium)(self.y, self.predictions, sample_weight=self.weights) for
                                     criterium in _average_criteria}, **{
                                     "target": f(None)(self.y, self.predictions,
                                                       sample_weight=self.weights, labels=self.classes).tolist()}}))
            except (ValueError, TypeError) as e:
                logger.warning("Unable to calculate metric %s: %s" % (metric, e))
                par_list.append((metric, np.nan))

        return collections.OrderedDict(par_list)

    def get_confusion_matrix(self):
        """
        Computes the confusion matrix

        Returns:
            numpy.ndarray: the confusion matrix
        """
        return metrics.confusion_matrix(self.y, self.predictions, labels=self.classes)

    def plot_confusion_matrix(self, figure_kwargs=None):
        """
        Plot the confusion matrix

        Args:
            figure_kwargs (dict): Parameters to create the Figure.

        Returns:
            Axes: An axes instance for further tweaking.

        """
        import matplotlib
        import seaborn as sns

        if figure_kwargs is None:
            figure_kwargs = {}
        fig = plt.figure(**figure_kwargs)

        gs0 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[20, 2], hspace=0.05)
        gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], hspace=0)

        ax = fig.add_subplot(gs0[0])
        cax1 = fig.add_subplot(gs00[0])
        cax2 = fig.add_subplot(gs00[1])

        m = self.get_confusion_matrix()

        vmin = np.min(m)
        vmax = np.max(m)
        off_diag_mask = np.eye(*m.shape, dtype=bool)

        sns.heatmap(m, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cax2,
                    )
        sns.heatmap(m, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cax1,
                    xticklabels=self.classes, yticklabels=self.classes, cbar_kws=dict(ticks=[]))

        return ax

    def get_roc_curve(self, positive_class):
        """
        Return the Receiver Operating Characteristic curve (ROC)

        Args:
            positive_class (str): Name of the class regarded as "positive".

        Returns:
            3-tuple with:
                - fpr (numpy.ndarray): Increasing false positive rates which define the ROC curve.
                - tpr (numpy.ndarray): Increasing true positive rates which define the ROC curve.
                - thresholds (numpy.ndarray): Decreasing thresholds used to compute the curve.

        """
        try:
            i = self.target_encoder.transform([positive_class])[0]
        except ValueError:
            logger.error("Positive class not found")
            raise
        if self.probabilities is not None:
            return metrics.roc_curve(self.target_encoder.transform(self.y), self.probabilities[:, i], pos_label=i)
        else:
            logger.warning("Probability predictions are not available. The ROC curve follows a sharp profile.")
            return metrics.roc_curve(self.target_encoder.transform(self.y),
                                     self.target_encoder.transform(self.predictions), pos_label=i)


class RegressorEvaluation(PredictorEvaluation):
    """Define the evaluation of a regressor"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = self._get_metrics()

    def _get_metrics(self):
        """Calculate the evaluation metrics"""
        par_list = []
        for k, f in _regression_metrics.items():
            try:
                par_list.append((k, f(self.y, self.predictions)))
            except (ValueError, TypeError):
                par_list.append((k, np.nan))
        return collections.OrderedDict(par_list)


def _remove_from_list(l, to_remove):
    """Remove a list of values from the target list, ignore those not found"""
    if to_remove is None:
        return l
    return list(set(l) - set(to_remove))


def get_preprocessor(types, attributes=None, ignore=None, scaler=None):
    """
    Get a preprocessor performing encoding and scaling.

    Args:
        types (dict): Mapping of attributes to types.
        attributes (list): Subset of attributes to include in the preprocessor.
        ignore (list of str): An optional attribute to exclude from the preprocessing (e.g., a target in supervised learning).
        scaler (ScalerConfig or dict): An object describing automatic data scaling.

    Returns:

    """
    # Prepare predictor pipeline
    transformers = []

    if attributes:
        types = {x: y for x, y in types.items() if x in attributes}

    numeric_features = _remove_from_list([key for (key, value) in types.items() if value == "real"], ignore)

    if numeric_features:
        if scaler:
            numeric_transformer = pipeline.Pipeline(steps=[
                ('imputer', impute.SimpleImputer(strategy='median')),
                ('scaler', scaler.get_scaler())
            ])
        else:
            numeric_transformer = pipeline.Pipeline(steps=[
                ('imputer', impute.SimpleImputer(strategy='median')),
            ])

        transformers.append(('num', numeric_transformer, numeric_features))

    categorical_features = _remove_from_list([key for (key, value) in types.items() if value == "categorical"], ignore)

    if categorical_features:
        categorical_transformer = pipeline.Pipeline(steps=[
            ('imputer', impute.SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore'))])

        transformers.append(('cat', categorical_transformer, categorical_features))

    return compose.ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)


class GenericPredictor:
    def __init__(self):
        pass

    def predict(self, data):
        """
        Use the predictor in a set of values.

        Args:
            data (2d-array): A list of points to predict.

        Returns:
            (1d-array): List of predictions.

        """
        raise NotImplementedError("This class must be inherited")

    def batch_predict(self, source, skip=None, feature_mapping=None):
        """
        Make a prediction for all instances in a source.

        Args:
            source (sources.Source): The source used to make predictions.
            skip (str): An optional name of an attribute to drop (e.g., the target if present).
                        Ignored if feature_mapping is not None.
            feature_mapping (list of int): Positions of the source matching the model input. E.g.: [1, 0, 0] describes
                                           a 3-attribute input, where the first value is obtained from the second column
                                           of the source (1 zero-based) and the second and third are obtained from the
                                           first column of the source (0 zero-based).


        Returns:
            list: a list with the predictions

        """
        if feature_mapping:
            logger.info("%s Batch-predicting for %s using feature mapping %s" % (
                str(self), source.get_description(), feature_mapping))
        elif skip is not None:
            logger.info(
                "%s Batch-predicting for %s skipping %s" % (str(self), source.get_description(), skip))
        else:
            logger.info("%s Batch-predicting for %s" % (str(self), source.get_description()))

        data = source.get_pandas()
        if feature_mapping:
            # Extract numpy to force renaming
            data = data.iloc[:, feature_mapping]
        elif skip is not None:
            try:
                data = data.drop(skip, axis=1)
            except KeyError:
                pass

        # Pass a numpy array to prevent attribute name variation
        predictions = self.predict(data.values)
        return predictions


class Predictor(Persistable, GenericPredictor):
    """Define a predictor, including source, preprocessing and evaluation


    Attributes:
        estimator (sklearn.base.BaseEstimator): estimator fitted by the last call to the fit method.
        evaluation (PredictorEvaluation): evaluation of the fitted_estimator.
    """
    class_kwargs = ["source", "target", "sample_weight", "attributes", "train_test_config",
                    "estimator_config", "sample_weight_evaluation",
                    "nlp_config", "imbalanced", "date_encoding"]
    kwarg_mapping = {"source": sources.Source, "train_test_config": TrainTestConfig,
                     "estimator_config": EstimatorConfig}
    category = ["models"]

    def __init__(self, source, target=None, sample_weight=None, attributes=None, train_test_config=None,
                 estimator_config=None, sample_weight_evaluation=True,
                 nlp_config=None, imbalanced=None, date_encoding=None,
                 **kwargs):
        """

        Args:
            source (sources.Source): The source of data.
            target (str): Name of the attribute to predict. If none is provided, try to use the one set in the source.
            sample_weight (str): Name of an attribute with the sample weights. The weight will not be used as a
                                 predictor, unless explicitely set in attributes.
            attributes (list of str): A subset of attributes used to build the model.
            train_test_config (TrainTestConfig): Instance defining how test-train splitting is performed.
            estimator_config (EstimatorConfig): Instance defining the estimator, possibly including preprocessing.

        """
        super().__init__()
        # This init MUST NOT raise exceptions, to allow persistence
        self.source = source
        self.target = target
        # Update from the source if needed. If still not provided, errors will raise on fit
        if not target:
            self.target = source.target
        self.sample_weight = sample_weight if sample_weight else None
        if sample_weight is not None and source.types[sample_weight] != "real":
            logger.error("The weight attribute must be continuous.")
        if attributes is not None:
            # Include the target so it is extracted in the df
            # Make a copy of the list
            self.attributes = attributes[:] + [self.target]
            # Include the sample weight if needed, but if must be removed later
            if sample_weight is not None and sample_weight not in self.attributes:
                self.attributes.append(sample_weight)
        else:
            self.attributes = None
        self.weight_is_attribute = attributes is None or sample_weight in attributes

        self.train_test_config = train_test_config if train_test_config is not None else TrainTestConfig()
        self.estimator_config = estimator_config if estimator_config is not None else EstimatorConfig("tree")
        self.nlp_config = nlp_config
        self.sample_weight_evaluation = sample_weight_evaluation
        self.imbalanced = (imbalanced if imbalanced is not None else "none").lower()

        if source.types.get(self.target) == "text":
            logger.error("The target attribute can not be text. Did you mean it to be treated as categorical?")
        self.regression = source.types[self.target] == "real"

        self.date_encoding = date_encoding

        self.searcher = None
        self.estimator = None
        self.evaluation = None

        self.target_encoder = None
        self.classes = None
        self.input_names = None

        # Example of data to generate a server
        self._example_x = None
        self._example_y = None

    def save(self, filename, *args, **kwargs):
        return super().save(filename, *args, **kwargs)

    @classmethod
    def load(cls, filename, *args, **kwargs):
        loaded = super().load(filename, *args, **kwargs)
        return loaded

    def get_pandas(self, encoding=True):
        """
        Get a DataFrame, properly encoding the variables if asked.

        Args:
            encoding (bool): Whether to automatically select a suitable encoding.

        Returns:
            pandas.DataFrame: The DataFrame.

        """
        # Cf drop_first=True in pd.get_dummies
        if encoding:
            if self.regression:
                return self.source.get_pandas(attributes=self.attributes, categorical_encoding="ohe")
            else:
                return self.source.get_pandas(attributes=self.attributes, categorical_encoding="ohe",
                                              specific_encoding={self.target: "label"})
        else:
            return self.source.get_pandas(attributes=self.attributes)

    def train_test_split(self, encoding=True):
        """
        Return a pair of dataframes with train and test data, according to instance configuration

        Args:
            encoding (bool): Whether to automatically select a suitable encoding.

        Returns:
            2-tuple of pandas.DataFrame: The train and test DataFrames.

        """
        return self.train_test_config.get_splitter()(self.get_pandas(encoding=encoding).dropna())

    def fit(self, full_refit=None, n_jobs=None):
        """
        Fit the sklearn estimator

        Args:
            full_refit (bool): Overwrite the Train-Test full_refit parameter.

        Returns:
            Predictor: An instance of self.

        """
        logger.info("Fitting %s:" % str(self))

        # Store the attribute names for model export
        self.get_input_names()

        if full_refit is None:
            full_refit = self.train_test_config.full_refit

        estimator = self.estimator_config.get_estimator(self.regression)
        df_train, df_test = self.train_test_split(encoding=False)

        if not self.regression:
            classes_train = list(df_train[self.target].unique())
            classes_test = list(df_test[self.target].unique())

            classes_not_in_test = list(set(classes_train) - set(classes_test))
            if classes_not_in_test:
                logger.warning("There are train classes not present in the test set: %s" % classes_not_in_test)
            classes_not_in_train = list(set(classes_test) - set(classes_train))
            if classes_not_in_train:
                logger.warning("There are test classes not available in the train set: %s" % classes_not_in_train)

            self.classes = list(set(classes_train) | set(classes_test))
        else:
            self.classes = None

        if self.sample_weight is not None:
            train_weights = df_train[self.sample_weight].values
            test_weights = df_test[self.sample_weight].values if self.sample_weight_evaluation else None
            if not self.weight_is_attribute:
                df_train = df_train.drop(self.sample_weight, axis=1)
                df_test = df_test.drop(self.sample_weight, axis=1)
        else:
            train_weights = None
            test_weights = None

        ignore_list = [self.target]
        # Detect trivial and problematic categorical attributes:
        categorical_features = [key for (key, value) in self.source.types.items() if value == "categorical"]
        trivial_categorical = []
        for c in categorical_features:
            cardinal = len(df_train[c].unique())
            if cardinal <= 1 or cardinal >= len(df_train):
                logger.warning("Discarding trivial categorical attribute: %s" % c)
                trivial_categorical.append(c)
            if cardinal > CATEGORICAL_MAX_CARDINAL:
                # Can be a problem with one-hot encoding
                logger.warning("Discarding big cardinal (%d) categorical attribute: %s" % (cardinal, c))
                trivial_categorical.append(c)

        ignore_list += trivial_categorical

        preprocessor = get_preprocessor(self.source.types, attributes=self.attributes,
                                        ignore=ignore_list, scaler=self.estimator_config.scaler)

        self.searcher = pipeline.Pipeline([("preprocessor", preprocessor), ("search", estimator)])

        x, y = _df_to_xy(df_train, self.target)

        self._example_x = x[:2].values.tolist()
        self._example_y = y[:2].values.tolist()

        # Target encoding
        if not self.regression:
            self.target_encoder = preprocessing.LabelEncoder()
            y = self.target_encoder.fit_transform(y)

        if isinstance(y, pd.Series):
            y = y.values

        # Fit the pipeline
        if train_weights is None:
            self.searcher.fit(x, y)
        else:
            self.searcher.fit(x, y, search__model__sample_weight=train_weights)

        # Extract the evaluation
        if not df_test.empty:
            if not self.regression:
                self.evaluation = ClassifierEvaluation(self.searcher, df_test, self.target, weights=test_weights,
                                                       target_encoder=self.target_encoder, classes=self.classes)
            else:
                self.evaluation = RegressorEvaluation(self.searcher, df_test, self.target, weights=test_weights,
                                                      target_encoder=self.target_encoder)
        else:
            self.evaluation = None

        if full_refit:
            df = self.get_pandas(encoding=False)
            if self.sample_weight is not None:
                refit_weights = df[self.sample_weight].values if self.sample_weight_evaluation else None
                if not self.weight_is_attribute:
                    df = df.drop(self.sample_weight, axis=1)
            else:
                refit_weights = None
            x, y = _df_to_xy(df, self.target)

            if not self.regression:
                y = self.target_encoder.fit_transform(y)

            if isinstance(y, pd.Series):
                y = y.values

            full_estimator = pipeline.Pipeline(
                steps=[("preprocessor", preprocessor),
                       ("model", self.searcher.named_steps["search"].named_steps["model"])])

            if refit_weights is None:
                full_estimator.fit(x, y)
            else:
                full_estimator.fit(x, y, model__sample_weight=refit_weights)

            self.estimator = full_estimator
        else:

            steps = [("preprocessor", preprocessor)]
            if "selector" in self.searcher.named_steps:
                steps.append(("selector", self.searcher["selector"]))
            # If no actual search was carried, just rename searcher to model

            steps.append(("model", self.searcher["search"]))

            self.estimator = pipeline.Pipeline(steps)

        return self

    def get_input_names(self):
        """
        Get the names of the inputs of the model.

        This value is cached to avoid reloading the source (and to allow without it).

        Returns:
            list of str: Ordered names of the inputs.

        """
        if self.input_names is None:
            if self.attributes is not None:
                attributes = self.attributes[:-1]  # Do not include the target
            else:

                attributes = list(self.source.get_pandas().columns)
                attributes.remove(self.target)
            self.input_names = attributes
        return self.input_names

    def get_internal_names(self):
        """
        Get the names of the internal inputs of the model, possibly including the effects of preprocessing.

        Returns:
            list of str: Ordered names of the internal inputs.

        """
        column_transformer = self.estimator.named_steps['preprocessor']
        attributes = self.get_input_names()

        col_name = []
        # Extracting the names from a column transformer is not trivial.
        # This code is adapted from https://github.com/scikit-learn/scikit-learn/issues/12525

        if column_transformer.transformers_[-1][0] == "remainder":
            transformers = column_transformer.transformers_[:-1]
        else:
            transformers = column_transformer.transformers_

        for transformer_in_columns in transformers:  # The last one is special, see later
            raw_col_name = transformer_in_columns[2]
            # The transformer is expected to be a pipeline such that the last element is the encoder.
            # Just in case

            if isinstance(transformer_in_columns[1], pipeline.Pipeline):
                transformer = transformer_in_columns[1].steps[-1][1]
            else:
                transformer = transformer_in_columns[1]
            try:
                names = transformer.get_feature_names()
            except AttributeError:  # if no 'get_feature_names' function, use raw column name
                names = raw_col_name

            if isinstance(names, np.ndarray):
                col_name += names.tolist()
            elif isinstance(names, list):
                col_name += names
            elif isinstance(names, str):
                col_name.append(names)

        # If there are remaining columns, the last transformer is like ('remainder', 'passthrough', [1, 2, 4]), so:
        if (column_transformer.remainder == "passthrough" and  # Always True, but check for versatility
                column_transformer.transformers_[-1][0] == "remainder"):
            col_name += [attributes[i] for i in self.estimator.named_steps['preprocessor'].transformers_[-1][-1]]
        return col_name

    def predict(self, data):
        logger.info("Predicting with %s" % str(self))
        if self.estimator is None:
            raise ValueError("Predictor not fit")

        if not isinstance(data, pd.DataFrame):
            # Add column name for preprocessing
            attributes = self.get_input_names()
            data = pd.DataFrame(data, columns=attributes)

        predictions = self.estimator.predict(data)

        if self.target_encoder is None:
            return predictions
        else:
            return self.target_encoder.inverse_transform(predictions)

    def predict_proba(self, data):
        """
        Use the predictor in a set of values to return a probability

        Only makes sense for certain classifier methods

        Args:
            data (2d-array): A list of points to predict

        Returns:
            2-tuple: A pair with:
                (2d-array): List with the list of probabilities for each class for each point.
                List of str: The ordered list of classes the probabilities refer to.

        """
        logger.info("Predicting probabilities with %s" % str(self))
        if self.estimator is None:
            raise ValueError("Predictor not fit")

        if not isinstance(data, pd.DataFrame):
            # Add column name for preprocessing
            attributes = self.get_input_names()
            data = pd.DataFrame(data, columns=attributes)

        predictions = self.estimator.predict_proba(data)
        classes = list(self.target_encoder.inverse_transform(self.estimator.classes_))

        return predictions, classes

    def __str__(self):
        """
        Get a short string description of the predictor

        Returns:
            dict: object description
        """

        d = {}

        d["source"] = str(self.source)
        d["target"] = self.target
        if self.attributes is not None:
            d["attributes"] = self.attributes
        if self.sample_weight is not None:
            d["sample_weight"] = self.sample_weight
        d["train_test_config"] = self.train_test_config.get_description()
        d["estimator_config"] = self.estimator_config.get_description()
        if self.nlp_config is not None:
            d["nlp_config"] = self.nlp_config.get_description()

        return "<Model %s>" % str(d)

    def get_feature_importance(self, aggregate_categorical=True):
        """
        Get a mapping of features and their importance in the predictor

        This method only makes sense for methods providing a measure of the importance, such as those based in
        decision trees or their ensembles, like random-forest or gradient boosting.

        Args:
            aggregate_categorical (bool): Whether to aggregate categorical variables. If False, values like
                                          <attribute>_<value> will be in the output.

        Returns:
            OrderedDict of str to float: A mapping of names to attribute importance. Follows the internal order of
                                           variables in the pipeline.

        Raises:
                AttributeError: If the underlying model does not provide importances.

        """
        features = _get_feature_names(self.estimator["preprocessor"])
        importances = self.estimator["model"]["model"].feature_importances_

        # Maps, e.g., 'cat' -> ['species']
        transformed_features = {x[0]: x[2] for x in self.estimator["preprocessor"].transformers}

        output = collections.OrderedDict()
        for feature, importance in zip(features, importances):
            if feature.startswith("num__"):
                output[feature[5:]] = importance
            # cat__onehot__... in the transformed, but output of _get_feature_names is like this
            elif feature.startswith(
                    "onehot__"):
                attribute, value = feature[8:].split("_", 1)  # x0_setosa -> x0, setosa
                number = int(attribute[1:])  # 0
                real_name = transformed_features["cat"][number]  # species
                if aggregate_categorical:
                    output[real_name] = output.get(real_name, 0) + importance
                else:
                    output[real_name + "_" + value] = importance

        return output

    def get_dependence(self):
        """Return a requirements-like description of the packages needed to run the predictor"""
        deps = _package_versions.copy()
        return deps

    def export_server(self, filename):
        """
        Create the files needed to independently serve the model.

        Args:
            filename (str): Path of the generated zip file.


        """
        with tempfile.TemporaryDirectory() as folder:
            logger.info("Exporting %s to %s using temporary %s" % (
                str(self), filename, folder))
            input_names = self.get_input_names()
            # Save binary files
            pickle.dump([self.estimator, self.target_encoder, input_names],
                        open(os.path.join(folder, "model.pkl"), "wb"))
            # Adapt the server
            with open(os.path.join(os.path.dirname(__file__), "templates", "server.py")) as file:
                template = jinja2.Template(file.read())

            # Ensure attribute names are strings with no quotes for the template.
            input_names = [str(s).replace('"', '') for s in input_names]

            template.stream(
                description='"Input for the model. A list with lists of parameters in the following order: %s."' % ", ".join(
                    input_names),  # Ensure string is correctly quoted!
                example=self._example_x,
                output_example=self._example_y).dump(os.path.join(folder, "server.py"))

            # Create the requirements file
            with open(os.path.join(os.path.dirname(__file__), "templates", "requirements.txt")) as file:
                template = jinja2.Template(file.read())

            template.stream(
                model_packages="\n".join(("%s==%s" % (k, v) for k, v in self.get_dependence().items()))).dump(
                os.path.join(folder, "requirements.txt"))

            zipf = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
            zipf.write(os.path.join(folder, "model.pkl"), "model.pkl")
            zipf.write(os.path.join(folder, "server.py"), "server.py")
            zipf.write(os.path.join(folder, "requirements.txt"), "requirements.txt")
            zipf.write(os.path.join(os.path.dirname(__file__), "templates", "readme.md"), "readme.md")
            zipf.close()


def _get_feature_names(column_transformer):
    """Get a list of feature names from columns transformers"""

    # Main limitations: one hot encoders output something like onehot__x0_setosa (with xi names rather than the original ones)
    # Code adapted from https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/compose/_column_transformer.py#L345
    # Which itself adapts the original sklearn method

    def get_names(trans):
        # column defined below
        if trans == 'drop' or (hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if (not isinstance(column, slice)) and all(isinstance(col, str) for col in column):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            # >>> Change: Return input column names if no method available
            # Turn error into a warning, unless we know it is a "safe" class
            if not isinstance(trans, impute.SimpleImputer):
                warnings.warn("Transformer %s (type %s) does not "
                              "provide get_feature_names. "
                              "Will return input column names if available"
                              % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [str(name) + "__" + str(f) for f in column]

        return [str(name) + "__" + str(f) for f in trans.get_feature_names()]

    # Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == pipeline.Pipeline:
            # Recursive call on pipeline
            _names = _get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names
