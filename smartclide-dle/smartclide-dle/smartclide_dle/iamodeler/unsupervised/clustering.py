import numpy as np
import pandas as pd
from sklearn import cluster, pipeline, mixture
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import logging

from .. import sources
from ..common import Persistable
from ..models import ScalerConfig, get_preprocessor

logger = logging.getLogger(__name__)

clustering_methods = {
    'kmeans': cluster.KMeans,
    'dbscan': cluster.DBSCAN,
    'gaussianmixture': mixture.GaussianMixture,
    "spectral": cluster.SpectralClustering,
    "agglomerative": cluster.AgglomerativeClustering,
}


class Clustering(Persistable):
    """
    A clustering performed on a source.
    """
    class_kwargs = ["source", "method", "parameters", "scaler", "nlp_config", "knn_pars"]
    kwarg_mapping = {"source": sources.Source, "scaler": ScalerConfig}
    category = ["unsupervised"]

    def __init__(self, source, method, parameters=None, scaler=None, nlp_config=None, knn_pars=None, **kwargs):
        """
        Attributes:
            source (Source): data source.
            method (str): Method to use for clustering. Available methods are 'kmeans' and 'dbscan'.
            parameters (dict of str): Parameters of the clustering method.
            scaler (ScalerConfig): An object describing automatic data scaling.
            nlp_config (text.NLPConfig): Instance defining the NLP.
            knn_pars (dict of str): Parameters for the kNN classifier used with methods that do not divide the
                                         parameter space ("spectral", "agglomerative").

        Raises:
            KeyError: if method is invalid.

        """
        super().__init__()
        self.source = source
        self.parameters = parameters if parameters is not None else {}
        self.method = method

        self.scaler = scaler
        self.nlp_config = nlp_config
        self.knn_pars = knn_pars

        # self.pipeline = pipeline.Pipeline(
        #     steps=[("preprocessor", get_preprocessor(source.types, scaler=scaler, nlp_config=nlp_config)),
        #            ("cluster", clustering_methods[method](**self.parameters))])
        nlp_config = []
        self.pipeline = pipeline.Pipeline(
            steps=[("preprocessor", get_preprocessor(source.types, scaler=scaler)),
                   ("cluster", clustering_methods[method](**self.parameters))])

        self.labels = None
        self.input_names = None

    def fit(self):
        """
        Performs clustering

        Returns:
            Clustering: The fitted clustering

        """

        df = self.source.get_pandas()

        self.pipeline.fit(df)

        if self.method == "gaussianmixture":
            self.labels = self.pipeline.predict(df)
        else:
            self.labels = self.pipeline.named_steps["cluster"].labels_

        return self

    def get_input_names(self):
        """
        Get the names of the inputs of the model.

        This value is cached to avoid reloading the source (and to allow without it).

        Returns:
            list of str: Ordered names of the inputs.

        """
        if self.input_names is None:
            self.input_names = list(self.source.get_pandas().columns)
        return self.input_names

    def predict(self, data):
        """
        Use the clustering to label a list of values

        Args:
            data (2d-array): A list of points to label

        Returns:
            (1d-array): List of clustering labels

        """
        logger.info("Predicting with %s" % str(self))
        if self.pipeline is None:
            raise ValueError("Clustering not fit")

        if not isinstance(data, pd.DataFrame):
            # Add column name for preprocessing
            attributes = self.get_input_names()
            data = pd.DataFrame(data, columns=attributes)

        if self.method == "dbscan":
            # DBSCAN does not provide a predict method, but we can manually build it, ensuring consistency with its
            # definition.
            # Assign the label of the nearest component, unless distance is greater than the threshold
            dbscan_model = self.pipeline.steps[-1][1]
            neighbors = NearestNeighbors(n_neighbors=1,
                                         metric=dbscan_model.metric,
                                         metric_params=dbscan_model.metric_params).fit(dbscan_model.components_)
            distances, indices = neighbors.kneighbors(self.pipeline[:-1].transform(data))
            predictions = np.where(distances[:, 0] <= dbscan_model.eps, dbscan_model.labels_[indices[:, 0]], -1)
        elif self.method in ["spectral", "agglomerative"]:
            # Methods which do not partition the input parameter space -> assign via nearest neighbors
            neighbors = KNeighborsClassifier().fit(self.pipeline[:-1].transform(self.source.get_pandas()), self.labels)
            predictions = neighbors.predict(self.pipeline[:-1].transform(data))
        else:
            predictions = self.pipeline.predict(data)

        return predictions

    def batch_predict(self, source, skip=None, feature_mapping=None):
        """
        Use the clustering to label all the instances in a source.

        Args:
            source (sources.Source): The source used to make predictions.
            skip: Ignored, kwarg provided to provide a common interface with supervised learning methods.
            feature_mapping (list of int): Positions of the source matching the training source.

        Returns:
            list: a list with the labels

        """
        if feature_mapping:
            logger.info("%s Batch-predicting for %s using feature mapping %s" % (
                str(self), source.get_description(), feature_mapping))
        else:
            logger.info("%s Batch-predicting for %s" % (str(self), source.get_description()))

        data = source.get_pandas()
        if feature_mapping:
            # Extract numpy to force renaming
            data = data.iloc[:, feature_mapping]

        # Pass a numpy array to prevent attribute name variation
        predictions = self.predict(data.values)
        return predictions

    def get_labels(self):
        return self.labels
