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
import json
import logging
import re
from os import path

import pandas as pd

from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from .config import config
from .files import get_file_path

logger = logging.getLogger(__name__)

# Maximum number of values to distinguish between a discrete variable and a continuous variable in the infer_types
# method
_MAX_DISCRETE_COUNT = 6

# A maximum number of instances to check when guessing types
_MAX_INSTANCES_TYPE_CHECK = 100


class Source:
    """A source of data"""

    def __init__(self, **kwargs):
        self.types = kwargs.get("types", {})
        if not kwargs.get("skip_type_inference"):
            self._infer_types()
        # The types are now assumed known, so in a reinstantiation type inference is not needed
        self.skip_type_inference = True

        self.target = kwargs.get("target")

    def _infer_types(self):
        """Infer the types of the attributes not already in self.types"""
        df = self.get_pandas()

        remaining = [col for col in df if col not in self.types.keys()]

        # Find out the categorical variables, including text and images
        categorical = [col for col in remaining if pd.api.types.is_string_dtype(df[col])]
        # The rest are assumed numerical
        remaining = [col for col in remaining if col not in categorical]

        # Numerical values with few values are assumed discrete
        # This should be checked manually
        discrete = [col for col in remaining if len(df[col].unique()) < _MAX_DISCRETE_COUNT]
        continuous = list(set(remaining) - set(discrete))

        self.types = {**self.types, **{c: "discrete" for c in discrete}, **{c: "real" for c in continuous},
                      **{c: "categorical" for c in categorical}}

    def get_attributes_by_type(self, target_type):
        """Get the name of the attributes of a certain type"""
        return [name for name, _type in self.types.items() if _type == target_type]

    def _get_pandas(self):
        """Get a basic Dataframe. Should be called thorough get_pandas instead"""
        raise NotImplementedError("A subclass of DataFrame must be used to be able to recover the data")

    def get_pandas(self, attributes=None, categorical_encoding=None, discrete_encoding=None, specific_encoding=None,
                   date_encoding=None, imputation=None):
        """
        Get a pandas DataFrame representing the data obtained from the Source.


        Available methods include: None (no encoding), "ohe" (one-hot-encoding), "label" (label encoding).

        Args:
            attributes (list of str): Subset of attributes to pick
            categorical_encoding (str): Method to encode categorical variables
            discrete_encoding (str): Method to encode discrete variables
            specific_encoding (dict of str: str): Mapping to define encoding for variables, with precedence over
                                                  the other kwargs.
            date_encoding (bool or str or list of str): parse date types to be compatible with general models.
                                                Available methods are:
                                                'drop': just remove the date column.
                                                'all': extract all of the available information.
                                                [list of str]: split the date column into separate columns, each one
                                                                containing a different attribute of the date. Valid
                                                                values include ['year', 'month', 'day', 'hour',
                                                                'minute', 'second', 'week', 'dayofweek', 'dayofyear'].
            imputation (dict of str to str): imputation strategy for each of the columns. If no strategy is given, na
                                             are allowed.
                                             Available methods are: 'mean', 'median', 'most_frequent' and
                                             'interpolation' (interpolates in the order of the dataframe).

        Returns:
            pandas.DataFrame: The resulting dataframe.

        """

        df = self._get_pandas()

        categorical = [c for c in self.types if self.types[c] == "categorical"]
        discrete = [c for c in self.types if self.types[c] == "discrete"]

        if attributes is not None:
            df = df[attributes]
            categorical = [c for c in categorical if c in attributes]
            discrete = [c for c in discrete if c in attributes]

        to_ohe = []

        if specific_encoding is None:
            specific_encoding = {}

        for c in specific_encoding:
            if specific_encoding[c] == "ohe":
                to_ohe.append(c)
            elif specific_encoding[c] == "label":
                df[c] = preprocessing.LabelEncoder().fit_transform(df[c])
            else:
                raise ValueError("Invalid encoding: %s" % specific_encoding[c])

        if categorical_encoding == "ohe":
            to_ohe += [c for c in categorical if c not in specific_encoding.keys()]
        elif categorical_encoding == "label":
            for c in self.types:
                if self.types[c] == "categorical" and c not in specific_encoding.keys():
                    df[c] = preprocessing.LabelEncoder().fit_transform(df[c])
        elif not categorical_encoding:
            # Any False value -> no encoding
            pass
        else:
            raise ValueError("Invalid categorical_encoding: %s" % categorical_encoding)

        if discrete_encoding == "ohe":
            to_ohe += [c for c in discrete if c not in specific_encoding.keys()]
        elif not discrete_encoding:
            # Any False value -> no encoding
            pass
        else:
            raise ValueError("Invalid discrete_encoding: %s" % discrete_encoding)

        if to_ohe:
            df = pd.get_dummies(df, columns=to_ohe)

        if imputation:
            for col, method in imputation.items():
                if method == 'interpolate':
                    df[col] = df[col].interpolate()
                elif method in ["mean", "median", "most_frequent"]:
                    df[col] = SimpleImputer(strategy=method).fit_transform(df[col].values.reshape(-1, 1))
                elif not method:
                    pass
                else:
                    raise ValueError("Invalid imputation method: " + str(method))

        return df

    def get_description(self):
        """Get a description of the source which can be stored somewhere"""
        d = collections.OrderedDict()

        d["types"] = self.types
        d["skip_type_inference"] = self.skip_type_inference
        # Which should be true since all types should be assumed known in a instance

        return d

    def save(self, name):
        """Store the source description in a file with the given name"""
        d = self.get_description()
        d["name"] = name
        with open(get_file_path(name + ".json", ["sources"]), 'w') as f:
            json.dump(d, f)

    @classmethod
    def load(cls, name):
        """Load a previously defined Source from the file system"""
        d = json.load(open(get_file_path(name + ".json", ["sources"])), object_pairs_hook=collections.OrderedDict)
        return Source.from_dict(d)

    @classmethod
    def from_dict(cls, d):
        """Create a new instance from its description"""
        if "class" not in d:
            raise ValueError("No source class specified in the file")
        if d["class"] not in _source_classes:
            raise ValueError("Unknown source class: %s" % d["class"])
        return _source_classes[d["class"]](**d)


class SourceCSV(Source):
    """A source of data which is a CSV file"""

    def __init__(self, path, sep=",", header="infer", **kwargs):
        """

        Args:
            path (str): Path to the CSV file.
            sep(str, default ','): Delimiter to use
            header(int, list of int, default 'infer'): Row number(s) to use as the column names, and the start of the
                                                       data

        """
        self.path = path
        self.subclass = "CSV"

        self.sep = sep
        self.header = header
        super().__init__(**kwargs)

    def get_description(self):
        d = super().get_description()
        d["class"] = "CSV"
        d["path"] = self.path
        d["sep"] = self.sep
        d["header"] = self.header
        return d

    def __str__(self):
        return "<SourceCSV %s>" % str(path.basename(self.path))

    def _get_pandas(self, **kwargs):
        return pd.read_csv(self.path, sep=self.sep, header=self.header)


class SourceExcel(Source):
    """A source of data which is an Excel file"""

    def __init__(self, path, **kwargs):
        """

        Args:
            path (str): Path to the Excel file.
        """
        self.path = path
        self.subclass = "Excel"
        # TODO: Sheet selection
        super().__init__(**kwargs)

    def get_description(self):
        d = super().get_description()
        d["class"] = "Excel"
        d["path"] = self.path
        return d

    def __str__(self):
        return "<SourceExcel %s>" % str(path.basename(self.path))

    def _get_pandas(self):
        return pd.read_excel(self.path)


def _uniquify(names):
    """Transform a list of names into a unique list of names (iterator)"""
    seen = set()

    for item in names:
        number = 1
        new_item = item

        while new_item in seen:
            number += 1
            new_item = "{}_{}".format(item, number)

        yield new_item
        seen.add(new_item)


class SourceJSON(Source):
    """A source of data which is a JSON file"""

    def __init__(self, path, **kwargs):
        """

        Args:
            path (str): Path to the file.

        """
        self.path = path
        self.subclass = "JSON"
        super().__init__(**kwargs)

    def get_description(self):
        d = super().get_description()
        d["class"] = "JSON"
        d["path"] = self.path
        return d

    def __str__(self):
        # Note this works also when the path is an URL
        return "<SourceJSON %s>" % str(path.basename(self.path))

    def _get_pandas(self, **kwargs):
        return pd.read_json(path_or_buf=self.path)


_source_classes = {"CSV": SourceCSV, "Excel": SourceExcel, "JSON": SourceJSON}
