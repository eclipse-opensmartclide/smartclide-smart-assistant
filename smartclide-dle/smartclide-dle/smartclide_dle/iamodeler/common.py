#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************

import copy
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .files import get_file_path

logger = logging.getLogger(__name__)

_package_versions = {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


def _version_tuple(s):
    """String version to tuple. ONLY INTENDED FOR NUMERIC COMPONENTS."""
    return tuple(map(int, (s.split("."))))


class TrainingError(Exception):
    """Raised when an object could not be trained"""
    pass


class TrainingNotFinished(Exception):
    """Raised when an object is still being trained"""
    pass


class Storable:
    """
    Abstract class of objects which can be described by elemental types (perhaps by recurrence)

    NOTE:
        Subclasses MUST define the class attributes, which must be in accordance to their constructor.
        The constructor MUST store its kwargs with exactly the same name. Storable.__init__ might be called for that
        purpose.

    Attributes:
        class_kwargs: list of specific kwargs the subclass takes for construction.
        kwarg_mapping: map of some the kwargs with their expected class types which implement a get_description method.

    """
    # BEWARE: This class attributes must be modified by the inheriting subclass
    class_kwargs = []  # Example: ["source", "target", ...]
    kwarg_mapping = {}  # Example: {"source": Source}

    def __init__(self, **kwargs):
        for attribute, value in kwargs:
            setattr(self, attribute, value)

    def get_description(self):
        """
        Get a description of the instance configuration.

        Returns:
            dict: object description

        """
        d = {}
        for kwarg in self.class_kwargs:
            value = getattr(self, kwarg)
            if value is not None:
                if kwarg in self.kwarg_mapping:
                    d[kwarg] = value.get_description()
                else:
                    d[kwarg] = value
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Create an instance from a dict

        Args:
            d (dict): Data needed to create the instance

        Returns:
            A new instance of the class

        """
        d = copy.deepcopy(d)
        for item, _class in cls.kwarg_mapping.items():
            if _class is None:
                # Note _class might be None if a class has been monkey-patched to None
                # e.g.: NLPConfig with no text capabilities
                d[item] = None
            else:
                try:
                    if d[item] is not None:
                        d[item] = _class.from_dict(d[item])
                except KeyError:
                    # No value, which equals to None
                    pass

        c = cls(**d)

        return c


class Persistable(Storable):
    """
        Abstract class of objects which can be persisted, both by their description (so they are Storable) and by a
        pickle.

        Subclasses MUST define the class attributes.


        Attributes:
            class_kwargs: list of specific kwargs the subclass takes for construction.
            kwarg_mapping: map of some the kwargs with their expected class types which implement a get_description
                           method.
            category: hierarchy of folder where the instance is saved.

        """
    category = []  # Example: ["unsupervised", "clustering"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save(self, filename, status=None, status_description=None):
        """
        Store the instance information and the instance itself

        Args:
            filename (str): name in which the instance will be stored

        """
        description = self.get_description()
        description["_iam_version"] = __version__
        # If the subclasses defines a get_dependence method, add a "hidden" attribute
        try:
            description["_iam_packages"] = self.get_dependence()
        except AttributeError:
            pass

        if status:
            if status not in ["training", "error"]:
                raise ValueError('Status must in be in ["training", "error"].')
            description["_status"] = status

            if status_description:
                description["_status_description"] = status_description

            logger.info("Saving %s state for %s as %s" % (status, str(self), filename))
        else:
            logger.info("Saving %s as %s" % (str(self), filename))

        # Save the description
        with open(get_file_path(filename + ".json", self.category), 'w') as f:
            json.dump(description, f)

        # Save the instance itself
        if not status:
            pickle.dump(self, open(get_file_path(filename + ".pkl", self.category), "wb"))

    @classmethod
    def load(cls, filename, method=None, rebuild=False):
        """
        Load an instance from a system file or pickle

        Args:
            filename (str): name of the file in which the association is stored
            method (str): A method to assume the file was saved as. All will be tried if None.
                          For available options, see the save method.
            rebuild (bool): whether to recreate the object instead of unpickling it. This can be used to rebuild
                            a model to upgrade it to a new version. Note the fit method has to be called after, perhaps
                            as a celery task.

        Returns:
            Persistable: The loaded object.

        Raises:
            FileNotFoundError: if the file doesn't exist.
            TrainingError: An error occurred during the training.

        """
        d = cls.load_description(filename)
        logger.info("Loading an instance of %s" % filename)

        # Check out errors:
        status = d.get("_status")
        if status:
            if status == "training":
                raise TrainingNotFinished(d.get("_status_description"))
            elif status == "error":
                raise TrainingError(d.get("_status_description"))
            else:
                raise ValueError("Unknown status: %s." % status)

        # Check out package version
        if "_iam_version" not in d:
            logger.warning("iamodeler version not found in the object configuration. The loaded object might not work.")

        else:
            if _version_tuple(d["_iam_version"]) > _version_tuple(__version__):
                logger.warning(
                    "Loaded object was generated using a more recent version of iamodeler. "
                    "The loaded object might not work.")
                logger.warning("The iamodeler deployment should be updated.")

            # Behaviour for older versions should be defined here in the future.

        # Check out external package versions
        if "_iam_packages" in d:
            for key, value in d["_iam_packages"].items():
                if _package_versions[key] != value:
                    logger.warning(
                        "Package version does not match for %s: %s != %s. The loaded object might not work." % (
                            key, _package_versions[key], value,))
                pass

        # A new instance could be returned from the description, loosing all training.
        if rebuild:
            return cls.from_dict(d)

        if os.path.isfile(get_file_path(filename + ".pkl", cls.category)):
            loaded = pickle.load(open(get_file_path(filename + ".pkl", cls.category), "rb"))
        else:
            raise ValueError("No %s stored object found" % filename)
        return loaded

    @classmethod
    def load_description(cls, filename):
        """
        Load an the description of an instance from the system file

        Args:
            filename (str): name of the file in which the association is stored

        Returns:
            dict: The description.

        Raises:
            FileNotFoundError: if the file doesn't exist.

        """
        logger.info("Loading description of %s" % filename)

        with open(get_file_path(filename + ".json", cls.category)) as data:
            d = json.loads(data.read())

        return d
