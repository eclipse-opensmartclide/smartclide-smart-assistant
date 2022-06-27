#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************

import logging

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded

from .config import config
from .models import Predictor
from .unsupervised import Clustering

logger = logging.getLogger(__name__)

celery = Celery(
    "iamodeler",
    broker=config["celery_broker"],
)

if config["celery"]:
    # Set up a celery queue
    celery.conf.CELERY_ALWAYS_EAGER = False
    celery.conf.CELERY_EAGER_PROPAGATES_EXCEPTIONS = False
    logger.info("Celery queue is set")
else:
    # Set EAGER execution
    celery.conf.CELERY_ALWAYS_EAGER = True
    celery.conf.CELERY_EAGER_PROPAGATES_EXCEPTIONS = True
    logger.info("Eager execution is set")

# Persistables which can be trained (must provide a fit method)
_persistables = {"predictor": Predictor,
                 "clustering": Clustering,
                 }


@celery.task(ignore_result=True)
def train_persistable(data, category):
    """
    Creates a Predictor from a json object

    Args:
        data(dict): json data needed to create a Predictor
        category (str): Category of the persistable.

    """
    error = None
    p = None
    # Note a timeout failure could be reached before saving p. Hence, this should be done outside the task.
    # p.save(filename, status="training")
    try:
        # Note a timeout or other exception could happen before creating p
        p = _persistables[category].from_dict(data)
        p.fit()
        p.save(data["filename"])
        return
    except SoftTimeLimitExceeded:
        logger.error("Timeout training model %s" % data["filename"])
        # Make sure the objects are created
        if p is None:
            p = _persistables[category].from_dict(data)
        p.save(data["filename"], status="error", status_description="Timeout")
        return
    except Exception as e:
        # Retry outside to prevent raising again in this handler
        logger.error("Error training model %s: %s" % (data["filename"], str(e)))
        error = str(e)

    if p is None:
        p = _persistables[category].from_dict(data)
    p.save(data["filename"], status="error", status_description=error)
