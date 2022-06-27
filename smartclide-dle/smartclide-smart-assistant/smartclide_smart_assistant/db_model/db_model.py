#!/usr/bin/python3
#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************


from pymongo import MongoClient
from smartclide_smart_assistant import config


class DBModel:
        
    def __init__(self):
        self.m = {}

    def store(self, table, _id, data):

        collection = MongoClient(config.MONGO_URI)[config.MONGO_DB][table]

        collection.update({
            "id": _id
        },{
            "$push": {
                "data": data
            }
        }, upsert=True)


    def load(self, table, _id):

        collection = MongoClient(config.MONGO_URI)[config.MONGO_DB][table]

        result = collection.find_one({"id": _id})
        result = [] if result is None else result['data']

        collection.delete_one({"id": _id})

        return result



