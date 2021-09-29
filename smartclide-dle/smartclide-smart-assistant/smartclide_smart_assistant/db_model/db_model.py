#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.


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



