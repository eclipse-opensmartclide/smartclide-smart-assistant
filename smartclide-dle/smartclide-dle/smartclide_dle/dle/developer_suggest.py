#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.

import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import interp1d


class CommitModel:
    
    DATA_PATH = "GitHub_api_09_07_2021 _ commits.csv"

    def __init__(self):
        self.df = self._preprocess(pd.read_csv(CommitModel.DATA_PATH))

    def _preprocess(self, df):
        # Extract json-like data
        # Note quotes must be fixed
        for label in ["total", "additions", "deletions"]:
            df[label] = df["sum_stats"].apply(lambda x: json.loads(x.replace("'", '"'))[label])
        # Assuming the dataset belongs to a single brach with a linear history,
        # extract the time in minutes
        df["inc_time"]=pd.to_datetime(df.sort_values("date")["date"]).diff().dt.seconds/60
        return df

    def ecdf(self, a):
        """Return lists of values xx, yy describing the Empirical Cummulative Distribution Function"""
        x, counts = np.unique(a, return_counts=True)
        cusum = np.cumsum(counts)
        return x, cusum / cusum[-1]

    def get_conviction_to_commit_from_files(self, number_of_files_modified):
        get_conviction_to_commit_from_files = interp1d(*self.ecdf(self.df["total"]))
        x = self.df["total"]
        y = get_conviction_to_commit_from_files(x)
        return get_conviction_to_commit_from_files(number_of_files_modified)

    def get_conviction_to_commit_from_time(self, time_since_last_commit):
        get_conviction_to_commit_from_time = interp1d(*self.ecdf(self.df["inc_time"]))
        x = self.df["inc_time"]
        y = get_conviction_to_commit_from_time(x)
        return get_conviction_to_commit_from_time(time_since_last_commit)

    def predict(self, number_of_files_modified, time_since_last_commit):
        conviction_to_commit_from_files = self.get_conviction_to_commit_from_files(number_of_files_modified)
        conviction_to_commit_from_time = self.get_conviction_to_commit_from_time(time_since_last_commit)
        return float(conviction_to_commit_from_files), float(conviction_to_commit_from_time)


internal_model = CommitModel()


class DeveloperSuggest:
    
    def predict(self, number_of_files_modified:int, time_since_last_commit:int) -> str:
        global internal_model
        conviction_to_commit_from_files, conviction_to_commit_from_time = internal_model.predict(number_of_files_modified, time_since_last_commit)
        return conviction_to_commit_from_files, conviction_to_commit_from_time

