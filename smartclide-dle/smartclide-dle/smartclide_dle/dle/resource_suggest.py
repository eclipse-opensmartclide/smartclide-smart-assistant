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


import zadeh
import itertools
import pandas as pd
from typing import Dict, List, Tuple, Any, Generator


class FuzzyRamModel:

  def __init__(self):
    self.fis, self.variables, self.rules = self._build_model()

  def _define_variables(self) -> Dict:
    variables = {}

    variables['ram'] = zadeh.FuzzyVariable(
        zadeh.FloatDomain('ram', 0, 1024, 1024),
        {
            '4gb': zadeh.TrapezoidalFuzzySet(0,0,4,4),
            '8gb': zadeh.TrapezoidalFuzzySet(4,4,8,8),
            '16gb': zadeh.TrapezoidalFuzzySet(8,8,16,16),
            '32gb': zadeh.TrapezoidalFuzzySet(16,16,32,32),
            '64gb': zadeh.TrapezoidalFuzzySet(32,32,64,64),
            '128gb': zadeh.TrapezoidalFuzzySet(64,64,128,128),
            '256gb': zadeh.TrapezoidalFuzzySet(128,128,256,256),
            '512gb': zadeh.TrapezoidalFuzzySet(256,256,512,512),
            '1Tb': zadeh.TrapezoidalFuzzySet(512,512,1024,1024),
        }
    )

    # divided all by 100000
    variables['user_volume'] = zadeh.FuzzyVariable(
        zadeh.FloatDomain('user_volume', 0, 10000, 10000),
        {
            '10U': zadeh.TrapezoidalFuzzySet(0,0,0.0001,0.0001),
            '100U': zadeh.TrapezoidalFuzzySet(0.0001,0.0001,0.001,0.001),
            '1K': zadeh.TrapezoidalFuzzySet(0.001,0.001,0.01,0.01),
            '10K': zadeh.TrapezoidalFuzzySet(0.01,0.01,0.1,0.1),
            '100K': zadeh.TrapezoidalFuzzySet(0.1,0.1,1,1),
            '1M': zadeh.TrapezoidalFuzzySet(1,1,10,10),
            '10M': zadeh.TrapezoidalFuzzySet(10,10,100,100),
            '100M': zadeh.TrapezoidalFuzzySet(100,100,1000,1000),
            '1B': zadeh.TrapezoidalFuzzySet(1000,1000,10000,10000),
        },
    )

    variables['num_th'] = zadeh.FuzzyVariable(
        zadeh.FloatDomain('num_th', 0, 512, 512),
        {
            '2th': zadeh.TrapezoidalFuzzySet(0,0,2,2),
            '4th': zadeh.TrapezoidalFuzzySet(2,2,4,4),
            '8th': zadeh.TrapezoidalFuzzySet(4,4,8,8),
            '16th': zadeh.TrapezoidalFuzzySet(8,8,16,16),
            '32th': zadeh.TrapezoidalFuzzySet(16,16,32,32),
            '64th': zadeh.TrapezoidalFuzzySet(32,32,64,64),
            '128th': zadeh.TrapezoidalFuzzySet(64,64,128,128),
            '256th': zadeh.TrapezoidalFuzzySet(128,128,256,256),
            '512th': zadeh.TrapezoidalFuzzySet(256,256,512,512),
        }
    )

    return variables

  def _define_rules(self, variables):

      rule_set = zadeh.FuzzyRuleSet.automatic(
          variables['user_volume'], variables['ram'], reverse=False,  weight=0.8
      ) + zadeh.FuzzyRuleSet.automatic(
          variables['num_th'], variables['ram'], reverse=False,  weight=0.3
      ) + zadeh.FuzzyRuleSet.automatic(
          variables['ram'], variables['ram'], reverse=False,  weight=1.0
      )
      return rule_set

  def _build_model(self):

    variables = self._define_variables()
    rules = self._define_rules(variables)

    inputs = [
      variables['ram'], 
      variables['num_th'], 
      variables['user_volume']
    ]
    output = variables['ram']

    fis = zadeh.FIS(inputs, rules, output, defuzzification='centroid')

    return fis, variables, rules

  def _predict(self, ram, num_th, user_volume):
    prediction = self.fis.get_crisp_output(
      {
          'ram': ram,
          'num_th': num_th, 
          'user_volume': user_volume/100000
      })

    mappings = [4 ,8 ,16 ,32 ,64 ,128 ,256 ,512 ,1024]
    possible_mappings = [m for m in mappings if m >= prediction]
    possible_mappings = possible_mappings if possible_mappings else mappings 
    result = min(possible_mappings)

    return result

  def predict_one(self, ram, num_th, user_volume):

    resulting_ram = self._predict(ram, num_th, user_volume)

    return {
        'previous_ram': ram,
        'num_th': num_th,
        'user_volume': user_volume,
        'reccomended_ram': resulting_ram
    }

  def predict_multiple(self, ram, num_th, from_user_volume) -> Generator[float, None, None]: 

    user_ranges = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
    user_ranges = [v for v in user_ranges if v >= from_user_volume]

    for v in user_ranges:
      resulting_ram = self._predict(ram, num_th, v)
      yield {
        'previous_ram': ram,
        'num_th': num_th,
        'user_volume': v,
        'reccomended_ram': resulting_ram
      }

class FuzzyDiskModel:

  def __init__(self):
    self.fis, self.variables, self.rules = self._build_model()

  def _define_variables(self) -> Dict:
    variables = {}

    # divided all by 1024
    variables['disk'] = zadeh.FuzzyVariable(
        zadeh.FloatDomain('disk', 0, 64, 64),
        {
            '256gb': zadeh.TrapezoidalFuzzySet(0,0,0.25,0.25),
            '512gb': zadeh.TrapezoidalFuzzySet(0.25,0.25,0.5,0.5),
            '1Tb': zadeh.TrapezoidalFuzzySet(0.5,0.5,1,1),
            '2Tb': zadeh.TrapezoidalFuzzySet(1,1,2,2),
            '4Tb': zadeh.TrapezoidalFuzzySet(2,2,4,4),
            '8gb': zadeh.TrapezoidalFuzzySet(4,4,8,8),
            '16Tb': zadeh.TrapezoidalFuzzySet(8,8,16,16),
            '32gb': zadeh.TrapezoidalFuzzySet(16,16,32,32),
            '64Tb': zadeh.TrapezoidalFuzzySet(32,32,64,64),
        }
    )

    # divided all by 100000
    variables['user_volume'] = zadeh.FuzzyVariable(
        zadeh.FloatDomain('user_volume', 0, 10000, 10000),
        {
            '10U': zadeh.TrapezoidalFuzzySet(0,0,0.0001,0.0001),
            '100U': zadeh.TrapezoidalFuzzySet(0.0001,0.0001,0.001,0.001),
            '1K': zadeh.TrapezoidalFuzzySet(0.001,0.001,0.01,0.01),
            '10K': zadeh.TrapezoidalFuzzySet(0.01,0.01,0.1,0.1),
            '100K': zadeh.TrapezoidalFuzzySet(0.1,0.1,1,1),
            '1M': zadeh.TrapezoidalFuzzySet(1,1,10,10),
            '10M': zadeh.TrapezoidalFuzzySet(10,10,100,100),
            '100M': zadeh.TrapezoidalFuzzySet(100,100,1000,1000),
            '1B': zadeh.TrapezoidalFuzzySet(1000,1000,10000,10000),
        },
    )

    variables['num_th'] = zadeh.FuzzyVariable(
        zadeh.FloatDomain('num_th', 0, 512, 512),
        {
            '2th': zadeh.TrapezoidalFuzzySet(0,0,2,2),
            '4th': zadeh.TrapezoidalFuzzySet(2,2,4,4),
            '8th': zadeh.TrapezoidalFuzzySet(4,4,8,8),
            '16th': zadeh.TrapezoidalFuzzySet(8,8,16,16),
            '32th': zadeh.TrapezoidalFuzzySet(16,16,32,32),
            '64th': zadeh.TrapezoidalFuzzySet(32,32,64,64),
            '128th': zadeh.TrapezoidalFuzzySet(64,64,128,128),
            '256th': zadeh.TrapezoidalFuzzySet(128,128,256,256),
            '512th': zadeh.TrapezoidalFuzzySet(256,256,512,512),
        }
    )

    return variables

  def _define_rules(self, variables):

      rule_set = zadeh.FuzzyRuleSet.automatic(
          variables['user_volume'], variables['disk'], reverse=False,  weight=0.8
      ) + zadeh.FuzzyRuleSet.automatic(
          variables['num_th'], variables['disk'], reverse=False,  weight=0.3
      ) + zadeh.FuzzyRuleSet.automatic(
          variables['disk'], variables['disk'], reverse=False,  weight=1.0
      )
      return rule_set

  def _build_model(self):

    variables = self._define_variables()
    rules = self._define_rules(variables)

    inputs = [
      variables['disk'], 
      variables['num_th'], 
      variables['user_volume']
    ]
    output = variables['disk']

    fis = zadeh.FIS(inputs, rules, output, defuzzification='centroid')

    return fis, variables, rules

  def _predict(self, disk, num_th, user_volume):
    prediction = self.fis.get_crisp_output(
      {
          'disk': disk/1024,
          'num_th': num_th, 
          'user_volume': user_volume/100000
      })

    mappings = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    possible_mappings = [m for m in mappings if m >= prediction]
    possible_mappings = possible_mappings if possible_mappings else mappings 
    result = int(min(possible_mappings) * 1024)

    return result

  def predict_one(self, disk, num_th, user_volume):

    resulting_disk = self._predict(disk, num_th, user_volume)

    return {
        'previous_disk': disk,
        'num_th': num_th,
        'user_volume': user_volume,
        'reccomended_disk': resulting_disk
    }

  def predict_multiple(self, disk, num_th, from_user_volume) -> Generator[float, None, None]: 

    user_ranges = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
    user_ranges = [v for v in user_ranges if v >= from_user_volume]

    for v in user_ranges:
      resulting_disk = self._predict(disk, num_th, v)
      yield {
        'previous_disk': disk,
        'num_th': num_th,
        'user_volume': v,
        'reccomended_disk': resulting_disk
      }


class ResourceSuggest:

    def __init__(self):
        self.ram_model = FuzzyRamModel()
        self.disk_model = FuzzyDiskModel()
    
    def predict(self, ram:int, disk:int, num_thread:int, initial_user_volume:int) -> dict:

        # perform predictions
        ram_predictions = self.ram_model.predict_multiple(ram=ram,num_th=num_thread,from_user_volume=initial_user_volume)
        disk_predictions = self.disk_model.predict_multiple(disk=disk,num_th=num_thread,from_user_volume=initial_user_volume)

        # ensamble predictions
        for ram_object, disk_object in itertools.zip_longest(ram_predictions, disk_predictions):

            obj = {
                'user_volume': ram_object['user_volume'],
                'memory': ram_object['reccomended_ram'],
                'space': disk_object['reccomended_disk']
            }

            yield obj
