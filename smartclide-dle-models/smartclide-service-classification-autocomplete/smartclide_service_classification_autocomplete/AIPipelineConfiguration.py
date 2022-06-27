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

class AIPipelineConfiguration:
        """
        :param defaultDatasetsFolder:       string param specifies default dataset folder path
        :param defaultTrainedModelPath:     string param specifies default trained models folder path 
        :param defaultServiceDataset:       string param specifies default service dataset file
        :param defaultCodeTrainDataset:     string param specifies default codes dataset file
        :param defaultServiceTrainDataset:  string param specifies default service train dataset file
        :param defaultServiceTestDataset:   string param specifies default service test dataset file
        """
        epoch = 2
        defaultDatasetsFolder = "data/"
        defaultTrainedModelPath = "trained_models/"
        defaultCodeTrainDataset="api_java_poject_source_codes.csv"
        defaultServiceDataset='services_clustered_augmented_train.csv'
        defaultServiceTestDataset='services_clustered_test.csv'
        defaultServiceTrainDataset='services_clustered_augmented_train.csv'


        def getDataDirectory(self):
            from smartclide_service_classification_autocomplete import getPackagePath
            packageRootPath = getPackagePath()
            return (packageRootPath + "/data/")
