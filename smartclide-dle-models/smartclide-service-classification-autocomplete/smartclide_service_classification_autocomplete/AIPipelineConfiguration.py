#!/usr/bin/python3
# Eclipse Public License 2.0

class AIPipelineConfiguration:
        """
        :param defaultDatasetsFolder:       string param specifies default dataset folder path
        :param defaultTrainedModelPath:     string param specifies default trained models folder path 
        :param defaultServiceDataset:       string param specifies default service dataset file
        :param defaultCodeTrainDataset:     string param specifies default codes dataset file
        :param defaultServiceTrainDataset:  string param specifies default service train dataset file
        :param defaultServiceTestDataset:   string param specifies default service test dataset file
        """
        epoch = 23
        defaultDatasetsFolder = "data/"
        defaultTrainedModelPath = "trained_models/"
        defaultServiceDataset='services_clustered_augmented_train.csv'
        defaultCodeTrainDataset="top_poject_source_codes.csv"
        defaultServiceTestDataset='services_clustered_test.csv'
        defaultServiceTrainDataset='services_clustered_augmented_train.csv'
