#!/usr/bin/python3
# Eclipse Public License 2.0

class AIPipelineConfiguration:
        """
        :param defaultDatasetsFolder:       string param specifies default dataset folder path
        :param defaultTrainedModelPath:     string param specifies default trained models folder path 
        :param defaultCodeTrainDataset:     string param specifies default codes dataset file
        """

        epoch = 2
        defaultDatasetsFolder = "data/"
        defaultTrainedModelPath = "trained_models/"
        defaultCodeTrainDataset="api_java_poject_source_codes.csv"

        code_generation_load_model= "Enabled"#"Enabled/Disable"
        code_generation_method= "Default"
        gpt_model="gpt2"



        def getDataDirectory(self):
            from servcodegen import getPackagePath
            packageRootPath = getPackagePath()
            return (packageRootPath + "/data/")
