# SmartCLIDE SmartCLIDE Smart-assistant

The SmartCLIDE smart assistant has brought together the IDE assistant features within one component. Proposed models try to provide a learning algorithm with the information that data carries, including internal data history and external online web services identified from online resources.
 After providing AI models, the smart assistant and DLE models are deployed as APIs REST encapsulated in a Python package, which guarantees the portability of this component between Operating Systems. Afterward, to expose the component functionality, we have chosen to visualize these requests and responses through the API swagger, which consists of an interactive web interface.



- [smartclide-dle](https://github.com/eclipse-opensmartclide/smartclide-smart-assistant/tree/main/smartclide-dle/smartclide-dle)
Flask-restx API developed to serve SmartCLIDE DLE (Deep Learning Engine), port 5001.

- [smartclide-smart-assistant](https://github.com/eclipse-opensmartclide/smartclide-smart-assistant/tree/main/smartclide-dle/smartclide-smart-assistant)
Flask-restx API developed to serve SmartCLIDE Smart Assistant, port 5000.

- [smartclide-template-code-generation](https://github.com/eclipse-opensmartclide/smartclide-smart-assistant/tree/main/smartclide-template-code-generation)
Component flow generation tool.
- [smartclide-dle-models](https://github.com/eclipse-opensmartclide/smartclide-smart-assistant/tree/main/smartclide-dle-models)
Repository with the models to be served by the DLE. It has automatic deployment so that being structured as packages, each change can be deployed automatically in the server where the DLE and SmartAssistant are deployed.




You can find detailed information [here](https://github.com/eclipse-opensmartclide/smartclide-smart-assistant/blob/main/docs/index.md)