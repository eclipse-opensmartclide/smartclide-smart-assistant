# [smartclide-dle](https://github.com/eclipse-researchlabs/smartclide-smart-assistant/tree/main/smartclide-dle/smartclide-dle)
Flask-restx API developed to serve SmartCLIDE DLE (Deep Learning Engine), port 5001.

# [smartclide-smart-assistant](https://github.com/eclipse-researchlabs/smartclide-smart-assistant/tree/main/smartclide-dle/smartclide-smart-assistant)
Flask-restx API developed to serve SmartCLIDE Smart Assistant, port 5000.

# [smartclide-dle-models](https://github.com/eclipse-researchlabs/smartclide-smart-assistant/tree/main/smartclide-dle-models)
Repository with the models to be served by the DLE. It has automatic deployment so that being structured as packages, each change can be deployed automatically in the server where the DLE and SmartAssistant are deployed.

- **cbr-gherkin-recommendation**: gherkin recommendation from BPMN files using case-based reasoning.
- **smartclide-service-classification-autocomplete**: service classification and code autocomplete based in markov-chains.

# [smartclide-template-code-generation](https://github.com/eclipse-researchlabs/smartclide-smart-assistant/tree/main/smartclide-template-code-generation)
Component flow generation tool.

# Deploy
This docker package contains two REST services that need to be executed as daemons or in a similar way, for this purpose you can use the attached docker-compose.yml file.

