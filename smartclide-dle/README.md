## SmartCLIDE DLE and Smart Asssistant Components

Flask-restx APIs developed to serve SmartCLIDE DLE (Deep Learning Engine) and Smart Assistant.

## Install prerequisites

```bash
sudo apt update
sudo apt install python3 python3-pip nodejs npm -y
sudo npm install -g pm2
```

## Architecture

The models served by the DLE are defined into the following repositories:
- Code skeleton generation via BPMN file** [(More info)](https://github.com/eclipse-opensmartclide/smartclide-smart-assistant/tree/main/smartclide-template-code-generation) 
- Various models, beign the most importants the CBR based gherking reccomendation with BPMN file, service classsification and code autocomplete generation [(More info)](https://github.com/eclipse-opensmartclide/smartclide-smart-assistant/tree/main/smartclide-dle-models
)  
<div align="center">
  <img src="https://github.com/AIRInstitute/smartclide-dle/blob/main/_static/architecture.jpg" hspace="20">
</div>

