# smartCLIDE-DLE
  Port  5001
# smartclide-smart-assistant
  Port  5000

This docker package contains two REST services that need to be executed as daemons or in a similar way, for this purpose you can use the following docker-compose:
```
version: '3'

services:
  smartclide_dle:
    restart: unless-stopped
    build: .
    working_dir: /app/smartclide-smart-assistant/smartclide-dle/smartclide-dle
    command: smartclide-dle
    ports:
      - "5001:5001"
  smartclide_smart_assistant:
    restart: unless-stopped
    build: .
    working_dir: /app/smartclide-smart-assistant/smartclide-dle/smartclide-smart-assistant
    command: smartclide-smart-assistant
    ports:
      - "5000:5000"
```
