FROM ubuntu:20.04

MAINTAINER AIR Institute "dberrocal@air-institute.com"

WORKDIR /app

# tzdata
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Main apt stuff
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \    
    python3.8-dev \
    python3.8-distutils \
    python-dev \
    libpython3.8-dev \
    libevent-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    ca-certificates \
    build-essential \
    wget \
    gcc \
    g++ \
    git \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# get pip
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py

# update pip
RUN python3.8 -m pip install --no-cache-dir -U pip
RUN python3.8 -m pip install --no-cache-dir cython

# install spacy and npl model
RUN python3.8 -m pip install --no-cache-dir setuptools wheel  && \
    python3.8 -m pip install --no-cache-dir spacy==2.3.5 && \
    python3.8 -m spacy download en_core_web_md

# clone smartclide-smart-assistant
RUN git clone https://github.com/eclipse-opensmartclide/smartclide-smart-assistant.git

# build cbr-gherkin-recommendation
RUN cd smartclide-smart-assistant/smartclide-dle-models/cbr-gherkin-recommendation && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt && \
    python3.8 -m pip install . --upgrade

# build smartclide-service-classification
RUN cd smartclide-smart-assistant/smartclide-dle-models/serviceclassification && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt && \
    python3.8 -m pip install . --upgrade

# build smartclide-service-autocomplete
RUN cd smartclide-smart-assistant/smartclide-dle-models/codeautocomplete && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt && \
    python3.8 -m pip install . --upgrade

# build smartclide-template-code-generation
RUN cd smartclide-smart-assistant/smartclide-template-code-generation && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt && \
    python3.8 -m pip install . --upgrade

# smartclide-dle and smartclide-smart-assistant
# Install extra requirements for the smart-assistant
RUN python3.8 -m pip install tensorflow nlpaug sentence_transformers==0.3.7.2 && \
    python3.8 -m pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    python3.8 -m pip install git+https://github.com/Dih5/zadeh

# build smartclide-dle
RUN cd smartclide-smart-assistant/smartclide-dle/smartclide-dle && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt && \
    python3.8 -m pip install . --upgrade

# build smartclide-smart-assistant
RUN cd smartclide-smart-assistant/smartclide-dle/smartclide-smart-assistant && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt && \
    python3.8 -m pip install . --upgrade

# Expose ports, you can override/map them at runtime by using the -p flag or ports
EXPOSE 5001 5000
