# pycbr

Microframework to implement Case-Based Reasoning systems

## Installation

Assuming you have a [Python3](https://www.python.org/) distribution with [pip](https://pip.pypa.io/en/stable/installing/), to install a development version, cd to the directory with this file and:

```bash
python3 -m pip install . --upgrade
```

As an alternative, a virtualenv might be used to install the package:

```bash
# Prepare a clean virtualenv and activate it
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
# Install the package
python3 -m pip install . --upgrade
```

To install also the dependencies to run the tests or to generate the documentation install some of the extras like (Mind the quotes):

```bash
python3 -m pip install '.[docs]' --upgrade
```

## Case database initialization

For that purpose, use the following command:

```bash
python3 initialize_cbr_db.py
```

## Usage

The main class is CBR wich also needs the clases Casebase, Recovery and Aggregation. You need a frist load with all your base cases. After that first inicial load you can pass an empty array to the class initializer:

```python
import pycbr
cbr = pycbr.CBR([],"ghrkn_recommendator","smartclide.ddns.net")
```

### Add case

The method to add a case must recibe a dictionary with this format:

```python
cbr.add_case({
    'name': "Sting with the file name",
    'text': "All the bpmn text",
    'gherkins': ["list with gherkins text"]
})
```

### Get recommendation

The  method to get a recommendation must recibe a string with all the bpmn text:

```python
cbr.recommend(bpmn_text)
>>> {
        'gherkins': [["List of list with all the recomended gherkins for the first 5 matches"]],
        'sims': ["List of similarity scores from 0 to 1"]
    }
```

## Documentation

To generate the documentation, the *docs* extra dependencies must be installed. Furthermore, **pandoc** must be available in your system.

To generate an html documentation with sphinx run:
```bash
make docs
```

To generate a PDF documentation using LaTeX:
```bash
make pdf
```