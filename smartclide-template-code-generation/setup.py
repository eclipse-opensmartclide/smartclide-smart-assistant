#!usr/bin/python

# Copyright 2021 AIR Institute
# See LICENSE for details.


import io
from setuptools import setup, find_packages


def readme():
    with io.open('README.md', encoding='utf-8') as f:
        return f.read()


def requirements(filename):
    reqs = list()
    with io.open(filename, encoding='utf-8') as f:
        for line in f.readlines():
            reqs.append(line.strip())
    return reqs


setup(
    name='smartclide_wizard',
    version='1.0',
    packages=find_packages(),
    url='https://github.com/AIRInstitute/smartclide-wizard',
    download_url='https://github.com/AIRInstitute/smartclide-wizard/archive/master.zip',
    license='Copyright',
    author='AIR institute',
    author_email='franpintosantos@air+institute.es',
    description='Utility to generate components using BPMN diagrams.',
    long_description=readme(),
    long_description_content_type='text/markdown',
    install_requires=requirements(filename='requirements.txt'),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries"
    ],
    entry_points={
        'console_scripts': [
            'smartclide_wizard=smartclide_wizard.cli:main'
        ],
    },
    python_requires='>=3',
    keywords=', '.join([
        'Apache Kafka', 'Kafka', 'Code Generation', 'cookiecutter'
    ]),
    project_urls={
        'Bug Reports': 'https://github.com/AIRInstitute/smartclide-wizard/issues',
        'Source': 'https://github.com/AIRInstitute/smartclide-wizard',
        'Documentation': 'https://github.com/AIRInstitute/smartclide-wizard'
    }
)