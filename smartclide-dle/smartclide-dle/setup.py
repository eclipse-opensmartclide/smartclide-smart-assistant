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


import io

from setuptools import setup, find_packages


def readme():
    with io.open('README.md', encoding='utf-8') as f:
        return f.read()

def read_requeriments_file(filename):
    with io.open(filename, encoding='utf-8') as f:
        for line in f.readlines():
            yield line.strip()


setup(
    name='smartclide-dle',
    version='2.0',
    packages=find_packages(),
    url='https://github.com/air-institute/smartclide-dle',
    download_url='https://github.com/air-institute/smartclide-dle/archive/master.zip',
    license='GNU Affero General Public License v3',
    author='AIR Institute',
    author_email='franpintosantos@usal.es',
    description=' ',
    long_description=readme(),
    long_description_content_type='text/markdown',
    install_requires=list(read_requeriments_file('requirements.txt')),
    entry_points={
        'console_scripts': [
            'smartclide-dle=smartclide_dle.run:main'
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Intended Audience :: Developers"
    ],
    keywords='IntelWines, AI, flask, python',
    python_requires='>=3',
    project_urls={
        'Bug Reports': 'https://github.com/air-institute/smartclide-dle/issues',
        'Source': 'https://github.com/air-institute/smartclide-dle',
        'Documentation': 'https://github.com/air-institute/smartclide-dle'
    },
)
