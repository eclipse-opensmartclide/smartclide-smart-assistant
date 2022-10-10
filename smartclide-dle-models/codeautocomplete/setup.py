#!/usr/bin/python3
# Eclipse Public License 2.0

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
    name='servcodegen',
    version='1.0',
    packages=find_packages(),
    url='',
    download_url='https://github.com/eclipse-opensmartclide/smartclide-smart-assistant/archive/refs/heads/main.zip',
    license='Copyright',
    author='AIR institute',
    author_email='zakieh@usal.es',
    description='Utility to autocomplete service codes',
    long_description=readme(),
    long_description_content_type='text/markdown',
    install_requires=requirements(filename='requirements.txt'),
    include_package_data=True,
    package_data={'': ['*.csv']},
    classifiers=[
        "Development Status :: 1.0 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "License :: OSI Approved :: Eclipse Public License 2.0",
        "Intended Audience :: Developers" 
        "Topic :: Software Development "
    ],
    entry_points={
        'console_scripts': [
            'smartclide_wizard=smartclide_wizard.cli:main'
        ],
    },
    python_requires='>=3',
    keywords=' AI, flask, python',
    project_urls={
        'Bug Reports': 'https://github.com/eclipse-opensmartclide/smartclide-smart-assistant/issues',
        'Source': 'https://github.com/eclipse-opensmartclide/smartclide-smart-assistant',
        'Documentation': 'https://github.com/eclipse-opensmartclide/smartclide-smart-assistant/README.md'
    },
)

