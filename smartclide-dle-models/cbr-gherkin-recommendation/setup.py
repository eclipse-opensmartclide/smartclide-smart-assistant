#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script"""

import os
import io
from setuptools import setup, find_packages


def requirements(filename):
    reqs = list()
    with io.open(filename, encoding='utf-8') as f:
        for line in f.readlines():
            reqs.append(line.strip())
    return reqs


setup(author="Dih5",
      author_email='dihedralfive@gmail.com',
      description='Package to implement Case-Based Reasoning systems',
      keywords=[],
      name='pycbr',
      packages=find_packages(),
      python_requires='>=3',
      install_requires=list(requirements('requirements.txt')),
        extras_require={
            "docs": requirements(filename='docs/requirements.txt')
        },
      url='https://github.com/dih5/pycbr',
      version='0.1.1',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)'
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      )
