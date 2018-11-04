#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

requirements = [
    'keras',
    'sklearn',
]

setup(
    name='seqvectorizer',
    version='0.0.1',
    description="Turns token sequences into dense vectors of fixed size",
    author="Sergey Zuev",
    url='https://github.com/zergey/seqvectorizer',
    packages=['seqvectorizer'],
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
