#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Libraries
from distutils.core import setup
from os import path
import codecs
import glob

##################################################################
# Variables and Constants
ENCODING = "utf-8"

PWD = path.abspath(path.dirname(__file__))

LONG_DESCRIPTION = ""
with codecs.open(path.join(PWD, "README.rst"), encoding=ENCODING) as ifile:
    LONG_DESCRIPTION = ifile.read()

INSTALL_REQUIRES = []
with codecs.open(path.join(PWD, "requirements.txt"),
                 encoding=ENCODING) as ifile:
    for iline in ifile:
        iline = iline.strip()
        if iline:
            INSTALL_REQUIRES.append(iline)

TEST_REQUIRES = []
with codecs.open(path.join(PWD, "test-requirements.txt"),
                 encoding=ENCODING) as ifile:
    for iline in ifile:
        iline = iline.strip()
        if iline:
            TEST_REQUIRES.append(iline)

##################################################################
# setup()
setup(
    name="dsenser",
    version="0.0.1",
    description=("sense disambiguation for PDTB-style discourse parsing"),
    long_description=LONG_DESCRIPTION,
    author="Wladimir Sidorenko (Uladzimir Sidarenka)",
    author_email="sidarenk@uni-potsdam.de",
    license="MIT",
    url="https://github.com/WladimirSidorenko/DiscourseSenser",
    include_package_data=True,
    packages=["dsenser", "dsenser.scorer"],
    package_data={
        "dsenser": [path.join("data", fname) for fname in ()]
    },
    install_requires=INSTALL_REQUIRES,
    scripts=[path.join("scripts", "discourse_senser")],
    classifiers=["Development Status :: 2 - Pre-Alpha",
                 "Environment :: Console",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Natural Language :: English",
                 "Operating System :: Unix",
                 "Operating System :: MacOS",
                 # "Operating System :: Microsoft :: Windows",
                 "Programming Language :: Python :: 2",
                 "Programming Language :: Python :: 2.6",
                 "Programming Language :: Python :: 2.7",
                 # "Programming Language :: Python :: 3",
                 "Topic :: Text Processing :: Linguistic"],
    keywords="discourse disambiguation NLP linguistics"
)
