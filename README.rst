================
Discourse Senser
================

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :alt: MIT License
   :align: right
   :target: http://opensource.org/licenses/MIT

Disambiguation of discourse connectives for PDTB-style shallow
discourse parsing.


Description
===========

This package provides core functionality for sense disambiguation of
explicit and implicit discourse connectives for PDTB-like discourse
parsing.  It is being created for the CoNLL-2016 shared task.

The main package **dsenser** currently comprises the following
subpackages:

**scorer**
 a collection of standard CoNLL scorers for estimating the quality of
 sense disambiguation (is retrieved as submodule).

Installation
============

To install this package, you need to checkout this git-project with
its submodules and subsequently run the following commands:

.. code-block:: shell

    git clone git@github.com:WladimirSidorenko/DiscourseSenser.git
    cd DiscourseSenser
    git submodule init
    git submodule update

    # download the `Skip-gram Neural Word Embeddings`_ and store the
    # unpacked archive at `dsenser/data/GoogleNews-vectors-negative300.bin`

    pip install -r requirements.txt . --user


Usage
=====

After installation, you can import the module in your python scripts, e.g.:

.. code-block:: python

    from dsenser import DiscoureSenser

    senser = DSenser()

or, alternatively, also use the delivered front-end script
`discourse_senser` to process your input data, e.g.:

.. code-block:: shell

    discourse_senser path/to/data_dir path/to/model_dir path/to/output_dir

Note that this script requires three mandatory arguments:

1. path to the directory with input PDTB data;
2. path to directory contaning pre-built models and auxiliary data;
3. path to the output directory.

See the official `CoNLL instructions`_ for complete specification of
the input and output format.

.. _`CoNLL instructions`: https://github.com/attapol/conll16st/blob/master/README.md
.. _`Skip-gram Neural Word Embeddings`: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
