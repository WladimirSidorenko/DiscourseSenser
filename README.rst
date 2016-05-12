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
parsing.  It has been created for the CoNLL-2016 shared task.

The main package **dsenser** currently comprises the following
classifiers which can be trained either separately or bundled into
ensembles:

**dsenser.major.MajorSenser**
  a simplistic classifier which returns the conditional probabilities
  of senses given the connective;

**dsenser.wang.WangSenser**
 an optimized reimplementation of `Wang. et al.`_'s sense classification
 system using the LinearSVC classifier;

**dsenser.xgboost.XGBoostSenser**
 an optimized reimplementation of `Wang. et al.`_'s sense classification
 system using the XGBoost decision forrest classifier;

**dsenser.svd.SVDSenser**
 a neural network classifier which uses the SVD decomposition of word
 embedding matrices of the arguments;

**dsenser.lstm.LSTMSenser**
 a neural network classifier which uses an LSTM recurrence with
 Bayesian dropout (cf. `Yarin Gal`_);

Installation
============

To install this package, you need to checkout this git-project with
its submodules and subsequently run the following commands:

.. code-block:: shell

    git clone git@github.com:WladimirSidorenko/DiscourseSenser.git
    cd DiscourseSenser
    git submodule init
    git submodule update

    # download the Skip-gram Neural Word Embeddings at
    # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
    # and store the unpacked archive at
    # `dsenser/data/GoogleNews-vectors-negative300.bin`

    pip install -r requirements.txt -E . --user

Beware, that this package does not include any pre-trained models.
Due to a big size of the serialized files, we cannot add them all to
this git project, but feel free to contact `the author`_ of this
program to obtain the PDTB models from him directly.

Usage
=====

After installation, you can import the module in your python scripts, e.g.:

.. code-block:: python

    from dsenser import DiscoureSenser

    ...

    senser = DiscoureSenser(None)
    senser.train(train_set, dsenser.WANG | dsenser.XGBOOST | dsenser.LSTM,
                 path_to_model, dev_set)

or, alternatively, you can also use the delivered front-end script
`pdtb_senser` to process your input data, e.g.:

.. code-block:: shell

    pdtb_senser train path/to/train_dir

    pdtb_senser test path/to/input_dir path/to/output_dir

Acknowledgment
==============

We gratefuly acknowledge the contribution of

* `Tatjana Scheffler`_ , who extended the original features of `Wang et al.`_.

.. _`the author`: mailto:sidarenk@uni-potsdam.de
.. _`Wang. et al.`: https://github.com/lanmanok/conll2015_discourse
.. _`Yarin Gal`: http://arxiv.org/abs/1512.05287
.. _`Skip-gram Neural Word Embeddings`: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
.. _`Tatjana Scheffler`: http://www.ling.uni-potsdam.de/~scheffler/
