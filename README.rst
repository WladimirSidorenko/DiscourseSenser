Discourse Senser
================
.. image:: https://travis-ci.org/WladimirSidorenko/DiscourseSenser.svg?branch=master
   :alt: Build Status
   :target: https://travis-ci.org/WladimirSidorenko/DiscourseSenser

.. image:: https://codecov.io/gh/WladimirSidorenko/DiscourseSenser/branch/master/graph/badge.svg
   :alt: Code Coverage
   :target: https://codecov.io/gh/WladimirSidorenko/DiscourseSenser

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :alt: MIT License
   :align: right
   :target: http://opensource.org/licenses/MIT

Sense disambiguation of discourse connectives for PDTB-style shallow
discourse parsing.


Description
-----------

This package provides core functionality for sense disambiguation of
explicit and implicit discourse connectives for PDTB-like discourse
parsing.  It has been created for the CoNLL-2016 shared task.

The main package **dsenser** currently comprises the following
classifiers which can be trained either individually or bundled into
ensembles:

**dsenser.major.MajorSenser**
  a simplistic classifier which returns the conditional probabilities
  of senses given the connective;

**dsenser.wang.WangSenser**
 an optimized reimplementation of `Wang et al.`_'s sense classification
 system using the LinearSVC classifier;

**dsenser.xgboost.XGBoostSenser**
 an optimized reimplementation of `Wang et al.`_'s sense classification
 system using the XGBoost decision forrest classifier;

**dsenser.svd.SVDSenser**
 a neural network classifier which uses the SVD decomposition of word
 embedding matrices of the arguments;

**dsenser.lstm.LSTMSenser**
 a neural network classifier which uses an LSTM recurrence with
 Bayesian dropout (cf. `Yarin Gal, 2016`_).

Installation
------------

To install this package, you need to checkout this git-project with
its submodules by subsequently running the following commands:

.. code-block:: shell

    # initialize the git project
    git clone git@github.com:WladimirSidorenko/DiscourseSenser.git
    cd DiscourseSenser
    git submodule init
    git submodule update

    # download the skip-gram word embeddings and store them to `dsenser/data/`
    wget http://angcl.ling.uni-potsdam.de/data/GoogleNews-vectors-negative300.bin.gz -O \
    dsenser/data/GoogleNews-vectors-negative300.bin.gz

    gunzip dsenser/data/GoogleNews-vectors-negative300.bin.gz

    # download the pre-trained models and store them to `dsenser/data/models`
    wget http://angcl.ling.uni-potsdam.de/data/pdtb.models.tgz

    tar -xzf pdtb.models.tgz -C dsenser/data/models

    # Beware, since this package is constantly being improved, the
    # most recent version might not be fully compatible in terms of
    # features  with the models we trained for the submission.  In
    # this case, we recommend you check out our evaluated version
    # by running the following command:
    # ``git checkout conll-asterisk-evaluation``

    # finally, install the package in an editable mode (no copying will be
    # required in this case)
    pip install --user -r requirements.txt -e .

To ease the installation process, we are currently working on creating
a `wheel`_ for this package, but are facing some problems due to the
large size of the included word embedding file which requires the
``zip64`` extension.

Usage
-----

After installation, you can import the module in your python scripts, e.g.:

.. code-block:: python

    from dsenser import DiscoureSenser

    ...

    senser = DiscoureSenser(None)
    senser.train(train_set, dsenser.WANG | dsenser.XGBOOST | dsenser.LSTM,
                 path_to_model, dev_set)

or, alternatively, you can also use the delivered front-end script
``pdtb_senser`` to process your input data, e.g.:

.. code-block:: shell

    pdtb_senser train --type=2 --type=8 path/to/train_dir

    pdtb_senser test path/to/input_dir path/to/output_dir

The data in the specified folders should be in the ConNLL JSON format,
and include the files ``parses.json`` and ``relations.json`` for
training, and ``parses.json`` and ``relations-no-senses.json`` for the
testing mode.  Alternatively, you can also specify a different input
relations file whose senses need to be predicted by using the option
``pdtb_senser test --rel-file=REL_FILE INPUT_DIR OUTPUT_DIR``.

Reproducibility
---------------

In order to reproduce our *asterisk results from the CoNLL Shared Task
submission, you need to repeat the steps described in Section
`Installation`_, but additionally run the checkout command to obtain
exactly the version that we were using for the evaluation:

.. code-block:: shell

    git checkout conll-asterisk-evaluation

Acknowledgment
--------------

We gratefuly acknowledge the contribution of

* `Tatjana Scheffler`_ who extended the original features of `Wang et al.`_

.. _`the author`: mailto:sidarenk@uni-potsdam.de
.. _`Wang et al.`: https://github.com/lanmanok/conll2015_discourse
.. _`Yarin Gal, 2016`: http://arxiv.org/abs/1512.05287
.. _`wheel`: https://pypi.python.org/pypi/wheel
.. _`Tatjana Scheffler`: http://www.ling.uni-potsdam.de/~scheffler/
