.. _embedding_distance:

Embedding Distance
==================

Description
-----------

This check calculates the distance between text embeddings, that are generated using machine learning.
The higher the distance, the more different are the datasets.

The check only works on textual data.

Example
-------

This section shows you how to use :ref:`embedding_distance` and interpret its result.

Code
++++

::

    from morpheus.detector import Detector
    from morpheus.checks.embedding_distance_check import EmbeddingDistanceCheck

    data_set_1 = 'examples/pokedex1.csv'
    data_set_2 = 'examples/pokedex2.csv'

    detector = Detector(
        data_set_1,
        data_set_2,
        delimiter=','
    )

    detector.run(EmbeddingDistanceCheck(model='word2vec'))
    detector.evaluate()

The code works as follows:

1. First, you create a :class:`~morpheus.detector.Detector` object to tell Morpheus
   which data sets you want to compare
2. Then, you start the detector with
   :meth:`~morpheus.detector.Detector.run` and the checks you want to run: in this case
   :class:`~morpheus.checks.embedding_distance_check.EmbeddingDistanceCheck`.
3. Finally, you print the result with
   :meth:`~morpheus.detector.Detector.evaluate`

Result
++++++

:ref:`embedding_distance` produces the following output:

::

    Embedding Distance Check
    Examined Columns: ['Name', 'Type 2', 'Type 1', 'Entry']
    Shifted Columns: ['Type 2', 'Entry']
            Baseline in Dataset 1  Baseline in Dataset 2  Distance between Datasets
    Name                 0.006821               0.012077                   0.015904
    Type 2               0.017263               0.138629                   0.003874
    Type 1               0.334552               0.402841                   0.086517
    Entry               11.298921               1.835465                  20.921940

Interpretation
++++++++++++++

The above report can be read as follows:

1. The examined columns are ``Name``, ``Type 2``, ``Type 1`` and ``Entry``
2. The columns that contain a shift according to the Sørensen Dice Check are ``Type 2`` and ``Entry``
3. The distance between the items within Dataset1 is at 11.299 in the column ``Entry``
4. The distance between the items within Dataset2 is at 1.835 in the column ``Entry``
5. The distance between Dataset1 and Dataset2 in the column ``Entry`` is at 20.922


Parameters
----------

:ref:`embedding_distance` provides the following parameters:

``model``:
    This parameter expects a string that describes the algorithm, the embeddings are generated with. 
    Possible values are ``word2vec`` [MCCD13]_. , ``fasttext`` [JGBM17]_. and ``None``. If ``model`` is ``None``, a trained model must be provided. 

``trained_model``:
    This parameter expects a trained gensim model, which will be used instead of training a new model.

Implementation
--------------

Algorithm
+++++++++

:ref:`embedding_distance` works as follows:

1. A machine learning model is trained on all texts of the examined column.
2. For each word in the examined column an embedding is calculated using the machine learning model.
3. All embeddings of a column of a dataset are added and divided by their overall quantity.
4. The euclidean distance between the two resulting embeddings (one for each dataset and column) is calculated. 

Notes
+++++

It's recommended to use the 'word2vec' embedding since it performs better.

References
----------

.. [MCCD13] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013a. Efficient Estimation of Word Representations in Vector Space. In ICLR Workshop Papers.
.. [JGBM17] Joulin, A., Grave, E., Bojanowski, P., and Mikolov, T. 2017. Bag of tricks for efficient text classification. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL).