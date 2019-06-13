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

This section shows you how to use :ref:`sorensen_dice` and interpret its result.

Code
++++

::

    from Morpheus.Detector import Detector
    from Morpheus.checks.EmbeddingDistanceCheck import EmbeddingDistanceCheck

    data_set_1 = 'examples/shoes_first.csv'
    data_set_2 = 'examples/shoes_second.csv'

    detector = Detector(
        data_set_1,
        data_set_2,
        delimiter=','
    )

    detector.run(EmbeddingDistanceCheck(model='word2vec'))
    detector.evaluate()

The code works as follows:

1. First, you create a :class:`~Morpheus.Detector.Detector` object to tell Morpheus
   which data sets you want to compare
2. Then, you start the detector with
   :meth:`~Morpheus.Detector.Detector.run` and the checks you want to run: in this case
   :class:`~Morpheus.checks.SorensenDiceCheck.EmbeddingDistanceCheck`.
3. Finally, you print the result with
   :meth:`~Morpheus.Detector.Detector.evaluate`

Result
++++++

The :ref:`embedding_distance` produces the following output:

::

    Embedding Distance Check
    Examined Columns: ['description', 'item_name']
    Shifted Columns: ['description', 'item_name']
    Column 'description':
        Baseline in Dataset1: 13.447
        Baseline in Dataset2: 219.189
        Distance between Datasets: 235.404
    Column 'item_name':
        Baseline in Dataset1: 2.412
        Baseline in Dataset2: 25.603
        Distance between Datasets: 50.208

Interpretation
++++++++++++++

The above report can be read as follows:

1. The examined columns are 'describtion' and 'item_name"
2. The columns that contain a shift according to the Sørensen Dice Check are 'description' and 'item_name'
3. The distance between the items within Dataset1 is at 13.447 in the column 'description'
4. The distance between the items within Dataset2 is at 219.189 in the column 'description'
5. The similarity of Dataset1 and Dataset2 in the column 'description' is at 235.405


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

:ref:`sorensen_dice` works as follows:

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