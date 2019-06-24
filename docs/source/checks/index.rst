.. _checks:

Checks
======

.. toctree::
    :caption: Contents:
    :maxdepth: 1

    conditional_probabilities
    distinction
    simple
    sorensen_dice
    embedding_distance
    word_prediction
    guide


Morpheus comprises several methods to systematically compare two
data sets and check for a :term:`shift`. This section introduces each of
those methods, which we call :term:`check`.

Example Data Sets
+++++++++++++++++

+------------+------------+-----------+
| Header 1   | Header 2   | Header 3  |
+============+============+===========+
| body row 1 | column 2   | column 3  |
+------------+------------+-----------+
| body row 2 | Cells may  | column3   |
+------------+------------+-----------+
| body row 3 | Cells may  | Cells     |
+------------+------------+-----------+