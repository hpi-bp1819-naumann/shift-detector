.. _distinction:

Distinction
===========

Description
-----------

This check investigates shifts between two data sets with the help of
a machine learning.

The result not only shows how good the model can distinguish the data sets.
It also shows which columns the model looks at when trying to distinguish the data sets.
These columns will contain a shift.

Example
-------

This section shows you how to use :ref:`distinction` and interpret
its result.

Code
++++

::

    from shift_detector.Detector import Detector
    from shift_detector.checks.DistinctionCheck import DistinctionCheck

    data_set_1 = 'examples/shoes_first.csv'
    data_set_2 = 'examples/shoes_second.csv'

    detector = Detector(
        data_set_1,
        data_set_2,
        delimiter=','
    )

    detector.run(DistinctionCheck())
    detector.evaluate()

The code works as follows:

1. First, you create a :class:`~shift_detector.Detector.Detector` object to tell Morpheus
   which data sets you want to compare.
2. Then, you start the detector with
   :meth:`~shift_detector.Detector.Detector.run` and the checks you want to run: in this case
   :class:`~shift_detector.checks.DistinctionCheck.DistinctionCheck`.
3. Finally, you print the result with
   :meth:`~shift_detector.Detector.Detector.evaluate`

Result
++++++

:ref:`distinction` produces the following report::

    Examined Columns: ['brand', 'payment', 'description']
    Shifted Columns: ['payment', 'description']

    Column 'brand':
    0.890625 -> 0.8375
    Column 'payment':
    0.890625 -> 0.7015625
    Column 'description':
    0.890625 -> 0.5671875

    'Classification Report':
                  precision    recall  f1-score   support

               A       1.00      0.62      0.77        16
               B       0.73      1.00      0.84        16

       micro avg       0.81      0.81      0.81        32
       macro avg       0.86      0.81      0.81        32
    weighted avg       0.86      0.81      0.81        32

    'F1 score df1':
    0.7692307692307693
    'F1 score df2':
    0.8421052631578948

Interpretation
++++++++++++++

The above report can be read as follows:

1. The examined columns are 'brand', 'payment', 'description'.
2. The columns that contain a shift according to the distinction check are
   'payment' and 'description'.
3. The accuracy when the column 'brand' was altered through our algorithm dropped
   from 89,06% to 83,75%. Columns 'payment' and 'description' respectively.
4. A report of the underlying machine learning. How good was it able to identify an
   entry from the ``data_set_1``, ``data_set_2``, etc.

This tells you that:

1. That there is a significant shift in the two data set.
2. The column that is most responsible for the shift is 'description' and then
   'payment'. The column 'brand' does not fall under the threshold and is therefore
   not considered shifted.

Parameters
----------

:ref:`distinction` provides the following parameter in order to improve
    the run time and the quality of the result:

.. _columns:

``columns``:
    This parameter expects a list of strings. These strings are the name of
    the columns that you want to inspect. If no columns are provided all columns
    of the data sets are used for machine learning.

``num_epochs``:
    This parameter expects an integer greater than 0 and defines the number of
    epochs the machine learning model will train. The default value is 10 epochs.

``relative_threshold``:
    This parameter expects a float between 0 and 1. If the altered column leads
    to a drop in the accuracy that falls below the relative threshold compared to
    the base accuracy the column contains a shift.

Implementation
--------------

Algorithm
+++++++++

:ref:`distinction` works as follows:

1. Every entry in the first data set receives the label 'A' and
   every entry in the second data set receives the label 'B'.
2. The labeled data sets are connected and shuffled in order to create data for
   training.
3. An data imputer is trained, that tries to label each entry in the training data set
   with label 'A' or 'B' based on the values in that entry. Therefore, the model needs to
   find features that are indicators for one of the data sets.
4. For each column we want to find out if they contain shift. To do so, we alter each
   column one after another and investigate the accuracy compared to a base accuracy that
   was calculated when no change was injected. If the accuracy drops significantly
   (below a certain threshold in respect to the base accuracy) the column was used from
   the model to distinguish between the data sets. This means that this column contains
   shift.
   A column is altered the following way. Shuffle the column in both data sets
   individually and switch the column between the data sets.

Notes
+++++

For the machine learning model we use an imputer from datawig_.

The method we use in order to investigate the change in the data sets is based
on the idea in [Shohei2008]_. We developed this algorithm further to investigate
the shift in a specific column.

The algorithm can contain the following problem and we advice to run the check multiple
times with changed parameters in order to receive a sufficient result:

A column can be sufficient for the model to distinguish between the two data sets. Even
though, other columns can contain shifts, too. The model overfits and the algorithm will
only detect a significant shift in this column without considering the other columns.
We advice to run another check without this column with the help of the columns_ parameter.


References
----------

.. [Shohei2008] Shohei Hido, Tsuyoshi Idé, Hisashi Kashima, Harunobu Kubo,
   and Hirofumi Matsuzawa. 2008. Unsupervised change analysis using supervised learning.
   In Proceedings of the 12th Pacific-Asia conference on Advances in knowledge discovery
   and data mining (PAKDD'08), Takashi Washio, Akihiro Inokuchi, Einoshin Suzuki, and
   Kai Ming Ting (Eds.). Springer-Verlag, Berlin, Heidelberg, 148-159.
.. _datawig: https://github.com/datawig