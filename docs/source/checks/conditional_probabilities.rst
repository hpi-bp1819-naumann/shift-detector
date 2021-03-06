.. _conditional_probabilities:

Conditional Probabilities
=========================

Description
-----------

This check uses conditional probabilities to compare your data sets.
It produces a comprehensive and comprehensible overview of the
differences in the value distributions of your data sets.

As :ref:`conditional_probabilities` performs exact matching, it works best
with data sets consisting of categorical columns only. A pre-processing
step is included to transform continuous to categorical columns.

The result not only shows you if a shift happened. It also shows you where
and to what extent your data sets differ from each other.

Example
-------

This section shows you how to use :ref:`conditional_probabilities` and interpret
its result.

Code
++++

::

    from shift_detector.detector import Detector
    from shift_detector.checks.conditional_probabilities_check import ConditionalProbabilitiesCheck
    from shift_detector.utils.column_management import ColumnType

    data_set_1 = 'data/pokedex1.csv'
    data_set_2 = 'data/pokedex2.csv'

    custom_column_types = {
        'Type 1': ColumnType.categorical,
        'Generation': ColumnType.categorical
    }
    detector = Detector(data_set_1, data_set_2, delimiter=',', **custom_column_types)
    detector.run(ConditionalProbabilitiesCheck())
    detector.evaluate()

The code works as follows:

1. First, you create a :class:`~shift_detector.detector.Detector` object to tell Morpheus
   which data sets you want to compare.
2. Then, you specify in :meth:`~shift_detector.detector.Detector.run`
   which check you want to run: in this case
   :class:`~shift_detector.checks.conditional_probabilities_check.ConditionalProbabilitiesCheck`.
3. Finally, you print the result with :meth:`~shift_detector.detector.Detector.evaluate`.

Result
++++++

.. image:: ../images/conditional_probabilities.png
  :width: 1200

Interpretation
++++++++++++++

The above result can be interpreted as follows:

1. First, a listing of all columns considered by the check is displayed.
2. For each data set, attribute value pairs that are unique to this data
   set are displayed together with their support (relative number of tuples
   that contain these values). Keep in mind, that it is still possible that
   those values also appear in the other data set. In this case, they don't
   exceed min_support and/or min_confidence.
3. Mutual rules that exceed min_delta_support and min_delta_confidence are
   displayed. Take a look at the reading aid for information on how to read them.
4. Mutual rules that only exceed min_delta_support or min_delta_confidence
   (exclusive or) are displayed.
5. A diagram is displayed that plots all mutual rules according to their
   delta_supports and delta_confidences values.

.. _conditional_probabilities_parameters:

Parameters
----------

:ref:`conditional_probabilities` provides several tuning knobs and adjustable
thresholds that control (a) the runtime,
(b) the size of the result and (c) the applied pre-processing:

``min_support``:
    This parameter expects a float between 0 and 1 and impacts both runtime
    and size of the result. :ref:`conditional_probabilities` only considers
    rules whose ``support_of_left_side`` and ``support`` exceed ``min_support``
    in at least one of the two data sets.

    The lower you choose ``min_support`` the more resources are required during
    computation both in terms of memory and CPU.
    The default value is 0.01. This means that :ref:`conditional_probabilities`
    only considers values which appear in at least 1% of your tuples.
    By adjusting this parameter you can adjust the granularity of the comparison
    of the two data sets.

``min_confidence``:
    This parameter expects a float between 0 and 1 and impacts the size of the
    result. :ref:`conditional_probabilities` only considers rules whose
    ``confidence`` exceeds ``min_confidence`` in at least one of the two data sets.

    The lower you choose ``min_confidence`` the more rules are considered.
    The default value is 0.15. This means that the conditional probability
    of a right side (consequence) given a left side (antecedent) has to be at least 15%.

``rule_limit``:
    This parameter expects an int and controls the maximum number of rules that are
    printed in each section of the report as a result of executing
    :ref:`conditional_probabilities`. The default value is 5.
    This parameter does not have any impact on the visualization.

``min_delta_supports``:
    This parameter expects a float between 0 and 1 and affects the granularity of the
    comparison of the two data sets. Only rules whose support values exhibit an absolute
    difference of more than ``min_delta_supports`` are considered to indicate a shift.
    A rule has to exceed ``min_delta_supports`` to be classified as an orange rule. If it
    also exceeds ``min_delta_confidences`` it is classified as a red rule.
    The default value is 0.05.

``min_delta_confidences``:
    This parameter expects a float between 0 and 1 and affects the granularity of the
    comparison of the two data sets. Only rules whose confidence values exhibit an absolute
    difference of more than ``min_delta_confidences`` are considered to indicate a shift.
    A rule has to exceed ``min_delta_confidences`` to be classified as an orange rule. If it
    also exceeds ``min_delta_supports`` it is classified as a red rule.
    The default value is 0.05.

``number_of_bins``:
    This parameter affects pre-processing of numerical columns.
    Numerical columns are binned into ``number_of_bins`` many bins. The default value is 50.
    This means that numerical columns are binned into 50 equal-width bins.

``number_of_topics``:
    This parameter affects pre-processing of textual columns.
    Textual columns are embedded into ``number_of_topics`` topics. The default value is 20.

Please keep in mind that a rule has to satisfy **all** of the requirements above
to appear in the result.

Implementation
--------------

Algorithm
+++++++++

:ref:`conditional_probabilities` proceeds in two phases:

Rule Computation
################

1. Both data sets are pre-processed: numerical columns are binned and textual columns are
   embedded.
2. Both data sets are transformed: each component of every tuple is replaced
   by an attribute-name, attribute-value pair. However, this transformation is
   applied on the fly; we never actually copy the data.
3. The FP-growth algorithm is used to generate *association rules* for both
   data sets. The parameters ``min_support`` and
   ``min_confidence`` are used as described in [Han2000]_ and
   [Agrawal1994]_. The only difference is that both parameters are relative and
   expect ``floats`` between 0 and 1, whereas [Han2000]_ and [Agrawal1994]_
   use an absolute value for ``min_support``.
4. Association rules exceeding ``min_support`` and ``min_confidence`` in both
   data sets can be compared directly. For each of those rule-pairs generate an
   intermediate result rule similar to the form of the red rules showed above.
5. If a rule exceeds ``min_support`` and ``min_confidence`` in
   one data set but not in the other, we don't know if this rule does not appear in
   the other data set at all or just does not exceed ``min_support`` and/or
   ``min_confidence``. We therefore scan both data sets one
   more time and count their appearances. This information at hand, we can
   generate the remaining intermediate result rules.

Rule Reduction
##############

6. Intermediate result rules are partitioned in those that only appear in
   the first data set, those that only appear in the second data set, and
   those that appear in both data sets and have a non-empty
   right side (called mutual rules).
7. Mutual rules are filtered for those exceeding ``min_delta_supports`` and
   ``min_delta_confidences`` and sorted in descending order according to the
   absolute difference of their confidence values and the maximum of their
   supports of left side values. Rules whose support value are 0 in one data
   set come last.
8. Rules that only appear in one data set are filtered for significant
   rules. A rule is significant if there exists no other rule whose set of
   attribute value pairs is a proper subset of the set of attribute value
   pairs of the significant rule. Significant rules are then sorted
   in descending order according to their support of left side and their
   support.

References
----------

.. [Han2000] Jiawei Han, Jian Pei, and Yiwen Yin. 2000. Mining frequent patterns
   without candidate generation. In Proceedings of the 2000 ACM SIGMOD international
   conference on Management of data (SIGMOD '00). ACM, New York, NY, USA, 1-12
.. [Agrawal1994] Rakesh Agrawal and Ramakrishnan Srikant. 1994. Fast Algorithms for
   Mining Association Rules in Large Databases. In Proceedings of the 20th
   International Conference on Very Large Data Bases (VLDB '94), Jorge B. Bocca,
   Matthias Jarke, and Carlo Zaniolo (Eds.). Morgan Kaufmann Publishers Inc., San
   Francisco, CA, USA, 487-499.