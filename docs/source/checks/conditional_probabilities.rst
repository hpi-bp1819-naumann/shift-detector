.. _conditional_probabilities:

Conditional probabilities
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

    from shift_detector.Detector import Detector
    from shift_detector.checks.FrequentItemRulesCheck import FrequentItemsetCheck

    data_set_1 = 'examples/shoes_first.csv'
    data_set_2 = 'examples/shoes_second.csv'

    detector = Detector(
        data_set_1,
        data_set_2,
        delimiter=','
    )
    detector.add_checks(
        FrequentItemsetCheck()
    )

    detector.run()
    detector.evaluate()

The code works as follows:

1. First, you create a :class:`~shift_detector.Detector.Detector` object to tell Morpheus
   which data sets you want to compare.
2. Then, you specify in :meth:`~shift_detector.Detector.Detector.add_checks`
   which check you want to run: in this case
   :class:`~shift_detector.checks.FrequentItemRulesCheck.FrequentItemsetCheck`.
3. Finally, you start the detector with
   :meth:`~shift_detector.Detector.Detector.run` and print the result with
   :meth:`~shift_detector.Detector.Detector.evaluate`.

Result
++++++

:ref:`conditional_probabilities` produces a set of rules::

    MAKE: Nike, COLOR: black ==> CATEGORY: football
    [SUPPORTS_OF_LEFT_SIDES: (0.3, 0.07), DELTA_SUPPORTS_OF_LEFT_SIDES: 0.23,
    SUPPORTS: (0.03, 0.05), DELTA_SUPPORTS: -0.02, CONFIDENCES: (0.1, 0.71),
    DELTA_CONFIDENCES: -0.61]
    ...

Interpretation
++++++++++++++

The above rule can be read as follows:

1. 30% of the tuples in ``data_set_1`` and 7% of the tuples in ``data_set_2``
   are about black Nike shoes. This accounts to a difference of 23%.
2. 3% of the tuples in ``data_set_1`` and 5% of the tuples in ``data_set_2``
   are about black Nike football shoes. This accounts to a difference of -2%.
3. If a tuple is about black Nike shoes the **conditional probability** that
   the category is football is 10% in ``data_set_1`` and 71% in ``data_set_2``.
   This accounts to a difference of -61%.

This tells you that:

1. ``data_set_1`` contains way more tuples about black Nike shoes than
   ``data_set_2``, however,
2. the probability that a black Nike shoe is made for football is way higher
   in ``data_set_2`` than in ``data_set_1``.

Parameters
----------

:ref:`conditional_probabilities` provides two tuning knobs that you can use
to control (a) the computational complexity and (b) the size of the result:

``min_support``:
    This parameter expects a float between 0 and 1 and impacts both runtime
    and size of the result. :ref:`conditional_probabilities` only produces
    rules whose ``support_of_left_side`` and ``support`` exceed ``min_support``
    in at least one data set.

    The lower you choose ``min_support`` the more resources are required during
    computation both in terms of memory and CPU.
    The default value is ``0.01``. This means that :ref:`conditional_probabilities`
    only considers values which appear in at least 1% of your tuples.
    By adjusting this parameter you can adjust the granularity of the comparison
    of the two data sets.

``min_confidence``:
    This parameter expects a float between 0 and 1 and impacts the size of the
    result. :ref:`conditional_probabilities` only produces rules whose
    ``confidence`` exceeds ``min_confidence`` in at least one data set.

    The lower you choose ``min_confidence`` the more rules are generated.
    The default value is ``0.15``. This means that the **conditional probability**
    of a specific right side given a specific left side has to be at least 15%.

Implementation
--------------

Algorithm
+++++++++

:ref:`conditional_probabilities` works as follows:

1. Both data sets are transformed: each component of every tuple is replaced
   by an attribute-name, attribute-value pair. However, this transformation is
   applied on the fly; we never actually copy the data.
2. The FP-growth algorithm is used to generate *association rules* for both
   data sets. The parameters ``min_support`` and
   ``min_confidence`` are used as described in [Han2000]_ and
   [Agrawal1994]_. The only difference is that both parameters are relative and
   expect ``floats`` between 0 and 1, whereas [Han2000]_ and [Agrawal1994]_
   use an absolute value for ``min_support``.
3. Association rules exceeding ``min_support`` and ``min_confidence`` in both
   data sets can be compared directly. For each of those rule-pairs generate a
   result rule of the form showed above.
4. If a rule exceeds ``min_support`` and ``min_confidence`` in
   one data set but not in the other, we don't know if this rule does not appear in
   the other data set at all or just does not exceed ``min_support`` and/or
   ``min_confidence``. We therefore scan both data sets one
   more time and count their appearances. This information at hand, we can
   generate the remaining result rules.

Notes
+++++

We use the FP-growth algorithm as proposed in [Han2000]_ to compute all relevant
conditional probabilities. The code is largely copied from fp-growth_.
The function ``generate_association_rules(...)`` is revised in the following ways:

1. A parameter called ``size`` is added to the *parameter list*.
   It expects the total number of transactions used to construct the *FP-tree* and
   is needed to compute relative support values.
2. The return value is changed to a *dictionary* of the form
   ``{(left_side, right_side): (support_of_left_side, support, confidence)}``.
   ``support_of_left_side`` and ``support`` give the
   percentage of tuples containing all attribute-value pairs from ``left_side``
   alone and ``left_side`` and ``right_side`` combined.

   * This additionally fixes a `bug
     <https://github.com/evandempsey/fp-growth/issues/11>`_ present in fp-growth_:
     if several rules have the same left side, fp-growth_ erroneously overwrites
     those rules and returns only one rule. The revised function present in this
     module does not contain this bug anymore.
3. fp-growth_ does not `generate rules having an empty right side
   <https://github.com/evandempsey/fp-growth/issues/6>`_. Those should
   however be part of a correct result and are vital for our purposes. We therefore
   adapted the function to include those rules too.

We feel very confident that the code is correct and reasonably fast:

1. We included unit tests to verify that our implementation produces the correct
   result for an example taken from [Agrawal1994]_.
2. We compared the result produced by this implementation on a large production
   data set with the result produced by an implementation of the Apriori algorithm
   [Agrawal1994]_ we used previously. Both results were identical. This is a strong
   indicator that either both results are false but in exactly the same way or both
   results are correct. We think it's the latter.

   * During this comparison we could confirm that FP-growth is both faster and
     requires less memory than the Apriori algorithm as is also shown in [Han2000]_.
     This is why we feel confident that FP-growth is the right choice for our
     use case.

As a last aside: we issued a `Pull Request <https://github.com/evandempsey/fp-growth/pull/17>`_
for fp-growth_ containing our bug fixes.

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
.. _fp-growth: https://github.com/evandempsey/fp-growth