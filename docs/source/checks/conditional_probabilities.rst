.. _conditional_probabilities

Conditional probabilities
=========================

Abstract
--------

This check compares two data sets using conditional probabilities.
Specifically, it compares the probability of co-occurring
values inside a tuple. Its output highlights where those
probabilities diverge.

Motivation
----------

.. _example:

We introduce this check by an example:

    Assume you have two data sets (``ds1`` and ``ds2``) about shoes extracted from a
    product catalog. Each tuple contains information about *make*, *color* and
    *category*.

    This check returns a list of rules obeying the following form::

        MAKE: Nike, COLOR: black ==> CATEGORY: football
        [SUPPORTS_OF_LEFT_SIDES: (0.3, 0.07), DELTA_SUPPORTS_OF_LEFT_SIDES: 0.23,
        SUPPORTS: (0.03, 0.05), DELTA_SUPPORTS: -0.02, CONFIDENCES: (0.1, 0.71),
        DELTA_CONFIDENCES: -0.61]

    This rule states that (a) 30% of the tuples in ``ds1`` and 7% of the tuples in
    ``ds2`` are about black Nike shoes, which accounts to a difference of 23%,
    (b) 3% of the tuples in ``ds1`` and 5% of the tuples in ``ds2`` are about black
    Nike football shoes, which accounts to a difference of -2%, and
    (c) if a tuple is about black Nike shoes the **conditional probability** that
    the category is football is 10% in ``ds1`` and 71% in ``ds2``, which
    accounts to a difference of -61%.

    On the basis of such rules, you can easily get insights into differences
    between the two data sets. The above rule alone tells you, that (a) ``ds1``
    contains way more tuples about black Nike shoes than ``ds2``, however, (b) the
    probability that such a shoe is made for football is way higher in ``ds2`` than
    in ``ds1``.

Example
-------

The basic example of this check is::

    from shift_detector.Detector import Detector
    from shift_detector.checks.FrequentItemRulesCheck import FrequentItemsetCheck

    detector = Detector('/path1.csv', '/path2.csv', delimiter=',')
    detector.add_checks(
        FrequentItemsetCheck(min_support=0.01, min_confidence=0.15)
    )

    detector.run()
    detector.evaluate()

1. You have to create a :class:`~shift_detector.Detector.Detector` object in order to tell Morpheus which data sets you want to compare.
2. Then you have to specify in :meth:`~shift_detector.Detector.Detector.add_checks`, which checks you want to run: in this case only :class:`~shift_detector.checks.FrequentItemRulesCheck.FrequentItemsetCheck`.
3. Finally you have to start the detector with :meth:`~shift_detector.Detector.Detector.run` and print the results with :meth:`~shift_detector.Detector.Detector.evaluate`.

Specification
-------------

The check proceeds with the following steps:

1. Both data sets are transformed: each component of every tuple is replaced by an
   attribute-name, attribute-value pair. This is required for the correct
   working of the next step. However, the transformation is applied on the fly; we
   never actually copy the data.
2. Our implementation of FP-growth is used to generate *association rules* for both
   data sets individually. The parameters ``min_support`` and
   ``min_confidence`` are used as described in [Han2000]_ and
   [Agrawal1994]_. The only difference is that both parameters are relative and
   expect ``floats`` between 0 and 1, whereas [Han2000]_ and [Agrawal1994]_ use an
   absolute value for ``min_support``:

   ``min_support``:
     This parameter mainly impacts the runtime of FP-growth. The lower
     ``min_support`` the more resources are required during computation
     both in terms of memory and CPU. The default value is ``0.01``, which is high
     enough to get a reasonably good performance and still low enough to not
     prematurely exclude significant association rules. This parameter allows you to
     adjust the granularity of the comparison of the two data sets.

   ``min_confidence``:
     This parameter impacts the amount of generated association rules. The higher
     ``min_confidence`` the more rules are generated. The default value is
     ``0.15``. There is no further significance in this value other than that it
     seems sufficiently reasonable.

   Only association rules whose support exceeds ``min_support`` and whose
   confidence exceeds ``min_confidence`` in at least one data set are
   included in the generated association rules.
3. All association rules exceeding ``min_support`` and
   ``min_confidence`` in both data sets can be compared directly. For each
   such rule generate one association rule of the form showed in the example_ above.
4. If a rule exceeds ``min_support`` and ``min_confidence`` in
   one data set but not in the other, we don't know if this rule does not appear in
   the other data set at all or just does not exceed ``min_support`` and/or
   ``min_confidence``. We therefore have to scan both data sets one
   last time to aggregate the counts of such rules. This information at hand, we can
   generate the remaining association rules and our algorithm terminates.


Implementation Notes
--------------------

We use the FP-growth algorithm as proposed in [Han2000]_ for association rule mining.
The code is largely copied from fp-growth_.
The function ``generate_association_rules(...)`` is revised in the following ways:

1. A parameter called ``size`` is added to the *parameter list*.
   It expects the total number of transactions used to construct the *FP-tree* and
   is needed to compute relative support values.
2. The return value is changed to a *dictionary* of the form
   ``{(left_side, right_side): (support_of_left_side, support, confidence)}``.
   ``support_of_left_side`` and ``support`` give the
   percentage of tuples containing all attribute-value pairs from ``left_side`` and
   ``right_side`` combined.

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
   indicator that (a) both results are false but in exactly the same way or (b) both
   results are correct. We opt for (b).

   * During this comparison we could confirm that FP-growth is both faster and
     requires less memory than the Apriori algorithm as is also shown in [Han2000]_.
     This is why we feel confident, that FP-growth is the better algorithm for our
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