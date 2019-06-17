.. _simple:

Simple Check
====================

Description
-----------

The Simple Check gives you basic information about your datasets and detects the simplest of datashifts.

Some shifts result in the change of basic metrics, that are commonly used for the analysis and exploration of datasets.
Those metrics and statistics are calculated on both datasets and compared afterwards.

The basic check should be used mostof the times, if you want to analyze your datasets because its metrics are
scientificaly grounded and commmonly used. It works directly on categorical and numerical data and, with additional
Precalculations, also on text data.


Example
-------

::

    from shift_detector.detector import Detector
    from shift_detector.checks.simple_check import DistinctionCheck

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

1. First, you create a :class:`~shift_detector.detector.Detector` object to tell Morpheus
   which data sets you want to compare.
2. Then, you start the detector with
   :meth:`~shift_detector.detector.Detector.run` and the checks you want to run: in this case
   :class:`~shift_detector.checks.simple_check.SimpleCheck`.
3. Finally, you print the result with
   :meth:`~shift_detector.detector.Detector.evaluate`

Result
++++++

:ref:`simple` produces the following report::

    Column 'payment':
    Metric: mean with Diff: -0.29 %
    Metric: std with Diff: +0.8 %

Interpretation
++++++++++++++

The checkresults show column-wise wich shifts appear in your dataset. In this case the two metrics 'mean' so the
mean of the column 'payment' and 'std', the standard-deviation of the dataset seem to differ between the
datasets.



Parameters
----------

This section is optional. Use it to discuss parameters of your check. Guide
the user to pick reasonable values for those parameters.

Implementation
--------------

This section is intended for interested readers. A user should be able to
use your check without having to read this section.

Algorithm
+++++++++

Use this subsection to describe how your check processes its input to produce
its output. If applicable you can link to scientific publications.

Notes
+++++

Use this subsection to include notes about your implementation:
design decisions, why you favored one algorithm over another, ...

References
----------

Reference section.