.. _dq_metrics:

DQ Metrics
==========

Description
-----------

The :ref:`dq_metrics` gives you basic information about your datasets and detects the most obvious of datashifts. The
'dq' stands for data quality, the metrics are in the style of the 'Unit Test for data'-framework Amazon Deequ.
For more information on Deequ see [SLSCB18]_.

Some shifts result in the change of basic metrics, that are commonly used for the analysis and exploration of datasets.
Those metrics and statistics are calculated on both datasets and compared afterwards.

The check should be used in most analyses, its metrics are commonly used to describe datasets.
It works directly on categorical and numerical data and, with additional Precalculations, also on text data.


Example
-------

::

    from Morpheus.detector import Detector
    from Morpheus.checks.dq_metrics_check import DQMetricsCheck

    data_set_1 = 'examples/pokedex1.csv'
    data_set_2 = 'examples/pokedex2.csv'

    detector = Detector(
        data_set_1,
        data_set_2,
        delimiter=','
    )

    detector.run(DQMetricsCheck())
    detector.evaluate()

The code works as follows:

1. First, you create a :class:`~Morpheus.detector.Detector` object to tell Morpheus
   which data sets you want to compare.
2. Then you start the detector with
   :meth:`~Morpheus.detector.Detector.run` and the checks you want to run: in this case
   :class:`~Morpheus.checks.dq_metrics_check.DQMetricsCheck`.
3. Finally, you print the result with
   :meth:`~Morpheus.detector.Detector.evaluate`

Result
++++++

The :ref:`dq_metrics` produces this report:

::

    Numerical Columns:

    Column 'payment':
    Metric: mean with Diff: -29.4 %, threshold: 20%
    Metric: std with Diff: +80.2 %, threshold: 50%

    (...)


    Categorical Columns

    Column 'payment_option':
    Attribute: 'debit' with Diff: +6.0 %, categorical threshold: 5%
    Attribute: 'cash' with Diff: -6.5 %, categorical threshold: 5%

    (...)



Interpretation
++++++++++++++

The results show column-wise wich shifts appear in your dataset. First the reports of your numerical columns are shown,
after that the reports of your categorical columns.

Numerical
~~~~~~~~~

In the example, below the heading 'numerical columns', you see that the column with the name 'payment' is detected
as shifted. Below the columnname, you can see that two numerical metrics detected a shift, *mean* and *std*
(standard deviation).

*Diff* describes the relative value-distance of the metrics that are calculated on your two datasets. If the Diff
exceeds a metric-specific threshold, the dataset differ that much, that the check calls shift on this column.

In this case, the mean-value of 'payment' could be 1000.0 in your first, but 706.0 in your second dataset. This difference is
results in an difference of -294 and an relative difference of -29.4%. The threshold set here is 20% which is exceeded
by the absolute value of -29.4%. All thresholds are customizable through the API of the
:class:`~Morpheus.checks.dq_metrics_check.DQMetricsCheck`. All the metrics are listed in the section
:ref:`dq_metrics_check_parameters`.

Categorical
~~~~~~~~~~~

Using the :ref:`dq_metrics` shifts can also be detected on categorical columns. Here the histograms over the attribute-values are
compared. If the difference between those values exceeds the *categorical_threshold* the check calls shift on this
column.

The example above shows a shift in the categorical column 'payment_option'. There are different
attribute-values in this column, i.e. 'debit and 'cash'. In those, the differences between the dataset are 6% and
-6.5% which both exceed the threshold of 5%. This indicates that in dataset2 more people use
cash as a payment-option and fewer use debit, the check calls shift on the column.


.. _dq_metrics_check_parameters:

Metrics & Parameters
--------------------

There are 9 different numerical metrics in the :ref:`dq_metrics` whose differences can indicate a shift. All
default-thresholds can be adjusted.

+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| metric_name                       |                                                                           explanation                                                   |
+===================================+=========================================================================================================================================+
| **quartile_1,                     |                                                                                                                                         |
| median,                           | Those are the .25- .5- and .75-quantiles of the column.                                                                                 |
| quartile_3**                      |                                                                                                                                         |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **value-range**                   | The range of values of the columns, calculated as difference between maximum and minimum of that column.                                |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **std**                           | the standard deviation in the column                                                                                                    |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **mean**                          | The means or averages of a column.                                                                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **uniqueness**                    | The ratio of values that are unique to the total number of values. A value is unique if it appears only one time in the whole dataset.  |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **completeness**                  | The ratio of non None-values to the total number of values.                                                                             |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **num_distinct**                  | The total number of distinct values in a column.                                                                                        |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+

All parameters expect float values larger than 0.0. The smallest value of 0.0 means that only the smallest of changes
(bigger than 0%) is going to result in the metric to call shift. For most cases a threshold smaller then 1.0, meaning
100% difference is recommended.

The 9 threshold parameters for numerical metrics are accessable through their name *'metric_name'_threshold*,
the threshold parameter for the categorical columns is called *categorical_threshold*.

Example
::

    from Morpheus.checks.dq_metrics_check import DQMetricsCheck
    sc = new DQMetricsCheck(median_threshold=.05, std_threshold=.42, categorical_threshold=1.05)




Implementation
--------------

Algorithm
+++++++++

The :ref:`DQMetricsCheck` works as follows:

1.  First, calculate the metrics for all usable columns of the datasets ds1 and ds2. Most metrics are build upon
    functions from the python library *pandas*
2.  Then, take the difference between each metric, so *diff_metric = metric(ds1) - metric(ds2)*
3.  Finally, compare those diffs to the predifined or custom thresholds. If the threshold is exceeded, indicate a shift

Notes
+++++

The 0.0- and 1.0-quantiles, so the minima and maxima, are not part of the shift-metrics because they have proven to be
very unresistant to outliers.

References
----------

.. [SLSCB18] Sebastian Schelter, Dustin Lange, Philipp Schmidt, Meltem Celikel, Felix Biessmann, and Andreas Grafberger.
   2018. Automating large-scale data quality verification. Proc. VLDB Endow. 11, 12 (August 2018), 1781-1794.