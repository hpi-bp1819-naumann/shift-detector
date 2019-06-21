.. _sorensen_dice:

Sørensen Dice
=============

Description
-----------

This check investigates shifts between two textual data sets with the
help of ngrams. The result shows how similar texts in each dataset are.

The check can only operate on textual data.

Example
-------

This section shows you how to use :ref:`sorensen_dice` and interpret its result.

Code
++++

::

    from morpheus.Detector import Detector
    from morpheus.checks.SorensenDiceCheck import SorensenDiceCheck

    data_set_1 = 'examples/shoes_first.csv'
    data_set_2 = 'examples/shoes_second.csv'

    detector = Detector(
        data_set_1,
        data_set_2,
        delimiter=','
    )

    detector.run(SorensenDiceCheck(ngram_type='character', n=5))
    detector.evaluate()

The code works as follows:

1. First, you create a :class:`~morpheus.Detector.Detector` object to tell morpheus
   which data sets you want to compare
2. Then, you start the detector with
   :meth:`~morpheus.Detector.Detector.run` and the checks you want to run: in this case
   :class:`~morpheus.checks.SorensenDiceCheck.SorensenDiceCheck`.
3. Finally, you print the result with
   :meth:`~morpheus.Detector.Detector.evaluate`

Result
++++++

:ref:`sorensen_dice` produces the following output:

::

    Sorensen Dice Check
    Examined Columns: ['item_name', 'description']
    Shifted Columns: ['item_name', 'description']
    Column 'item_name':
        Baseline in Dataset1: 0.882
        Baseline in Dataset2: 0.501
        Sorensen Dice Coefficient between Datasets: 0.545
    Column 'description':
        Baseline in Dataset1: 0.9355557653895485
        Baseline in Dataset2: 0.5669534250708042
        Sorensen Dice Coefficient between Datasets: 0.623536146479158

Interpretation
++++++++++++++

The above report can be read as follows:

1. The examined columns are 'description' and 'item_name"
2. The columns that contain a shift according to the Sørensen Dice Check are 'item_name' and 'description'
3. The similarity of the item names within Dataset1 is at 0.882
4. The similarity of the item names within Dataset2 is at 0.501
5. The similarity of Dataset1 and Dataset2 in the column 'item_name' is at 0.545


Parameters
----------

:ref:`sorensen_dice` provides the following parameters:

``ngram_type``:
    This parameter expects a string that describes on which basis the ngrams are generated. 
    Possible values are ``character``, which produces a character ngram, and ``word``, which produces a word ngram.

``n``:
    This parameter expects an integer, that determines the number of characters/words a tuple in the ngram is containing.

Implementation
--------------

Algorithm
+++++++++

:ref:`sorensen_dice` works as follows:

1. For each text an ngram is generated
2. For both datasets all ngrams of a columns are combined
3. All values in the resulting ngrams are devided by the total number of ngrams in the respective dataset
4. The Sørensen Dice Coefficient between the two ngrams is calculated