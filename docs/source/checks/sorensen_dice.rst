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

    from morpheus.detector import Detector
    from morpheus.checks.sorensen_dice_check import SorensenDiceCheck

    data_set_1 = 'data/pokedex1.csv'
    data_set_2 = 'data/pokedex2.csv'

    detector = Detector(
        data_set_1,
        data_set_2,
        delimiter=','
    )

    detector.run(SorensenDiceCheck(ngram_type='character', n=5))
    detector.evaluate()

The code works as follows:

1. First, you create a :class:`~Morpheus.Detector.Detector` object to tell Morpheus
   which data sets you want to compare
2. Then, you start the detector with
   :meth:`~Morpheus.Detector.Detector.run` and the checks you want to run: in this case
   :class:`~Morpheus.checks.SorensenDiceCheck.SorensenDiceCheck`.
3. Finally, you print the result with
   :meth:`~Morpheus.Detector.Detector.evaluate`

Result
++++++

:ref:`sorensen_dice` produces the following output:

::

    Sorensen Dice Check
    Examined Columns: ['Type 2', 'Type 1', 'Name', 'Entry']
    Shifted Columns: ['Type 1', 'Name', 'Type 2']

            Baseline in Dataset 1  Baseline in Dataset 2  Distance between Datasets
    Type 1               0.778443               0.510345                   0.839186
    Entry                0.260510               0.209295                   0.506504
    Name                 0.007456               0.030494                   0.022639
    Type 2               0.689764               0.561594                   0.701052

Interpretation
++++++++++++++

The above report can be read as follows:

1. The examined columns are ``Type 2``, ``Type 1``, ``Name`` and ``Entry``
2. The columns that contain a shift according to the Sørensen Dice Check are ``Type 2``, ``Type 1`` and ``Name``
3. The similarity of the column ``Type 1`` within Dataset1 is at 0.778
4. The similarity of the column ``Type 1`` within Dataset2 is at 0.510
5. The similarity of Dataset1 and Dataset2 in the column ``Type 1`` is at 0.839
6. Since the similarity within the Datasets differs strong enough, there is probably a shift in column ``Type 1``


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