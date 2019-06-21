.. _word_prediction:

WordPredictionCheck
====================

Description
-----------

This check uses Machine Learning to investigate shifts in textual columns between two datasets.
It combines a ML-model, which learns word embeddings, and a custom ML-model to predict next words.

These models are able to learn text structures by trying to predict next words looking at some fixed number of last words.

The output consists of two floats, which represent the model's losses in the first and the second dataset.

Example
-------

This section provides a complete and **self-contained** example of
how to use this check.

Code
++++

::

    from shift_detector.detector import Detector
    from shift_detector.checks.word_prediction_check import WordPredictionCheck

    data_set_1 = 'data/pokedex1.csv'
    data_set_2 = 'data/pokedex2.csv'

    detector = Detector(
        data_set_1,
        data_set_2
    )

    detector.run(WordPredictionCheck(lstm_window=5, relative_thresh=.15))
    detector.evaluate()

The code works as follows:

1. First, you create a :class:`~shift_detector.Detector.Detector` object to tell Morpheus
   which data sets you want to compare.
2. Then, you start the detector with
   :meth:`~shift_detector.Detector.Detector.run` and the checks you want to run: in this case
   :class:`~shift_detector.checks.WordPredictionCheck.WordPredictionCheck`.
3. Finally, you print the result with
   :meth:`~shift_detector.Detector.Detector.evaluate`

Result
++++++

:ref:`word_prediction` produces the following report::

    WordPredictionCheck
    Examined Columns: ['Name', 'Entry', 'Type 1', 'Type 2']
    Shifted Columns: []

    Column 'Entry':
    0.11589096725815919 -> 0.12034393218721265
    Column 'Name':
    Cannot execute Check. Text column does not contain any row with num words > lstm_window(5)
    Column 'Type 1':
    Cannot execute Check. Text column does not contain any row with num words > lstm_window(5)
    Column 'Type 2':
    Cannot execute Check. Text column does not contain any row with num words > lstm_window(5)


Interpretation
++++++++++++++

The above report can be read as follows:

1. The examined columns are 'Name', 'Entry', 'Type 1' and 'Type 2'.
2. The column that contains a shift according to the word prediction check is 'Entry'.
3. The loss increased from ~0.116 on test data from the first dataset to ~0.12 on the second dataset.
   This is a increase of the loss of ~3.8%, which does not exceed the check's threshold.
4. Because the columns 'Name', 'Type 1' and 'Type 2' just contain a single word in each row, the check could not execute on that ones.
   It needs a number of words larger than lstm_window size.

This tells you that:

There is no significant shift in the textual column 'Entry'.


Parameters
----------

:ref:`word_prediction` provides the following parameter in order to adjust
    the run time and the quality of the result:

``columns=None``:
    This parameter expects a list of strings. These strings are the name of
    the columns that you want to be inspected. If no columns are provided all textual columns
    of the data sets are examined.

``ft_window_size=5``:
    This parameter lets you configure the window size of the FastText model.
    It defines the surrounding number of words the FastText model looks at to train each word embedding.

``ft_size=100``
    This parameter lets you configure the vector size of the FastText model.

``ft_workers=4``
    This parameter lets you configure the number threads the FastText model trains itself with.
    The higher ``ft_workers`` the faster the training process.

``seed=None``
    To reach reproducibility please set this parameter to a fixed seed.
    Moreover, you need to set the Python environment variable ``PYTHONHASHSEED``.
    This can be done using the following code snippet:
::

    import os
    os.environ['PYTHONHASHSEED'] = "0"

``lstm_window=5``
    This parameter lets you define the number of word vectors the custom ML model looks at to predict the next word vector.
    ``lstm_window=5`` means that the model looks at 5 word vectors to predict the 6th word vector.
    That's why a num words per row > ``lstm_window`` is needed.

``relative_thresh=.15``
    This is probably the most important parameter. It lets you define the check's threshold. If the relative difference between the losses exceed this threshold, the check will detect a shift in the examined column.


Implementation
--------------

.. _algorithm:

Algorithm
+++++++++

Firstly, a FastText model is trained. You can find parameters you might want to configure under section Parameters.
The output of this first step is a list of word vectors per row.

Then, a custom ML model using LSTM cells is used to predict the next word looking at a row of ``lstm_window`` words.

References
----------

.. [LSTM1997] Sepp Hochreiter, Jürgen Schmidhuber Long short-term memory In: Neural Computation (journal), vol. 9, issue 8, S. 1735–1780, 1997
   https://www.bioinf.jku.at/publications/older/2604.pdf
