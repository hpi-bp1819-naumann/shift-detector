.. _quickstart:

Quickstart
==========

Eager to get started? This page gives a good introduction to Icarus. It
assumes you already have Icarus installed. If you do not, head over to the
:ref:`installation` section.


A Minimal Application
---------------------

A minimal Icarus application looks something like this::

    from icarus import Detector

    detector = Detector('pat1.csv', 'path2.csv')
    detector.run()

So what did that code do?