.. _quickstart:

Quickstart
==========

Eager to get started? This page gives a good introduction to Morpheus. It
assumes you already have Morpheus installed. If you do not, head over to the
:ref:`installation` section.


A Minimal Example
-----------------

A minimal example for Morpheus looks something like this::

    from morpheus import Detector

    detector = Detector('pat1.csv', 'path2.csv')
    detector.run()

So what did that code do?