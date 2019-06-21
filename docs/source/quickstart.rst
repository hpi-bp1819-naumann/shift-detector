.. _quickstart:

Quickstart
==========

Eager to get started? This page gives a good introduction to morpheus. It
assumes you already have morpheus installed. If you do not, head over to the
:ref:`installation` section.


A Minimal Example
-----------------

A minimal example for morpheus looks something like this::

    from morpheus import Detector

    detector = Detector('path1.csv', 'path2.csv')
    detector.run()

So what did that code do?