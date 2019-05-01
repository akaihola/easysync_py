====================
 EasySync in Python
====================

EasySync is the protocol powering the Etherpad and Etherpad Lite
collaborative on-line text editors.

This project aims to port the protocol into Python.
This is done gradually by first translating the original JavaScript library
into Python using Js2Py_ and then rewriting it piece by piece in Python.
The original EasySync test suite is kept green at every commit.

To understand how this is being done in detail,
it helps to go through commits from the very beginning of this repository


Requirements
============

- Python 3.7
- Js2Py


Quickstart
==========

Run::

    pip install Js2Py
    python easysync_tests.py


.. _Js2Py: https://github.com/PiotrDabkowski/Js2Py
