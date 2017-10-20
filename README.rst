=========
pydftools
=========


.. image:: https://img.shields.io/pypi/v/pydftools.svg
        :target: https://pypi.python.org/pypi/pydftools

.. image:: https://img.shields.io/travis/steven-murray/pydftools.svg
        :target: https://travis-ci.org/steven-murray/pydftools

.. image:: https://readthedocs.org/projects/pydftools/badge/?version=latest
        :target: https://pydftools.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/steven-murray/pydftools/shield.svg
     :target: https://pyup.io/repos/github/steven-murray/pydftools/
     :alt: Updates


A pure-python port of the ``dftools`` R package.

This package attempts to imitate the ``dftools`` package (repo: https://github.com/obreschkow/dftools ) quite closely,
while being as Pythonic as possible. Do note that 2D+ models are not yet implemented in this Python port, and neither
are non-parametric models. Hopefully they will be along soon.

From ``dftool``'s description:

    Description: This package can find the most likely P parameters of a D-dimensional distribution function (DF) generating
    N objects, where each object is specified by D observables with measurement uncertainties. For instance, if the objects
    are galaxies, it can fit a MF (P=1), a mass-size distribution (P=2) or the mass-spin-morphology distribution (P=3).
    Unlike most common fitting approaches, this method accurately accounts for measurement is uncertainties and complex
    selection functions. A full description of the algorithm can be found in Obreschkow et al. (2017).

In short, clean out Eddington bias from your fits:

.. image:: https://user-images.githubusercontent.com/1272030/31757852-60cb6ebc-b4dd-11e7-8ce9-32b3232e8f94.png
   :height: 100px
   :width: 200 px
   :scale: 50 %

* Free software: MIT license
* Documentation: https://pydftools.readthedocs.io.


Features
--------

* Simple and fast parameter fitting for generative distribution functions
* Several examples (with astronomical applications in mind)
* Several plotting routines so that you can go from nothing to a plot in minutes
* A ``mockdata()`` function which can produce data to fit.
* Support for arbitrary 1D models, several kinds of selection functions, jackknife and bootstrap resampling, Gaussian
  error estimation and more.

Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

