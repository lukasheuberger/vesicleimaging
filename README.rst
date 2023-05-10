==========
vesicleimaging
==========


.. image:: https://img.shields.io/pypi/v/vesicleimaging.svg
        :target: https://pypi.python.org/pypi/vesicleimaging

.. image:: https://img.shields.io/travis/lukasheuberger/vesicleimaging.svg
        :target: https://travis-ci.com/lukasheuberger/vesicleimaging

.. image:: https://readthedocs.org/projects/vesicleimaging/badge/?version=latest
        :target: https://vesicleimaging.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



* Free software: Apache Software License 2.0
* Documentation: https://vesicleimaging.readthedocs.io.


Features
--------

vesicleimaging is a package to handle and analyze czi images of polymer vesicles.\
It is currently under construction...

To Do
-----

- jupyter file templates
- make file selection and handling easier
- adapt algorithm detect inner and outer diameter
- fix metadata load to incorporate all metadata (part is commented in imgfileutils.py)
- pre-filtering for better houghes circles outcome
- hough circles should compare output of generated circles and improve params so that they are uniform (in case of
  uniform sample)
- manual interaction to deselect single circles
- make log that is put out with the most important parameters to analyze data
- make individual filenames for output (e.g. XXXX_analysis)
- lightweight czi to png converter with scalebar
- make napari work again with big sur openGL
- add environment.yaml to create a working virtualenv
- calculate from scaling and image size the expected GUV size and use this to detect GUVs instead of manual input
- improve statistics (mean, average, maybe boxplot, normalizing, ...)
- use multithreading / multiprocessing
- speed up image import, e.g. with AICSImageIO
- choose which slices to actually analyze (a slider would be nice)
- 3d plot of fluo data, maybe normalize first
- set all params first and analyze then later according to these params (dist from boarder, param1,2, etc.)
- make consistent img vs image
- combine data, metadata and add metadata into one thing and then extract later to recude no. of inputs
- make possible to change cmap


Contact
-------

For questions or suggestions regarding the code, please use the
issue tracker. For any other inquiries, please contact me
by email: lukas.heuberger@gmail.com



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

