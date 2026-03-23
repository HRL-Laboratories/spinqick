
Developers Guide
================

While spinQICK is MIT licensed, allowing for unfettered modification and distribution, we do invite those who would like to improve spinQICK to contribute! In addition to standard issues,
this may also include the creation of new submodules, and the addition of tools and features shared across existing modules. This guide will explain the steps and expectations to make contributions
to spinQICK. These guidelines are based off of the existing best-practices imposed by HRL-Laboratories LLC, and are themselves a combination of established best practices for code contribution.
Before any contribution, there are a few key checks one should make to ensure that spinQICK and any contribution made maintains its open-source status.


==========
Disclaimer
==========

It is the responsibility of the contributor to ensure that contributed code does not contain confidential information, intellectual property, or materials that have been deemed dual-use or are protected under trade restrictions imposed by their country of origin. HRL-Laboratories LLC is not responsible for this determination. It is also understood by the contributor that any code released is provided under the same license as the spinQICK package. No contributor may modify this license. This license may be found `here <https://github.com/HRL-Laboratories/spinqick/blob/main/LICENSE>`__.

==========================
spinQICK Development Scope
==========================

spinQICK makes heavy use of the `QICK <https://docs.qick.dev/latest/index.html>`__ API and firmware. The stable fork of QICK API and firmware supported by spinQICK is present `here <https://github.com/HRL-Laboratories/qick>`__. The spinQICK developers guide does not cover the contents of this repository and any modification to these repositories must be done through issue creation and feature requests with QICK. Issues can be made `here <https://github.com/openquantumhardware/qick/issues>`__. If there is a desire to create a custom firmware based on the spinQICK supported firmware, Vivado project files can be found `here <https://s3df.slac.stanford.edu/people/meeg/qick/tprocv2/2025-08-01_216_tprocv2r25_rfbv2_16fullspeed_11xtalk/>`__



=======================
Development Environment
=======================

The spinQICK development environment is located in dev_environment.yml. This includes the necessary packages for pre-commit checks based on HRL best practices. To create this environment run:

.. code-block:: bash

    conda env create -f dev_environment.yml

And then activate the ``spinqick_dev`` environment.

======================
Local editable install
======================

It is advised that spinQICK be installed as an editable package. This can be done by first navigating to the location of the setup.py and pyproject.toml files in the main repository directory and running the following PIP command:

.. code-block:: bash

    pip install -e .


======================
git pre-commit checks
======================

SpinQICK is setup to automatically provide pre-commit checks using the ``pre-commit`` and ``pre-commit-hooks`` packages. The checking is outlined in the ``.pre-commit-config.yaml`` file. It is not a bad idea to periodically check for problems and clean up code. To do this without a commit simply run the following in the top-level directory where the pre-commit config file is located.

.. code-block:: bash

    pre-commit run --all-files
