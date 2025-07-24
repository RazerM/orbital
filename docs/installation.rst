************
Installation
************

Orbital is available from PyPI_:

.. code:: bash

  pip install orbitalpy

.. _PyPI: https://pypi.org/project/parver

or to install from git:

.. code:: bash

    pip install 'OrbitalPy @ git+https://github.com/RazerM/orbital.git@master'


Development Environment
=======================

You can work on the repository by following these steps:

#. `Install uv <https://docs.astral.sh/uv/getting-started/installation/>`_

#. Create and activate your virtual environment

   .. code-block:: bash

       uv sync
       source .venv/bin/activate

#. Run tests

   .. code-block:: bash

       nox -s tests
       nox -s tests-3.13

#. Build documentation

   .. code-block:: bash

       nox -s docs
