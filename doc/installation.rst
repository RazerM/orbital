************
Installation
************

The Orbital package can be installed on any system running **Python 2.7 or 3**.

The recommended installation method is using ``pip``.

pip
===

.. code:: bash

    $ pip install orbitalpy

Git
===

.. code-block:: sh

	$ git clone https://github.com/RazerM/orbital.git
	Cloning into 'orbital'...

Check out a `release tag <https://github.com/RazerM/orbital/releases>`_:

.. parsed-literal::

	$ cd orbital
	$ git checkout |version|

Test and run in a venv:

.. code-block:: sh

	$ python -m venv venv
	$ venv/bin/pip install numpy scipy astropy matplotlib represent sgp4 pytest
	# Without installing (just runs the code from the source dir):
	# Run tests:
	$ PYTHONPATH=. venv/bin/pytest
	# Run a script that uses the library:
	$ PYTHONPATH=. venv/bin/python <script>

	# Installing into the venv:
	$ venv/bin/pip install setuptools
	$ venv/bin/python setup.py install
	# Run tests (must run install every time code changes):
	$ venv/bin/pytest
	# Run a script that uses the library:
	$ venv/bin/python <script>

Install to system:

.. code-block:: sh

	$ python setup.py install
	running install...
