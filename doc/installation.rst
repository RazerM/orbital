************
Installation
************

The Orbital package can be installed on any system running **Python 3**.

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

Test and install:

.. code-block:: sh

	$ python setup.py test
	running test...
	$ python setup.py install
	running install...