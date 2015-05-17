**************
orbital.bodies
**************

.. automodule:: orbital.bodies
   :members:

The planets of the solar system are defined in this module.

For example, earth can be imported as follows:

.. code:: python

  from orbital.bodies import earth

The definition of specific apside names allows the following code:

.. code:: python

   >>> from orbital import KeplerianElements, uranus
 
   >>> orbit = KeplerianElements.with_altitude(6e8, body=uranus)
   >>> orbit.apouranion_radius
   625362000.0
   >>> orbit.apocenter_radius == orbit.apouranion_radius
   True

The full list of defined planets and their apside names is shown below:

+---------+----------------+-----------------+
|   Body  | Apoapsis Names | Periapsis Names |
+=========+================+=================+
| mercury | Aphermion      | Perihermion     |
+---------+----------------+-----------------+
| venus   | Apocytherion   | Pericytherion   |
|         +----------------+-----------------+
|         | Apocytherean   | Pericytherean   |
|         +----------------+-----------------+
|         | Apokrition     | Perikrition     |
+---------+----------------+-----------------+
| earth   | Apogee         | Perigee         |
+---------+----------------+-----------------+
| mars    | Apoareion      | Periareion      |
+---------+----------------+-----------------+
| jupiter | Apozene        | Perizene        |
|         +----------------+-----------------+
|         | Apojove        | Perijove        |
+---------+----------------+-----------------+
| saturn  | Apokrone       | Perikrone       |
|         +----------------+-----------------+
|         | Aposaturnium   | Perisaturnium   |
+---------+----------------+-----------------+
| uranus  | Apouranion     | Periuranion     |
+---------+----------------+-----------------+
| neptune | Apoposeidon    | Periposeidon    |
+---------+----------------+-----------------+

