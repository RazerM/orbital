****************
orbital.maneuver
****************

.. automodule:: orbital.maneuver
   :members: Maneuver

Operations
==========

.. note::

   The operations below are used to construct the user-friendly maneuvers, as created using the :py:class:`~orbital.maneuver.Maneuver` class methods.

   To construct maneuvers manually using operations, one must be careful. A :py:class:`~orbital.maneuver.SetApocenterRadiusTo` operation can only be applied if the orbital position is at perigee, therefore it should be preceded by a :py:class:`~orbital.maneuver.PropagateAnomalyTo` operation, with argument :code:`f=0`.

.. automodule:: orbital.maneuver
   :members:
   :no-index:
   :exclude-members: Maneuver
