# encoding: utf-8
from __future__ import absolute_import, division, print_function

from .bodies import *
from .constants import *
from .elements import *
from .maneuver import *
from .plotting import *
from .utilities import *

# Don't add constants to __all__, but they can be imported from here
__all__ = (bodies.__all__ + elements.__all__ + maneuver.__all__ +
           plotting.__all__ + utilities.__all__)

__author__ = 'Frazer McLean <frazer@frazermclean.co.uk>'
__version__ = '0.6.2'
__license__ = 'MIT'
__description__ = 'High level orbital mechanics package.'
