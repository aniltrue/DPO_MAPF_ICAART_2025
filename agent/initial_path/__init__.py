"""
    **Initial Path Strategy**: Employs *centralized* algorithm to determine the initial other_path for each agent where only
    *static obstacles* exist and fully observable scenario.
"""

from .InitialPathStrategy import AbstractInitialPathStrategy
from .EECBS import EECBS
from .CBS import CBS
from .LPStrategy import LPStrategy
from .SingleACO import SingleAgentACO
