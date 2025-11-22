"""
Fluid Quantum Logic
===================

A quantum computing framework where logic operations are inherent to circuit
geometry rather than learned parameters, enabling zero-shot reprogrammability
via ancilla qubit control.

Modules
-------
topology : Defines domain-agnostic graph structures (Vision, Audio, Language).
binding  : Implements geometric quantum primitives for information integration.

Patent Pending - USPTO Serial No. 63/921,961
"""

__version__ = "1.0.0"
__author__ = "Larsen James Close"

from . import topology
from . import binding

__all__ = ['topology', 'binding']
