"""FreeTO-Python: 3D freeform topology optimisation with smooth boundaries.

Python port of FreeTO (O. Ibhadode et al., MATLAB, 2024).
"""
from .core import FreeTOConfig, FreeTOResult, run_freeto
from .errors import FreeTOError
from .examples import EXAMPLES, example_config
from .postprocess import FieldSnapshot, surface_from_field
from .stl_io import read_stl, write_stl

__version__ = "1.0.0"

__all__ = ["FreeTOConfig", "FreeTOResult", "run_freeto", "read_stl",
           "write_stl", "EXAMPLES", "example_config", "FieldSnapshot",
           "surface_from_field", "FreeTOError", "__version__"]
