"""
PDAQP Configuration Generator Package

This package provides tools to generate Verilog configuration files
from C/H benchmark sources with support for multiple data formats.

Modules:
    parsers:    C/H file parsing utilities
    converters: Data format conversion (FP32/FP16/FIX16)
    writers:    Verilog and memory file generation
"""

__version__ = '1.0.0'
__author__ = 'Jingmin'

# Import main components for easier access
from .parsers import (
    extract_arrays_from_c,
    extract_constants_from_header,
    validate_benchmark_structure
)

from .converters import (
    DataConverter,
    float_to_fp32_hex,
    float_to_fp16_hex
)

from .writers import (
    VerilogWriter,
    MemoryWriter
)

__all__ = [
    # Parsers
    'extract_arrays_from_c',
    'extract_constants_from_header',
    'validate_benchmark_structure',
    
    # Converters
    'DataConverter',
    'float_to_fp32_hex',
    'float_to_fp16_hex',
    
    # Writers
    'VerilogWriter',
    'MemoryWriter',
]