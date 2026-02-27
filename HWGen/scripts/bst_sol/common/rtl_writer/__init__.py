"""
rtl_writer/__init__.py
"""

from pathlib import Path
from .rtl_writer import RTLWriter
from .rtl_writer_fixed import FixedPointRTLWriter
from .rtl_writer_float import FloatingPointRTLWriter


def create_rtl_writer(output_file: Path, config) -> RTLWriter:
    """
    Factory function to create appropriate RTL writer
    
    Args:
        output_file: Output file path
        config: PDQAPConfig object
    
    Returns:
        Appropriate RTLWriter subclass
    """
    if config.data_format.startswith('fixed'):
        return FixedPointRTLWriter(
            output_file,
            data_width=config.data_width,
            frac_bits=config.output_frac_bits or 8
        )
    elif config.data_format == 'fp32':
        return FloatingPointRTLWriter(
            output_file,
            fp_mult_latency=config.fp_mult_latency or 3,
            fp_add_latency=config.fp_add_latency or 3
        )
    else:
        # Fallback to base writer
        return RTLWriter(output_file)


__all__ = [
    'RTLWriter',
    'FixedPointRTLWriter', 
    'FloatingPointRTLWriter',
    'create_rtl_writer'
]