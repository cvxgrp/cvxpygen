"""
PDAQP Interface Generator Package
Auto-generates RTL top module with pipelined AXI4-Stream interface

Modules:
    config_parser:     Parse pdaqp_config.vh
    packing_strategy:  Optimize port packing
    top_generator:     Generate pdaqp_top.v
    timing_generator:  Generate pdaqp_timing.vh placeholder
"""

__version__ = '1.0.0'
__author__ = 'Jingmin'

from .config_parser import VerilogConfigParser
from .packing_strategy import PackingStrategy
from .top_generator import TopModuleGenerator
from .timing_generator import TimingHeaderGenerator

__all__ = [
    'VerilogConfigParser',
    'PackingStrategy',
    'TopModuleGenerator',
    'TimingHeaderGenerator',
]