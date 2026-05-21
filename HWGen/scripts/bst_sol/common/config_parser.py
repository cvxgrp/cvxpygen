"""
Verilog Configuration Parser
Extracts problem dimensions and data format from .vh files
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PDQAPConfig:
    """PDAQP problem configuration"""
    # Problem dimensions
    n_parameters: int
    n_solutions: int
    n_tree_nodes: int
    n_halfplanes: int  # Added: total halfplane entries
    n_feedbacks: int   # Added: total feedback entries
    estimated_bst_depth: int
    
    # Memory addressing
    halfplane_stride: int  # Added: elements per halfplane (n_param + 1)
    sol_per_node: int      # Added: solutions per leaf node
    
    # Data format
    data_format: str  # 'fixed16', 'fixed32', 'fp16', 'fp32'
    data_width: int
    
    # Fixed-point Q formats
    halfplane_frac_bits: Optional[int] = None
    halfplane_int_bits: Optional[int] = None
    halfplane_scale_factor: Optional[int] = None  # Added: 2^frac_bits
    
    feedback_frac_bits: Optional[int] = None
    feedback_int_bits: Optional[int] = None
    feedback_scale_factor: Optional[int] = None   # Added: 2^frac_bits
    
    output_frac_bits: Optional[int] = None
    output_int_bits: Optional[int] = None
    
    # Floating-point specific
    fp_mult_latency: Optional[int] = None
    fp_add_latency: Optional[int] = None
    fp_metadata_width: Optional[int] = None
    
    # Paths
    project_name: str = "pdaqp"
    config_file: Optional[Path] = None
    
    def __str__(self):
        return (f"PDAQP Config: {self.n_parameters} params -> {self.n_solutions} solutions "
                f"({self.data_format}, {self.n_tree_nodes} nodes)")


class VerilogConfigParser:
    """Parse Verilog configuration files"""
    
    def __init__(self, config_file: str):
        self.config_file = Path(config_file)
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        self.defines: Dict[str, str] = {}
        self._parse_defines()
        self.config = self._build_config()
    
    def _parse_defines(self):
        """Extract all define statements from config file"""
        with open(self.config_file, 'r') as f:
            content = f.read()
        
        # Match `define NAME VALUE
        pattern = r'`define\s+(\w+)\s+(.+?)(?:\n|$)'
        matches = re.findall(pattern, content, re.MULTILINE)
        
        for name, value in matches:
            # Strip comments and whitespace
            value = value.strip()
            value = re.sub(r'//.*$', '', value).strip()
            self.defines[name] = value
    
    def _get_int(self, name: str, default: int = 0) -> int:
        """Get define value as integer"""
        value = self.defines.get(name, str(default))
        
        if isinstance(value, str):
            # Remove inline comments
            value = value.split('//')[0].strip()
            try:
                return int(value)
            except ValueError:
                # Handle hex values
                if value.startswith(('0x', '0X')):
                    return int(value, 16)
        return default
    
    def _detect_data_format(self) -> tuple[str, int]:
        """Detect data format and width from config defines"""
        
        # Priority 1: Check explicit fixed-point mode flags
        if self._get_int('PDAQP_USE_FIX16', 0) == 1 or self._get_int('PDAQP_FIX16_MODE', 0) == 1:
            return 'fixed16', 16
        
        # Priority 2: Check FP flags
        if self._get_int('PDAQP_USE_FP32', 0) == 1 or self._get_int('PDAQP_FP32_MODE', 0) == 1:
            return 'fp32', 32
        
        if self._get_int('PDAQP_USE_FP16', 0) == 1:
            return 'fp16', 16
        
        # Priority 3: Check FP_DATA_WIDTH from fp_config.vh
        fp_width = self._get_int('FP_DATA_WIDTH', 0)
        if fp_width > 0:
            format_map = {16: 'fp16', 32: 'fp32', 64: 'fp64'}
            return format_map.get(fp_width, 'fp32'), fp_width
        
        # Priority 4: Check explicit DATA_FORMAT define
        if 'DATA_FORMAT' in self.defines:
            fmt = self.defines['DATA_FORMAT'].strip('"\'')
            if fmt in ['fixed16', 'fixed32', 'fp16', 'fp32']:
                width = 16 if fmt in ['fixed16', 'fp16'] else 32
                return fmt, width
        
        # Default: fixed16
        return 'fixed16', 16
    
    def _build_config(self) -> PDQAPConfig:
        """Build configuration object from parsed defines"""
        fmt, width = self._detect_data_format()
        
        config = PDQAPConfig(
            # Basic dimensions
            n_parameters=self._get_int('PDAQP_N_PARAMETER', 1),
            n_solutions=self._get_int('PDAQP_N_SOLUTION', 1),
            n_tree_nodes=self._get_int('PDAQP_TREE_NODES', 1),
            n_halfplanes=self._get_int('PDAQP_HALFPLANES', 1),
            n_feedbacks=self._get_int('PDAQP_FEEDBACKS', 1),
            estimated_bst_depth=self._get_int('PDAQP_ESTIMATED_BST_DEPTH', 4),
            
            # Memory addressing
            halfplane_stride=self._get_int('PDAQP_HALFPLANE_STRIDE', 3),
            sol_per_node=self._get_int('PDAQP_SOL_PER_NODE', 9),
            
            # Data format
            data_format=fmt,
            data_width=width,
            
            # Metadata
            project_name=self.config_file.stem.replace('_config', ''),
            config_file=self.config_file
        )
        
        # Extract fixed-point parameters if applicable
        if fmt.startswith('fixed'):
            config.halfplane_frac_bits = self._get_int('HALFPLANE_FRAC_BITS', 0)
            config.halfplane_int_bits = self._get_int('HALFPLANE_INT_BITS', 0)
            config.halfplane_scale_factor = self._get_int('HALFPLANE_SCALE_FACTOR', 0)
            
            config.feedback_frac_bits = self._get_int('FEEDBACK_FRAC_BITS', 0)
            config.feedback_int_bits = self._get_int('FEEDBACK_INT_BITS', 0)
            config.feedback_scale_factor = self._get_int('FEEDBACK_SCALE_FACTOR', 0)
            
            config.output_frac_bits = self._get_int('OUTPUT_FRAC_BITS', 0)
            config.output_int_bits = self._get_int('OUTPUT_INT_BITS', 0)
        
        # Extract floating-point parameters if applicable
        elif fmt.startswith('fp'):
            config.fp_mult_latency = self._get_int('FP_MULT_LATENCY', 9)
            config.fp_add_latency = self._get_int('FP_ADD_LATENCY', 12)
            config.fp_metadata_width = self._get_int('FP_METADATA_WIDTH', 32)
        
        return config
    
    def get_config(self) -> PDQAPConfig:
        """Get parsed configuration"""
        return self.config