"""
Verilog Configuration Parser
Extracts parameters from pdaqp_config.vh
"""

import os
import re


class VerilogConfigParser:
    """Parse Verilog header file for PDAQP configuration"""
    
    def __init__(self, vh_file):
        """
        Args:
            vh_file: Path to pdaqp_config.vh
        """
        self.vh_file = vh_file
        self.params = {}
        self._parse()
    
    def _parse(self):
        """Parse configuration file and extract parameters"""
        if not os.path.exists(self.vh_file):
            raise FileNotFoundError(f"Config file not found: {self.vh_file}")
        
        with open(self.vh_file, 'r') as f:
            content = f.read()
        
        # Extract `define parameters
        pattern = r'`define\s+(\w+)\s+(\d+)'
        for match in re.finditer(pattern, content):
            name = match.group(1)
            value = int(match.group(2))
            self.params[name] = value
        
        # Validate required parameters
        required = [
            'PDAQP_N_PARAMETER',
            'PDAQP_N_SOLUTION',
            'INPUT_DATA_WIDTH',
            'OUTPUT_DATA_WIDTH'
        ]
        
        missing = [p for p in required if p not in self.params]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        
        # Generate default names
        self._generate_default_names()
    
    def _generate_default_names(self):
        """Generate standard parameter and solution names"""
        # Standard naming: param_0, param_1, ...
        self._parameter_names = [f"param_{i}" for i in range(self.n_parameters)]
        
        # Standard naming: x_0, x_1, ...
        self._solution_names = [f"x_{i}" for i in range(self.n_solutions)]
    
    @property
    def n_parameters(self):
        """Number of input parameters"""
        return self.params['PDAQP_N_PARAMETER']
    
    @property
    def n_solutions(self):
        """Number of output solutions"""
        return self.params['PDAQP_N_SOLUTION']
    
    @property
    def input_width(self):
        """Input data width in bits"""
        return self.params['INPUT_DATA_WIDTH']
    
    @property
    def output_width(self):
        """Output data width in bits"""
        return self.params['OUTPUT_DATA_WIDTH']
    
    @property
    def data_format(self):
        """Data format type (fp32/fp16/fix16)"""
        if self.params.get('PDAQP_USE_FP32', 0):
            return 'fp32'
        elif self.params.get('PDAQP_USE_FP16', 0):
            return 'fp16'
        else:
            return 'fix16'
    
    @property
    def parameter_names(self):
        """List of parameter names (param_0, param_1, ...)"""
        return self._parameter_names
    
    @property
    def solution_names(self):
        """List of solution names (x_0, x_1, ...)"""
        return self._solution_names
    
    def __repr__(self):
        return (f"VerilogConfig(params={self.n_parameters}, "
                f"sols={self.n_solutions}, "
                f"format={self.data_format}, "
                f"width={self.input_width})")