"""
Timing Header Generator
Creates placeholder pdaqp_timing.vh with conservative estimates
Will be overwritten by BST generator with actual values
"""

import os
import math
from datetime import datetime


class TimingHeaderGenerator:
    """Generate pdaqp_timing.vh with placeholder or actual timing values"""
    
    def __init__(self, config):
        """
        Args:
            config: VerilogConfigParser instance
        """
        self.config = config
    
    def generate_placeholder(self, output_file):
        """
        Generate placeholder timing header with conservative estimate
        
        Args:
            output_file: Path to pdaqp_timing.vh
        """
        estimated_latency = self._estimate_latency()
        
        with open(output_file, 'w') as f:
            self._write_header(f, is_placeholder=True)
            self._write_latency_define(f, estimated_latency, is_placeholder=True)
            self._write_footer(f)
        
        return estimated_latency
    
    def generate_actual(self, output_file, actual_latency, tree_depth, 
                       stages_breakdown):
        """
        Generate timing header with actual measured values from BST
        (Future use - called by BST generator)
        
        Args:
            output_file: Path to pdaqp_timing.vh
            actual_latency: Measured pipeline latency
            tree_depth: BST tree depth
            stages_breakdown: Dict with stage-by-stage latency info
        """
        with open(output_file, 'w') as f:
            self._write_header(f, is_placeholder=False)
            self._write_latency_define(f, actual_latency, is_placeholder=False)
            self._write_breakdown(f, tree_depth, stages_breakdown)
            self._write_footer(f)
    
    def _estimate_latency(self):
        """
        Conservative latency estimation based on configuration
        Returns: Estimated pipeline depth (int)
        """
        # Estimate tree depth from parameter count
        n_params = self.config.n_parameters
        estimated_regions = 2 ** max(n_params - 1, 3)
        tree_depth = int(math.ceil(math.log2(estimated_regions)))
        
        # Latency per tree level based on data format
        if self.config.data_format == 'fp32':
            latency_per_level = 13  # FP32 comparator + MUX + registers
        elif self.config.data_format == 'fp16':
            latency_per_level = 7   # FP16 comparator + MUX + registers
        else:  # fixed-point
            latency_per_level = 2   # Combinational compare + register
        
        # Calculate total with safety margin
        tree_latency = tree_depth * latency_per_level
        buffer_latency = 2  # Input/output buffering
        safety_margin = 5   # Conservative headroom
        
        total = tree_latency + buffer_latency + safety_margin
        
        # Round up to next power of 2 for efficient counter
        return 2 ** math.ceil(math.log2(max(total, 16)))
    
    def _write_header(self, f, is_placeholder):
        """Write file header with generation info"""
        f.write("//=======================================================================\n")
        f.write("// PDAQP Timing Parameters\n")
        f.write("//=======================================================================\n")
        f.write(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"// Configuration: {self.config.n_parameters} params -> {self.config.n_solutions} sols\n")
        f.write(f"// Data format: {self.config.data_format}\n")
        
        if is_placeholder:
            f.write("//\n")
            f.write("// STATUS: PLACEHOLDER - Conservative estimate\n")
            f.write("// This file will be overwritten by BST LUT generator with actual values\n")
        else:
            f.write("//\n")
            f.write("// STATUS: ACTUAL - Measured from BST implementation\n")
        
        f.write("//=======================================================================\n\n")
    
    def _write_latency_define(self, f, latency, is_placeholder):
        """Write main latency macro definition"""
        if is_placeholder:
            f.write("// BST pipeline latency (estimated)\n")
            f.write("// Conservative value - will be updated after BST generation\n")
        else:
            f.write("// BST pipeline latency (measured from actual implementation)\n")
        
        f.write(f"`ifndef PDAQP_BST_LATENCY\n")
        f.write(f"`define PDAQP_BST_LATENCY {latency}\n")
        f.write(f"`endif\n\n")
    
    def _write_breakdown(self, f, tree_depth, stages):
        """
        Write detailed timing breakdown
        Only used when generating actual timing (not placeholder)
        """
        f.write("// Detailed timing breakdown\n")
        f.write(f"`define PDAQP_TREE_DEPTH      {tree_depth}\n")
        f.write(f"`define PDAQP_INPUT_STAGES    {stages.get('input', 0)}\n")
        f.write(f"`define PDAQP_COMPARE_STAGES  {stages.get('compare', 0)}\n")
        f.write(f"`define PDAQP_MUX_STAGES      {stages.get('mux', 0)}\n")
        f.write(f"`define PDAQP_OUTPUT_STAGES   {stages.get('output', 0)}\n\n")
    
    def _write_footer(self, f):
        """Write file footer"""
        f.write("// End of timing parameters\n")