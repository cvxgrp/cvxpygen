"""
Output File Writers
Generates Verilog header files and memory initialization files
"""

import os
import math
from datetime import datetime

class VerilogWriter:
    """Writes Verilog header configuration files"""
    
    def __init__(self, data_format):
        """
        Initialize writer
        
        Args:
            data_format: 'fp32', 'fp16', or 'fix16'
        """
        self.data_format = data_format
    
    def write_config_file(self, output_file, constants, float_arrays, int_arrays=None):
        """
        Write Verilog header file with configuration
        
        Args:
            output_file: Output .vh file path
            constants: Dictionary of constants
            float_arrays: Dictionary of converted floating-point arrays
            int_arrays: Dictionary of integer arrays (optional)
        """
        if int_arrays is None:
            int_arrays = {}
            
        with open(output_file, 'w') as f:
            self._write_header(f)
            self._write_basic_parameters(f, constants, float_arrays, int_arrays)
            self._write_derived_parameters(f, constants)
            self._write_data_format_selection(f)
            self._write_format_specific_config(f, float_arrays)
            self._write_compatibility_defines(f)
            self._write_footer(f)
    
    def _write_header(self, f):
        """Write file header comment"""
        f.write("// Auto-generated PDAQP configuration\n")
        f.write("// Source: pdaqp.h, pdaqp.c\n")
        
        if self.data_format == 'fp32':
            f.write("// Mode: FP32 (IEEE 754 Single Precision)\n")
        elif self.data_format == 'fp16':
            f.write("// Mode: FP16 (IEEE 754 Half Precision)\n")
        else:
            f.write("// Mode: Fixed-Point\n")
        
        f.write("\n")
    
    def _write_basic_parameters(self, f, constants, float_arrays, int_arrays):
        """Write basic benchmark parameters"""
        # Extract key constants
        n_parameter = constants.get('PDAQP_N_PARAMETER', 0)
        n_solution = constants.get('PDAQP_N_SOLUTION', 0)
        
        # Calculate derived values from arrays
        tree_nodes = len(int_arrays.get('pdaqp_hp_list', {}).get('values', []))
        halfplanes_total = len(float_arrays.get('pdaqp_halfplanes', {}).get('values', []))
        feedbacks_total = len(float_arrays.get('pdaqp_feedbacks', {}).get('values', []))
        
        f.write(f"`define PDAQP_N_PARAMETER {n_parameter}\n")
        f.write(f"`define PDAQP_N_SOLUTION {n_solution}\n")
        f.write(f"`define PDAQP_TREE_NODES {tree_nodes}\n")
        f.write(f"`define PDAQP_HALFPLANES {halfplanes_total}\n")
        f.write(f"`define PDAQP_FEEDBACKS {feedbacks_total}\n")
        f.write("\n")
    
    def _write_derived_parameters(self, f, constants):
        """Write derived parameters"""
        n_parameter = constants.get('PDAQP_N_PARAMETER', 0)
        n_solution = constants.get('PDAQP_N_SOLUTION', 0)
        
        # Calculate derived values
        sol_per_node = n_solution * (n_parameter + 1)
        halfplane_stride = n_parameter + 1
        
        # Estimate BST depth (log2 of typical tree size)
        estimated_depth = 4  # Conservative estimate
        
        f.write(f"`define PDAQP_SOL_PER_NODE {sol_per_node}\n")
        f.write(f"`define PDAQP_HALFPLANE_STRIDE {halfplane_stride}\n")
        f.write(f"`define PDAQP_ESTIMATED_BST_DEPTH {estimated_depth}\n")
        f.write("\n")
    
    def _write_data_format_selection(self, f):
        """Write data format selection flags"""
        f.write("// Data format selection\n")
        
        use_fp32 = 1 if self.data_format == 'fp32' else 0
        use_fp16 = 1 if self.data_format == 'fp16' else 0
        use_fix16 = 1 if self.data_format == 'fix16' else 0
        
        f.write(f"`define PDAQP_USE_FP32 {use_fp32}\n")
        f.write(f"`define PDAQP_USE_FP16 {use_fp16}\n")
        f.write(f"`define PDAQP_USE_FIX16 {use_fix16}\n")
        
        if self.data_format == 'fp32':
            f.write("`define INPUT_DATA_WIDTH 32\n")
            f.write("`define OUTPUT_DATA_WIDTH 32\n")
        else:  # fp16 or fix16
            f.write("`define INPUT_DATA_WIDTH 16\n")
            f.write("`define OUTPUT_DATA_WIDTH 16\n")
        
        f.write("\n")
    
    def _write_format_specific_config(self, f, float_arrays):
        """Write format-specific configuration"""
        if self.data_format == 'fp32':
            self._write_fp32_config(f)
        elif self.data_format == 'fp16':
            self._write_fp16_config(f)
        else:  # fix16
            self._write_fix16_config(f, float_arrays)
    
    def _write_fp32_config(self, f):
        """Write FP32-specific configuration"""
        f.write("// FP32 mode - IEEE 754 single precision\n")
        f.write("`define PDAQP_FP32_MODE 1\n")
        f.write("\n")
    
    def _write_fp16_config(self, f):
        """Write FP16-specific configuration"""
        f.write("// FP16 mode - IEEE 754 half precision (16-bit)\n")
        f.write("`define PDAQP_FP16_MODE 1\n")
        f.write("\n")
        f.write("// FP16 format specifications\n")
        f.write("// Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits\n")
        f.write("// Range: ±6.55e4, Precision: ~3.3 decimal digits\n")
        f.write("\n")
    
    def _write_fix16_config(self, f, float_arrays):
        """Write FIX16-specific configuration"""
        f.write("// Fixed-point mode (16-bit)\n")
        f.write("`define PDAQP_FIX16_MODE 1\n")
        f.write("\n")
        
        # Get Q formats from arrays
        halfplane_metadata = None
        feedback_metadata = None
        
        if 'pdaqp_halfplanes' in float_arrays:
            halfplane_metadata = float_arrays['pdaqp_halfplanes']['metadata']
        
        if 'pdaqp_feedbacks' in float_arrays:
            feedback_metadata = float_arrays['pdaqp_feedbacks']['metadata']
        
        # Write halfplane Q format
        if halfplane_metadata:
            f.write("// Fixed-point Q format for halfplanes\n")
            f.write(f"`define HALFPLANE_INT_BITS {halfplane_metadata['integer_bits']}\n")
            f.write(f"`define HALFPLANE_FRAC_BITS {halfplane_metadata['fractional_bits']}\n")
            f.write(f"`define HALFPLANE_SCALE_FACTOR {halfplane_metadata['scale_factor']}\n")
        else:
            f.write("// Fixed-point Q format for halfplanes (default)\n")
            f.write("`define HALFPLANE_INT_BITS 3\n")
            f.write("`define HALFPLANE_FRAC_BITS 13\n")
            f.write("`define HALFPLANE_SCALE_FACTOR 8192\n")
        
        f.write("\n")
        
        # Write feedback Q format
        if feedback_metadata:
            f.write("// Fixed-point Q format for feedbacks\n")
            f.write(f"`define FEEDBACK_INT_BITS {feedback_metadata['integer_bits']}\n")
            f.write(f"`define FEEDBACK_FRAC_BITS {feedback_metadata['fractional_bits']}\n")
            f.write(f"`define FEEDBACK_SCALE_FACTOR {feedback_metadata['scale_factor']}\n")
        else:
            f.write("// Fixed-point Q format for feedbacks (default)\n")
            f.write("`define FEEDBACK_INT_BITS 2\n")
            f.write("`define FEEDBACK_FRAC_BITS 14\n")
            f.write("`define FEEDBACK_SCALE_FACTOR 16384\n")
        
        f.write("\n")
        
        # Backward compatibility - use feedback format as default
        if feedback_metadata:
            f.write("// Backward compatibility\n")
            f.write(f"`define PDAQP_FIXED_POINT_BITS {feedback_metadata['fractional_bits']}\n")
            f.write(f"`define PDAQP_SCALE_FACTOR {feedback_metadata['scale_factor']}\n")
            f.write(f"`define INPUT_INT_BITS {feedback_metadata['integer_bits']}\n")
            f.write(f"`define OUTPUT_INT_BITS {feedback_metadata['integer_bits']}\n")
            f.write(f"`define OUTPUT_FRAC_BITS {feedback_metadata['fractional_bits']}\n")
        else:
            f.write("// Backward compatibility (default)\n")
            f.write("`define PDAQP_FIXED_POINT_BITS 14\n")
            f.write("`define PDAQP_SCALE_FACTOR 16384\n")
            f.write("`define INPUT_INT_BITS 2\n")
            f.write("`define OUTPUT_INT_BITS 2\n")
            f.write("`define OUTPUT_FRAC_BITS 14\n")
        
        f.write("\n")
    
    def _write_compatibility_defines(self, f):
        """Write compatibility defines for non-active formats"""
        if self.data_format != 'fix16':
            f.write("// Compatibility defines (not used in FP mode)\n")
            f.write("`define HALFPLANE_INT_BITS 0\n")
            f.write("`define HALFPLANE_FRAC_BITS 0\n")
            f.write("`define HALFPLANE_SCALE_FACTOR 0\n")
            f.write("`define FEEDBACK_INT_BITS 0\n")
            f.write("`define FEEDBACK_FRAC_BITS 0\n")
            f.write("`define FEEDBACK_SCALE_FACTOR 0\n")
            f.write("`define PDAQP_FIXED_POINT_BITS 0\n")
            f.write("`define PDAQP_SCALE_FACTOR 0\n")
            f.write("`define INPUT_INT_BITS 0\n")
            f.write("`define OUTPUT_INT_BITS 0\n")
            f.write("`define OUTPUT_FRAC_BITS 0\n")
    
    def _write_footer(self, f):
        """Write file footer (none needed for this format)"""
        pass

class MemoryWriter:
    """Writes floating-point memory initialization files"""
    
    def __init__(self, data_format):
        """
        Initialize writer
        
        Args:
            data_format: 'fp32', 'fp16', or 'fix16'
        """
        self.data_format = data_format
    
    def write_memory_file(self, output_file, hex_values, array_name="unknown"):
        """
        Write memory initialization file for floating-point data
        
        Args:
            output_file: Output .mem file path
            hex_values: List of hex strings
            array_name: Name of array (for comments)
        """
        with open(output_file, 'w') as f:
            f.write(f"// Memory initialization file for: {array_name}\n")
            f.write(f"// Format: {self.data_format.upper()}\n")
            f.write(f"// Size: {len(hex_values)} elements\n")
            f.write(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("//\n")
            f.write("// Format: One hex value per line (no 0x prefix)\n")
            f.write("//============================================\n\n")
            
            for hex_val in hex_values:
                f.write(f"{hex_val}\n")

class IntegerMemoryWriter:
    """Writes integer memory initialization files"""
    
    def __init__(self):
        """Initialize integer memory writer"""
        pass
    
    def write_memory_file(self, output_file, int_values, array_name="unknown"):
        """
        Write memory initialization file for integer data
        
        Args:
            output_file: Output .mem file path
            int_values: List of integer values
            array_name: Name of array (for comments)
        """
        # Determine bit width needed
        max_val = max(int_values) if int_values else 0
        min_val = min(int_values) if int_values else 0
        
        # Calculate required bits
        if max_val > 0:
            bits_positive = max_val.bit_length()
        else:
            bits_positive = 0
            
        if min_val < 0:
            bits_negative = (abs(min_val) - 1).bit_length() + 1
        else:
            bits_negative = 0
        
        required_bits = max(bits_positive, bits_negative, 1)
        
        # Round up to common widths: 8, 16, 32
        if required_bits <= 8:
            bit_width = 8
        elif required_bits <= 16:
            bit_width = 16
        else:
            bit_width = 32
        
        hex_width = bit_width // 4
        
        with open(output_file, 'w') as f:
            f.write(f"// Memory initialization file for: {array_name}\n")
            f.write(f"// Format: Integer ({bit_width}-bit)\n")
            f.write(f"// Size: {len(int_values)} elements\n")
            f.write(f"// Range: [{min_val}, {max_val}]\n")
            f.write(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("//\n")
            f.write("// Format: One hex value per line (no 0x prefix)\n")
            f.write("//============================================\n\n")
            
            for val in int_values:
                # Convert to unsigned representation for hex
                if val < 0:
                    unsigned_val = (1 << bit_width) + val
                else:
                    unsigned_val = val
                
                f.write(f"{unsigned_val:0{hex_width}X}\n")