"""
Data Format Converters
Converts double precision values to FP32/FP16/FIX16 formats
"""

import struct
import math
import sys

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: NumPy not available, using fallback FP16 conversion")

class DataConverter:
    """Handles conversion between different numeric formats"""
    
    def __init__(self, data_format, verbose=False):
        """
        Initialize converter
        
        Args:
            data_format: 'fp32', 'fp16', or 'fix16'
            verbose: Enable verbose output
        """
        self.data_format = data_format.lower()
        self.verbose = verbose
        
        if self.data_format not in ['fp32', 'fp16', 'fix16']:
            raise ValueError(f"Unsupported format: {data_format}")
    
    def convert_array(self, values, array_name="unknown"):
        """
        Convert array of doubles to target format
        
        Args:
            values: List of float/double values
            array_name: Name of array (for logging)
            
        Returns:
            dict: {
                'hex_values': [hex_string, ...],
                'metadata': {format-specific info}
            }
        """
        if self.data_format == 'fp32':
            return self._convert_fp32(values, array_name)
        elif self.data_format == 'fp16':
            return self._convert_fp16(values, array_name)
        else:  # fix16
            return self._convert_fix16(values, array_name)
    
    def _convert_fp32(self, values, array_name):
        """Convert to 32-bit IEEE 754 floating point"""
        hex_values = []
        
        for val in values:
            hex_val = float_to_fp32_hex(val)
            hex_values.append(f"{hex_val:08X}")
        
        if self.verbose:
            print(f"  {array_name}: FP32 conversion")
            print(f"    Range: [{min(values):.6f}, {max(values):.6f}]")
            print(f"    Sample: {values[0]:.6f} -> 0x{hex_values[0]}")
        
        return {
            'hex_values': hex_values,
            'metadata': {
                'format': 'fp32',
                'bits': 32,
                'range': [min(values), max(values)]
            }
        }
    
    def _convert_fp16(self, values, array_name):
        """Convert to 16-bit IEEE 754 floating point"""
        hex_values = []
        
        for val in values:
            hex_val = float_to_fp16_hex(val)
            hex_values.append(f"{hex_val:04X}")
        
        if self.verbose:
            print(f"  {array_name}: FP16 conversion")
            print(f"    Range: [{min(values):.6f}, {max(values):.6f}]")
            print(f"    Sample: {values[0]:.6f} -> 0x{hex_values[0]}")
        
        return {
            'hex_values': hex_values,
            'metadata': {
                'format': 'fp16',
                'bits': 16,
                'range': [min(values), max(values)]
            }
        }
    
    def _convert_fix16(self, values, array_name):
        """Convert to 16-bit fixed point with auto Q format"""
        # Calculate optimal Q format
        max_abs = max(abs(min(values)), abs(max(values)))
        
        if max_abs == 0:
            integer_bits = 1
        else:
            # +1 for sign bit
            integer_bits = math.ceil(math.log2(max_abs + 1e-10)) + 1
        
        fractional_bits = 16 - integer_bits
        
        if fractional_bits < 0:
            raise ValueError(
                f"Array '{array_name}': Values too large for 16-bit fixed point!\n"
                f"  Max value: {max_abs}\n"
                f"  Required integer bits: {integer_bits}"
            )
        
        q_format = f"Q{integer_bits}.{fractional_bits}"
        scale_factor = 2 ** fractional_bits
        
        if self.verbose:
            print(f"  {array_name}: FIX16 conversion")
            print(f"    Range: [{min(values):.6f}, {max(values):.6f}]")
            print(f"    Q Format: {q_format}")
            print(f"    Scale: {scale_factor}")
        
        # Convert values
        hex_values = []
        for val in values:
            fixed_val = int(round(val * scale_factor))
            
            # Handle overflow
            if fixed_val > 32767:
                fixed_val = 32767
            elif fixed_val < -32768:
                fixed_val = -32768
            
            # Convert to unsigned 16-bit for hex representation
            unsigned_val = fixed_val & 0xFFFF
            hex_values.append(f"{unsigned_val:04X}")
        
        if self.verbose:
            print(f"    Sample: {values[0]:.6f} -> {int(round(values[0]*scale_factor))} -> 0x{hex_values[0]}")
        
        return {
            'hex_values': hex_values,
            'metadata': {
                'format': 'fix16',
                'bits': 16,
                'q_format': q_format,
                'integer_bits': integer_bits,
                'fractional_bits': fractional_bits,
                'scale_factor': scale_factor,
                'range': [min(values), max(values)]
            }
        }

# ===================== Helper Functions =====================

def float_to_fp32_hex(value):
    """
    Convert float to 32-bit IEEE 754 hex value
    
    Args:
        value: Float number
        
    Returns:
        int: 32-bit unsigned integer representation
    """
    packed = struct.pack('>f', float(value))
    return struct.unpack('>I', packed)[0]

def float_to_fp16_hex(value):
    """
    Convert float to 16-bit IEEE 754 hex value
    
    Args:
        value: Float number
        
    Returns:
        int: 16-bit unsigned integer representation
    """
    if HAS_NUMPY:
        # Use NumPy for accurate FP16 conversion
        fp16_val = np.float16(value)
        return np.frombuffer(fp16_val.tobytes(), dtype=np.uint16)[0]
    else:
        # Fallback implementation (less accurate)
        return float_to_fp16_fallback(value)

def float_to_fp16_fallback(value):
    """
    Fallback FP16 conversion without NumPy
    
    Note: This is a simplified implementation and may lose precision
    """
    if value == 0:
        return 0x0000
    
    # Handle sign
    sign = 0x8000 if value < 0 else 0x0000
    abs_val = abs(value)
    
    # Handle special cases
    if abs_val >= 65504:  # FP16 max value
        return sign | 0x7BFF
    elif abs_val < 2**-24:  # Denormal threshold
        return sign | 0x0000
    
    # Compute exponent and mantissa
    exponent = math.floor(math.log2(abs_val))
    exponent_biased = exponent + 15
    
    if exponent_biased <= 0:
        return sign | 0x0000
    if exponent_biased >= 31:
        return sign | 0x7C00  # Infinity
    
    mantissa_float = abs_val / (2 ** exponent) - 1.0
    mantissa = int(round(mantissa_float * 1024)) & 0x3FF
    
    return sign | (exponent_biased << 10) | mantissa