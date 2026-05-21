"""
C/H File Parsers
Extracts array definitions and constants from C source and header files
"""

import re
import sys

def extract_arrays_from_c(c_file):
    """
    Extract array definitions from C source file
    
    Supports formats:
      - double array_name[SIZE] = {val1, val2, ...};
      - float array_name[SIZE] = {val1, val2, ...};
      - c_float_store array_name[SIZE] = {val1, val2, ...};
      - typedef_type array_name[SIZE] = {val1, val2, ...};
    
    Args:
        c_file: Path to C source file
        
    Returns:
        dict: {array_name: {'size': int, 'values': [float, ...]}}
    """
    with open(c_file, 'r') as f:
        content = f.read()
    
    arrays = {}
    
    # Pattern: type_name array_name[SIZE] = {values};
    # Supports: double, float, c_float, c_float_store, or any identifier
    pattern = r'(\w+)\s+(\w+)\s*\[\s*(\d+)\s*\]\s*=\s*\{([^}]+)\}'
    
    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
        type_name = match.group(1)
        array_name = match.group(2)
        size = int(match.group(3))
        values_str = match.group(4)
        
        # Skip if it's clearly an integer array (c_int, int, unsigned, etc.)
        if 'int' in type_name.lower():
            continue
        
        # Parse values - handle type casts like (c_float_store)0.5
        values = []
        
        # Remove type casts: (c_float_store)value -> value
        cleaned_str = re.sub(r'\([^)]+\)', '', values_str)
        
        for val_str in cleaned_str.split(','):
            val_str = val_str.strip()
            if val_str:
                try:
                    # Handle scientific notation and regular floats
                    values.append(float(val_str))
                except ValueError:
                    print(f"WARNING: Cannot parse value '{val_str}' in array '{array_name}'")
                    continue
        
        # Validate size
        if len(values) != size:
            print(f"WARNING: Array '{array_name}' size mismatch: "
                  f"declared={size}, actual={len(values)}")
        
        arrays[array_name] = {
            'size': size,
            'values': values,
            'type': type_name
        }
    
    return arrays

def extract_constants_from_header(h_file):
    """
    Extract #define constants from header file
    
    Supports format: #define CONSTANT_NAME VALUE
    
    Args:
        h_file: Path to header file
        
    Returns:
        dict: {constant_name: int_value}
    """
    with open(h_file, 'r') as f:
        content = f.read()
    
    constants = {}
    
    # Pattern: #define NAME VALUE (where VALUE is a number)
    pattern = r'#define\s+([A-Z_][A-Z0-9_]*)\s+(\d+)'
    
    for match in re.finditer(pattern, content, re.MULTILINE):
        name = match.group(1)
        value = int(match.group(2))
        constants[name] = value
    
    return constants

def extract_integer_arrays_from_c(c_file):
    """
    Extract integer array definitions from C source file
    
    Useful for index arrays like pdaqp_hp_list, pdaqp_jump_list
    
    Args:
        c_file: Path to C source file
        
    Returns:
        dict: {array_name: {'size': int, 'values': [int, ...]}}
    """
    with open(c_file, 'r') as f:
        content = f.read()
    
    arrays = {}
    
    # Pattern for integer arrays
    pattern = r'(\w*int\w*)\s+(\w+)\s*\[\s*(\d+)\s*\]\s*=\s*\{([^}]+)\}'
    
    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
        type_name = match.group(1)
        array_name = match.group(2)
        size = int(match.group(3))
        values_str = match.group(4)
        
        # Parse integer values - handle type casts like (c_int)1
        values = []
        
        # Remove type casts: (c_int)value -> value
        cleaned_str = re.sub(r'\([^)]+\)', '', values_str)
        
        for val_str in cleaned_str.split(','):
            val_str = val_str.strip()
            if val_str:
                try:
                    values.append(int(val_str))
                except ValueError:
                    print(f"WARNING: Cannot parse integer value '{val_str}' in array '{array_name}'")
                    continue
        
        # Validate size
        if len(values) != size:
            print(f"WARNING: Array '{array_name}' size mismatch: "
                  f"declared={size}, actual={len(values)}")
        
        arrays[array_name] = {
            'size': size,
            'values': values,
            'type': type_name
        }
    
    return arrays

def validate_benchmark_structure(arrays, constants, required_constants=None):
    """
    Validate that benchmark has required structure
    
    Args:
        arrays: Extracted arrays dictionary
        constants: Extracted constants dictionary
        required_constants: List of required constant names
        
    Returns:
        bool: True if valid, raises exception otherwise
    """
    if required_constants is None:
        required_constants = ['N', 'M']  # Common requirements
    
    # Check required constants
    for const in required_constants:
        if const not in constants:
            raise ValueError(f"Required constant '{const}' not found in header")
    
    # Check arrays not empty
    if not arrays:
        raise ValueError("No arrays found in C source file")
    
    return True