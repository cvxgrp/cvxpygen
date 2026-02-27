#!/usr/bin/env python3
"""
PDAQP Configuration Generator
Generates Verilog configuration files from C/H benchmark sources

Supports three data formats:
  - FP32:  32-bit IEEE 754 floating point
  - FP16:  16-bit IEEE 754 floating point  
  - FIX16: 16-bit fixed point with auto Q format detection

Usage:
  python3 generate_config.py -c benchmark.c -H benchmark.h -o output/
  python3 generate_config.py -c benchmark.c -H benchmark.h -o output/ --fp32
  python3 generate_config.py -c benchmark.c -H benchmark.h -o output/ --fp16
"""

import argparse
import sys
import os

# Support both direct execution and module import
try:
    from .parsers import (extract_arrays_from_c, extract_constants_from_header,
                         extract_integer_arrays_from_c)
    from .converters import DataConverter
    from .writers import VerilogWriter, MemoryWriter, IntegerMemoryWriter
except ImportError:
    from parsers import (extract_arrays_from_c, extract_constants_from_header,
                        extract_integer_arrays_from_c)
    from converters import DataConverter
    from writers import VerilogWriter, MemoryWriter, IntegerMemoryWriter

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate PDAQP configuration from C benchmark'
    )
    
    parser.add_argument('-c', '--c-file', required=True,
                       help='Input C source file containing arrays')
    parser.add_argument('-H', '--header-file', required=True,
                       help='Input header file containing constants')
    parser.add_argument('-o', '--output-dir', required=True,
                       help='Output directory for generated files')
    
    # Data format selection (mutually exclusive)
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument('--fp32', action='store_true',
                             help='Use 32-bit floating point format')
    format_group.add_argument('--fp16', action='store_true',
                             help='Use 16-bit floating point format')
    # Default is FIX16 if neither flag specified
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

def determine_data_format(args):
    """Determine data format from arguments"""
    if args.fp32:
        return 'fp32'
    elif args.fp16:
        return 'fp16'
    else:
        return 'fix16'  # Default

def validate_inputs(args):
    """Validate input files exist"""
    if not os.path.isfile(args.c_file):
        print(f"ERROR: C file not found: {args.c_file}")
        sys.exit(1)
    
    if not os.path.isfile(args.header_file):
        print(f"ERROR: Header file not found: {args.header_file}")
        sys.exit(1)

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Validate inputs
    validate_inputs(args)
    
    # Determine data format
    data_format = determine_data_format(args)
    
    print("=" * 70)
    print("PDAQP Configuration Generator")
    print("=" * 70)
    print(f"Input C file:      {args.c_file}")
    print(f"Input header file: {args.header_file}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Data format:       {data_format.upper()}")
    print("=" * 70)
    print()
    
    # Step 1: Parse input files
    print("[1/5] Parsing input files...")
    try:
        float_arrays = extract_arrays_from_c(args.c_file)
        int_arrays = extract_integer_arrays_from_c(args.c_file)
        constants = extract_constants_from_header(args.header_file)
    except Exception as e:
        print(f"ERROR: Failed to parse input files: {e}")
        sys.exit(1)
    
    print(f"  Found {len(float_arrays)} floating-point arrays")
    print(f"  Found {len(int_arrays)} integer arrays")
    print(f"  Found {len(constants)} constants")
    
    if args.verbose or True:  # Always show for debugging
        print(f"  Floating-point arrays: {list(float_arrays.keys())}")
        for name, data in float_arrays.items():
            print(f"    - {name}: {data['size']} elements (type: {data.get('type', 'unknown')})")
        
        print(f"  Integer arrays: {list(int_arrays.keys())}")
        for name, data in int_arrays.items():
            print(f"    - {name}: {data['size']} elements (type: {data.get('type', 'unknown')})")
        
        print(f"  Constants: {list(constants.keys())}")
        for name, value in constants.items():
            print(f"    - {name} = {value}")
    print()
    
    # Check if no arrays found
    if not float_arrays and not int_arrays:
        print("WARNING: No arrays found in C file!")
        print()
    
    # Step 2: Convert floating-point data to target format
    print(f"[2/5] Converting floating-point data to {data_format.upper()} format...")
    converter = DataConverter(data_format, verbose=args.verbose)
    
    converted_float_arrays = {}
    for name, array_data in float_arrays.items():
        converted = converter.convert_array(
            array_data['values'],
            array_name=name
        )
        converted_float_arrays[name] = {
            'size': array_data['size'],
            'values': array_data['values'],
            'hex_values': converted['hex_values'],
            'metadata': converted['metadata']
        }
    
    print()
    
    # Step 3: Create output directory
    create_output_directory(args.output_dir)
    
    # Step 4: Generate Verilog configuration file
    print("[3/5] Generating Verilog configuration file...")
    verilog_writer = VerilogWriter(data_format)
    vh_file = os.path.join(args.output_dir, 'pdaqp_config.vh')
    verilog_writer.write_config_file(
        vh_file,
        constants,
        converted_float_arrays,
        int_arrays
    )
    print(f"  Created: {vh_file}")
    print()
    
    # Step 5: Generate floating-point memory initialization files
    print("[4/5] Generating floating-point memory files...")
    memory_writer = MemoryWriter(data_format)
    for name, array_data in converted_float_arrays.items():
        mem_file = os.path.join(args.output_dir, f'{name}.mem')
        memory_writer.write_memory_file(
            mem_file,
            array_data['hex_values'],
            array_name=name
        )
        print(f"  Created: {mem_file}")
    print()
    
    # Step 6: Generate integer memory initialization files
    print("[5/5] Generating integer memory files...")
    int_memory_writer = IntegerMemoryWriter()
    for name, array_data in int_arrays.items():
        mem_file = os.path.join(args.output_dir, f'{name}.mem')
        int_memory_writer.write_memory_file(
            mem_file,
            array_data['values'],
            array_name=name
        )
        print(f"  Created: {mem_file}")
    
    print()
    print("=" * 70)
    print("✓ Configuration generation completed successfully!")
    print("=" * 70)
    
    # Summary
    print()
    print("Generated files:")
    print(f"  • {vh_file}")
    for name in converted_float_arrays.keys():
        print(f"  • {os.path.join(args.output_dir, name + '.mem')}")
    for name in int_arrays.keys():
        print(f"  • {os.path.join(args.output_dir, name + '.mem')}")
    print()

if __name__ == '__main__':
    main()