#!/usr/bin/env python3
"""
PDAQP Memory File Generator with Exact BST Depth Calculation
Generates Verilog configuration and memory initialization files
"""

import re
import os
import argparse
import sys
import math
from collections import deque

def extract_arrays_from_c(c_file_path):
    """Extract arrays from C file with robust regex patterns"""
    try:
        with open(c_file_path, 'r') as f:
            c_content = f.read()
    except Exception as e:
        print(f"Error reading C file '{c_file_path}': {e}")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(c_file_path))[0]
    
    # Robust regex with escaped base_name and optional whitespace
    halfplanes_match = re.search(
        rf'{re.escape(base_name)}_halfplanes\s*\[\s*(\d+)\s*\]', c_content)
    feedbacks_match = re.search(
        rf'{re.escape(base_name)}_feedbacks\s*\[\s*(\d+)\s*\]', c_content)
    hp_list_match = re.search(
        rf'{re.escape(base_name)}_hp_list\s*\[\s*(\d+)\s*\]', c_content)
    jump_list_match = re.search(
        rf'{re.escape(base_name)}_jump_list\s*\[\s*(\d+)\s*\]', c_content)
    
    n_halfplanes = int(halfplanes_match.group(1)) if halfplanes_match else 0
    n_feedbacks = int(feedbacks_match.group(1)) if feedbacks_match else 0
    n_hp_list = int(hp_list_match.group(1)) if hp_list_match else 0
    n_jump_list = int(jump_list_match.group(1)) if jump_list_match else 0
    
    # Extract array contents with DOTALL for multiline
    halfplanes_content = re.search(
        rf'{re.escape(base_name)}_halfplanes\[\d+\]\s*=\s*\{{([^}}]+)\}};',
        c_content, re.DOTALL)
    feedbacks_content = re.search(
        rf'{re.escape(base_name)}_feedbacks\[\d+\]\s*=\s*\{{([^}}]+)\}};',
        c_content, re.DOTALL)
    hp_list_content = re.search(
        rf'{re.escape(base_name)}_hp_list\[\d+\]\s*=\s*\{{([^}}]+)\}};',
        c_content, re.DOTALL)
    jump_list_content = re.search(
        rf'{re.escape(base_name)}_jump_list\[\d+\]\s*=\s*\{{([^}}]+)\}};',
        c_content, re.DOTALL)
    
    # Parse float arrays
    halfplanes = []
    if halfplanes_content:
        halfplanes_str = halfplanes_content.group(1)
        float_vals = re.findall(
            r'\(c_float_store\)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
            halfplanes_str)
        halfplanes = [float(val) for val in float_vals]
        # Infer size if not declared
        if n_halfplanes == 0:
            n_halfplanes = len(halfplanes)
    
    feedbacks = []
    if feedbacks_content:
        feedbacks_str = feedbacks_content.group(1)
        float_vals = re.findall(
            r'\(c_float_store\)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
            feedbacks_str)
        feedbacks = [float(val) for val in float_vals]
        if n_feedbacks == 0:
            n_feedbacks = len(feedbacks)
    
    # Parse int arrays
    hp_list = []
    if hp_list_content:
        hp_list_str = hp_list_content.group(1)
        int_vals = re.findall(r'\(c_int\)(\d+)', hp_list_str)
        hp_list = [int(val) for val in int_vals]
        if n_hp_list == 0:
            n_hp_list = len(hp_list)
    
    jump_list = []
    if jump_list_content:
        jump_list_str = jump_list_content.group(1)
        int_vals = re.findall(r'\(c_int\)(\d+)', jump_list_str)
        jump_list = [int(val) for val in int_vals]
        if n_jump_list == 0:
            n_jump_list = len(jump_list)
    
    # Validate consistency
    if n_halfplanes > 0 and len(halfplanes) != n_halfplanes:
        print(f"WARNING: Halfplanes mismatch: declared={n_halfplanes}, parsed={len(halfplanes)}")
    if n_hp_list > 0 and len(hp_list) != n_hp_list:
        print(f"WARNING: HP_list mismatch: declared={n_hp_list}, parsed={len(hp_list)}")
    
    return {
        'halfplanes': halfplanes,
        'feedbacks': feedbacks,
        'hp_list': hp_list,
        'jump_list': jump_list,
        'n_halfplanes': n_halfplanes,
        'n_feedbacks': n_feedbacks,
        'n_hp_list': n_hp_list,
        'n_jump_list': n_jump_list,
        'base_name': base_name
    }

def extract_params_from_h(h_file_path):
    """Extract N_PARAMETER and N_SOLUTION from header"""
    try:
        with open(h_file_path, 'r') as f:
            h_content = f.read()
    except Exception as e:
        print(f"Error reading H file '{h_file_path}': {e}")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(h_file_path))[0]
    
    # Try project-specific then generic pattern
    n_parameter_match = re.search(
        rf'#define\s+{base_name.upper()}_N_PARAMETER\s+(\d+)', h_content)
    if not n_parameter_match:
        n_parameter_match = re.search(r'#define\s+\w*N_PARAMETER\s+(\d+)', h_content)
    
    n_solution_match = re.search(
        rf'#define\s+{base_name.upper()}_N_SOLUTION\s+(\d+)', h_content)
    if not n_solution_match:
        n_solution_match = re.search(r'#define\s+\w*N_SOLUTION\s+(\d+)', h_content)
    
    n_parameter = int(n_parameter_match.group(1)) if n_parameter_match else 2
    n_solution = int(n_solution_match.group(1)) if n_solution_match else 2
    
    return {
        'n_parameter': n_parameter,
        'n_solution': n_solution
    }

def calculate_exact_bst_depth(jump_list, hp_list, debug=False):
    """Calculate exact BST depth via BFS (edge count from root to deepest leaf)"""
    if not jump_list or not hp_list:
        return 4  # Safe minimum for empty tree
    
    max_depth = 0
    leaf_count = 0
    total_nodes = len(jump_list)
    
    queue = deque([(0, 0)])  # (node_id, depth)
    visited = set()
    
    if debug:
        print(f"\n{'='*60}\nBST Depth Analysis (BFS)\n{'='*60}")
        print(f"Total nodes: {total_nodes}\n")
    
    while queue:
        node_id, depth = queue.popleft()
        
        if node_id in visited or node_id >= total_nodes:
            continue
        
        visited.add(node_id)
        jump = jump_list[node_id]
        
        if jump == 0:  # Leaf node
            leaf_count += 1
            if depth > max_depth:
                max_depth = depth
                if debug:
                    print(f"  Max depth: {depth} at node {node_id} (hp={hp_list[node_id]})")
        else:  # Internal node
            left_child = node_id + jump
            right_child = node_id + jump + 1
            if left_child < total_nodes:
                queue.append((left_child, depth + 1))
            if right_child < total_nodes:
                queue.append((right_child, depth + 1))
    
    if debug:
        print(f"\nVisited: {len(visited)}/{total_nodes}, Leaves: {leaf_count}")
        print(f"Max depth: {max_depth}\n{'='*60}\n")
    
    return max(4, max_depth)  # Clamp to minimum 4

def analyze_array_range(values):
    """Determine optimal Q format for fixed-point representation"""
    if not values:
        return {
            'min': 0.0, 'max': 0.0, 'abs_max': 0.0,
            'int_bits': 2, 'frac_bits': 14, 'scale_factor': 16384
        }
    
    min_val = min(values)
    max_val = max(values)
    abs_max = max(abs(min_val), abs(max_val))
    
    # Calculate required integer bits (including sign)
    if abs_max < 1.0:
        int_bits = 2
    else:
        int_bits = int(math.ceil(math.log2(abs_max + 1))) + 1
    
    int_bits = max(2, int_bits)
    frac_bits = 16 - int_bits
    
    return {
        'min': min_val, 'max': max_val, 'abs_max': abs_max,
        'int_bits': int_bits, 'frac_bits': frac_bits,
        'scale_factor': 2**frac_bits
    }

def analyze_all_arrays(arrays, override_int_bits=None):
    """Analyze both halfplanes and feedbacks arrays"""
    formats = {
        'halfplanes': analyze_array_range(arrays['halfplanes']),
        'feedbacks': analyze_array_range(arrays['feedbacks'])
    }
    
    if override_int_bits is not None:
        for fmt in formats.values():
            fmt['int_bits'] = override_int_bits
            fmt['frac_bits'] = 16 - override_int_bits
            fmt['scale_factor'] = 2**(16 - override_int_bits)
    
    return formats

def float_to_fixed(value, format_info):
    """Convert float to 16-bit fixed-point with saturation"""
    scale = format_info['scale_factor']
    max_val = (2**(format_info['int_bits']-1)) - (1.0/scale)
    min_val = -(2**(format_info['int_bits']-1))
    
    # Saturate to valid range
    value = max(min_val, min(max_val, value))
    
    # Round and convert
    fixed_val = int(round(value * scale))
    if fixed_val < 0:
        fixed_val = (1 << 16) + fixed_val
    
    return format(fixed_val & 0xFFFF, '04X')

def generate_verilog_defines(params, arrays, formats, bst_depth):
    """Generate Verilog configuration header"""
    sol_per_node = params['n_solution'] * (params['n_parameter'] + 1)
    halfplane_stride = params['n_parameter'] + 1
    base_name = arrays['base_name']
    
    return f'''// Auto-generated PDAQP configuration
// Source: {base_name}.h, {base_name}.c

`define PDAQP_N_PARAMETER {params['n_parameter']}
`define PDAQP_N_SOLUTION {params['n_solution']}
`define PDAQP_TREE_NODES {arrays['n_hp_list']}
`define PDAQP_HALFPLANES {arrays['n_halfplanes']}
`define PDAQP_FEEDBACKS {arrays['n_feedbacks']}

`define PDAQP_SOL_PER_NODE {sol_per_node}
`define PDAQP_HALFPLANE_STRIDE {halfplane_stride}
`define PDAQP_ESTIMATED_BST_DEPTH {bst_depth}

// Fixed-point Q format for halfplanes
`define HALFPLANE_INT_BITS {formats['halfplanes']['int_bits']}
`define HALFPLANE_FRAC_BITS {formats['halfplanes']['frac_bits']}
`define HALFPLANE_SCALE_FACTOR {formats['halfplanes']['scale_factor']}

// Fixed-point Q format for feedbacks
`define FEEDBACK_INT_BITS {formats['feedbacks']['int_bits']}
`define FEEDBACK_FRAC_BITS {formats['feedbacks']['frac_bits']}
`define FEEDBACK_SCALE_FACTOR {formats['feedbacks']['scale_factor']}

`define INPUT_DATA_WIDTH 16
`define OUTPUT_DATA_WIDTH 16

// Backward compatibility
`define PDAQP_FIXED_POINT_BITS {formats['feedbacks']['frac_bits']}
`define PDAQP_SCALE_FACTOR {formats['feedbacks']['scale_factor']}
`define INPUT_INT_BITS {formats['feedbacks']['int_bits']}
`define OUTPUT_INT_BITS {formats['feedbacks']['int_bits']}
`define OUTPUT_FRAC_BITS {formats['feedbacks']['frac_bits']}
'''

def generate_mem_files(arrays, params, output_dir, formats, debug=False):
    """Generate .mem files for Verilog $readmemh"""
    os.makedirs(output_dir, exist_ok=True)
    base_name = arrays['base_name']
    
    if debug:
        print("\nFixed-point conversion:")
        print(f"  Halfplanes: Q{formats['halfplanes']['int_bits']}.{formats['halfplanes']['frac_bits']}")
        print(f"  Feedbacks:  Q{formats['feedbacks']['int_bits']}.{formats['feedbacks']['frac_bits']}")
    
    # Halfplanes
    with open(os.path.join(output_dir, f'{base_name}_halfplanes.mem'), 'w') as f:
        for val in arrays['halfplanes']:
            f.write(f"{float_to_fixed(val, formats['halfplanes'])}\n")
    
    # Feedbacks
    with open(os.path.join(output_dir, f'{base_name}_feedbacks.mem'), 'w') as f:
        for val in arrays['feedbacks']:
            f.write(f"{float_to_fixed(val, formats['feedbacks'])}\n")
    
    # HP list
    with open(os.path.join(output_dir, f'{base_name}_hp_list.mem'), 'w') as f:
        for val in arrays['hp_list']:
            f.write(f"{format(val & 0x3FF, '03X')}\n")
    
    # Jump list
    with open(os.path.join(output_dir, f'{base_name}_jump_list.mem'), 'w') as f:
        for val in arrays['jump_list']:
            f.write(f"{format(val & 0x3FF, '03X')}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate PDAQP Verilog config')
    parser.add_argument('-c', '--c-file', required=True, help='C source file')
    parser.add_argument('-H', '--header-file', required=True, help='Header file')
    parser.add_argument('-o', '--output-dir', default='generated_files', help='Output directory')
    parser.add_argument('-i', '--int-bits', type=int, help='Override integer bits')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug output')
    
    args = parser.parse_args()
    
    print("Extracting data...")
    arrays = extract_arrays_from_c(args.c_file)
    params = extract_params_from_h(args.header_file)
    
    bst_depth = calculate_exact_bst_depth(arrays['jump_list'], arrays['hp_list'], args.debug)
    formats = analyze_all_arrays(arrays, args.int_bits)
    
    print(f"\nSummary:")
    print(f"  Project: {arrays['base_name']}")
    print(f"  Params: {params['n_parameter']}, Solutions: {params['n_solution']}")
    print(f"  Tree nodes: {arrays['n_hp_list']}, BST depth: {bst_depth}")
    
    # Generate files
    verilog_defines = generate_verilog_defines(params, arrays, formats, bst_depth)
    os.makedirs(args.output_dir, exist_ok=True)
    config_file = os.path.join(args.output_dir, f'{arrays["base_name"]}_config.vh')
    with open(config_file, 'w') as f:
        f.write(verilog_defines)
    
    generate_mem_files(arrays, params, args.output_dir, formats, args.debug)
    
    print(f"\n{'='*60}\nGeneration complete!\n{'='*60}")
    print(f"Output: {args.output_dir}/")
    print(f"  • {arrays['base_name']}_config.vh")
    print(f"  • {arrays['base_name']}_{{halfplanes,feedbacks,hp_list,jump_list}}.mem")

if __name__ == "__main__":
    main()