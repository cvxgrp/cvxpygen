#!/usr/bin/env python3
"""
Automatic Hardware Generator Selection for PDAQP
Analyzes configuration and DSP constraints to select optimal implementation
"""

import argparse
import subprocess
import sys
from pathlib import Path
import re

def parse_config(config_file):
    """Extract key parameters from config file"""
    params = {}
    try:
        with open(config_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Config file {config_file} not found")
        sys.exit(1)
    
    # Extract parameters
    for param in ['PDAQP_N_PARAMETER', 'PDAQP_N_SOLUTION', 'PDAQP_ESTIMATED_BST_DEPTH']:
        match = re.search(rf'`define\s+{param}\s+(\d+)', content)
        if match:
            params[param] = int(match.group(1))
    
    return params

def get_script_dir():
    """Get the directory where this script is located"""
    return Path(__file__).parent.resolve()

def select_generator(n_param, n_solution, dsp_limit):
    """Select optimal generator based on constraints"""
    dsp_needed = n_param * n_solution
    
    if dsp_limit >= 1000:  # Effectively unlimited
        return 'unconstrained', f'No DSP constraints - free to use DSP/LUT optimally'
    elif dsp_limit >= dsp_needed:
        return 'hybrid', f'DSP sufficient ({dsp_limit} >= {dsp_needed}) - Hybrid Parallel'
    elif dsp_limit >= n_param:
        parallel_solutions = dsp_limit // n_param
        return 'solution', f'DSP moderate ({n_param} <= {dsp_limit} < {dsp_needed}) - Solution-Level TD ({parallel_solutions} solutions/cycle)'
    else:
        parallel_params = dsp_limit
        return 'parameter', f'DSP limited ({dsp_limit} < {n_param}) - Parameter-Level TD ({parallel_params} params/cycle)'

def estimate_performance(gen_type, n_param, n_solution, dsp_limit, bst_depth):
    """Estimate performance metrics for selected generator"""
    metrics = {}
    
    if gen_type == 'unconstrained':
        metrics['DSP Usage'] = f'{n_param * n_solution} (optimal)'
        metrics['LUT Usage'] = 'Minimal (DSP handles multiplications)'
        metrics['Latency'] = f'{bst_depth + 2} cycles'
        metrics['Throughput'] = '100% (1 input/cycle)'
        metrics['Resource'] = 'Optimal DSP/LUT balance'
        
    elif gen_type == 'hybrid':
        metrics['DSP Usage'] = f'{n_param * n_solution} (feedback only)'
        metrics['LUT Usage'] = 'Moderate (halfplane computations)'
        metrics['Latency'] = f'{bst_depth + 2} cycles'
        metrics['Throughput'] = '100% (1 input/cycle)'
        metrics['Resource'] = 'DSP for feedback, LUT for halfplanes'
        
    elif gen_type == 'solution':
        parallel_solutions = dsp_limit // n_param
        solution_cycles = (n_solution + parallel_solutions - 1) // parallel_solutions * 3
        metrics['DSP Usage'] = f'{min(dsp_limit, n_param * parallel_solutions)}'
        metrics['LUT Usage'] = 'Low (control logic only)'
        metrics['Latency'] = f'{bst_depth + solution_cycles + 2} cycles'
        metrics['Throughput'] = f'{(parallel_solutions/n_solution)*100:.1f}%'
        metrics['Resource'] = f'{parallel_solutions} solutions in parallel'
        
    else:  # parameter
        parallel_params = dsp_limit
        param_cycles = (n_param + parallel_params - 1) // parallel_params
        total_cycles = param_cycles * n_solution * 3
        metrics['DSP Usage'] = f'{parallel_params}'
        metrics['LUT Usage'] = 'Low (control logic only)'
        metrics['Latency'] = f'{bst_depth + total_cycles + 2} cycles'
        metrics['Throughput'] = f'{(parallel_params/n_param)*100:.1f}%'
        metrics['Resource'] = f'{parallel_params} params in parallel'
    
    return metrics

def generate_verilator_testbench(config_file, output_dir, benchmark_dir):
    """Generate Verilator C++ testbench"""
    print(f"\n{'='*70}")
    print("Generating Verilator C++ Testbench")
    print(f"{'='*70}")
    
    # Check if benchmark directory exists
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        print(f"Warning: Benchmark directory {benchmark_dir} not found")
        print("C++ testbench will use dummy data")
    
    # Create c_tb directory
    ctb_dir = Path(output_dir) / "c_tb"
    ctb_dir.mkdir(parents=True, exist_ok=True)
    
    # Get script directory and construct full path
    script_dir = get_script_dir()
    testbench_script = script_dir / 'generate_verilator_testbench.py'
    
    # Verify script exists
    if not testbench_script.exists():
        print(f"Error: Testbench script not found at {testbench_script}")
        return False
    
    # Run testbench generator
    cmd = [
        'python3', 
        str(testbench_script),
        config_file,
        '-b', str(benchmark_path),
        '-o', str(ctb_dir)
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n[OK] C++ testbench generated successfully")
        print(f"[OK] Output: {ctb_dir}/sim_main.cpp")
    else:
        print(f"\n[WARNING] C++ testbench generation failed (return code {result.returncode})")
        print("Continuing without C++ testbench...")
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Auto-select PDAQP hardware generator')
    parser.add_argument('config_file', help='Configuration .vh file')
    parser.add_argument('-o', '--output', default='generated_hardware', help='Output directory')
    parser.add_argument('-d', '--dsp-limit', type=int, default=1000, help='DSP block limit')
    parser.add_argument('-b', '--benchmark', default='benchmark', help='Benchmark data directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--skip-testbench', action='store_true', help='Skip C++ testbench generation')
    
    args = parser.parse_args()
    
    # Parse configuration
    config = parse_config(args.config_file)
    n_param = config.get('PDAQP_N_PARAMETER', 2)
    n_solution = config.get('PDAQP_N_SOLUTION', 2)
    bst_depth = config.get('PDAQP_ESTIMATED_BST_DEPTH', 7)
    
    # Select generator
    gen_type, reason = select_generator(n_param, n_solution, args.dsp_limit)
    
    # Estimate performance
    metrics = estimate_performance(gen_type, n_param, n_solution, args.dsp_limit, bst_depth)
    
    # Format DSP constraint display
    dsp_display = "No DSP constraints" if args.dsp_limit >= 1000 else f"{args.dsp_limit} blocks"
    
    print(f"\n{'='*70}")
    print("PDAQP Hardware Generator Auto-Selection")
    print(f"{'='*70}")
    print(f"Configuration file: {args.config_file}")
    print(f"Problem size: {n_param} parameters × {n_solution} solutions")
    print(f"BST depth: {bst_depth}")
    print(f"DSP constraint: {dsp_display}")
    print(f"\n{'-'*70}")
    print(f"Selected Implementation: {gen_type.upper()}")
    print(f"Reason: {reason}")
    print(f"\n{'-'*70}")
    print("Expected Performance:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")
    
    # Map generator type to script
    script_map = {
        'unconstrained': 'generate_unconstrained.py',
        'hybrid': 'generate_hybrid_parallel.py',
        'solution': 'generate_solution_level.py',
        'parameter': 'generate_parameter_level.py'
    }
    
    script_name = script_map[gen_type]
    
    # Get script directory and construct full path
    script_dir = get_script_dir()
    script_path = script_dir / script_name
    
    # Verify script exists
    if not script_path.exists():
        print(f"Error: Generator script not found at {script_path}")
        print(f"Expected location: {script_dir}/{script_name}")
        sys.exit(1)
    
    # Use the output directory directly without suffix
    output_dir = args.output
    
    # Build command with full script path
    cmd = ['python3', str(script_path), args.config_file, '-o', output_dir]
    if gen_type != 'unconstrained':
        cmd.extend(['-d', str(args.dsp_limit)])
    if args.verbose:
        cmd.append('-v')
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Execute hardware generation
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n[OK] Successfully generated {gen_type} implementation")
        print(f"[OK] Output directory: {output_dir}")
        print(f"\nGenerated structure:")
        print(f"  {output_dir}/")
        print(f"  |- rtl/         (hardware modules)")
        print(f"  +- tb/          (testbenches)")
        
        # Generate Verilator C++ testbench (only for unconstrained for now)
        if not args.skip_testbench:
            if gen_type == 'unconstrained':
                testbench_success = generate_verilator_testbench(
                    args.config_file,
                    output_dir,
                    args.benchmark
                )
                if testbench_success:
                    print(f"  +- c_tb/        (Verilator C++ testbench)")
            else:
                print(f"\n[INFO] C++ testbench generation currently only supported for unconstrained implementation")
        
    else:
        print(f"\n[ERROR] Generation failed with return code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == '__main__':
    main()