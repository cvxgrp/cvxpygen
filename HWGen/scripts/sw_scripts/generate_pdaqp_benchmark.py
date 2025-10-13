#!/usr/bin/env python3
"""
PDAQP Benchmark Generator
Generates an ipynb file for benchmarking PDAQP implementations
"""

import re
import json
import argparse
import os.path

def extract_data_from_c_file(c_file_path, h_file_path):
    """
    Extract arrays and parameters from C and H files
    """
    # Read C file content
    with open(c_file_path, 'r') as f:
        c_content = f.read()
    
    # Read H file content
    with open(h_file_path, 'r') as f:
        h_content = f.read()
    
    # Extract base name from filename
    base_name = os.path.splitext(os.path.basename(c_file_path))[0]
    
    # Extract parameters from H file
    n_parameter_match = re.search(r'#define\s+(\w+)_N_PARAMETER\s+(\d+)', h_content)
    n_solution_match = re.search(r'#define\s+(\w+)_N_SOLUTION\s+(\d+)', h_content)
    
    if not n_parameter_match or not n_solution_match:
        raise ValueError("Could not find N_PARAMETER or N_SOLUTION in header file")
    
    prefix = n_parameter_match.group(1)
    n_parameter = int(n_parameter_match.group(2))
    n_solution = int(n_solution_match.group(2))
    
    # Modified regex patterns to match the actual format in your C file
    halfplanes_pattern = re.compile(r'c_float_store\s+(\w+)_halfplanes\[\d+\]\s*=\s*\{([^}]+)\}', re.DOTALL)
    feedbacks_pattern = re.compile(r'c_float_store\s+(\w+)_feedbacks\[\d+\]\s*=\s*\{([^}]+)\}', re.DOTALL)
    hp_list_pattern = re.compile(r'c_int\s+(\w+)_hp_list\[\d+\]\s*=\s*\{([^}]+)\}', re.DOTALL)
    jump_list_pattern = re.compile(r'c_int\s+(\w+)_jump_list\[\d+\]\s*=\s*\{([^}]+)\}', re.DOTALL)
    
    # Function to extract float values from array string
    def extract_float_values(array_str):
        # Find all float values in the string (handle scientific notation)
        values = re.findall(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', array_str)
        return [float(v) for v in values]
    
    # Function to extract int values from array string
    def extract_int_values(array_str):
        # Find all int values in the string
        values = re.findall(r'\(\w+\)(\d+)', array_str)
        return [int(v) for v in values]
    
    # Extract arrays
    halfplanes_match = halfplanes_pattern.search(c_content)
    feedbacks_match = feedbacks_pattern.search(c_content)
    hp_list_match = hp_list_pattern.search(c_content)
    jump_list_match = jump_list_pattern.search(c_content)
    
    if not halfplanes_match:
        print("Failed to extract halfplanes array")
    if not feedbacks_match:
        print("Failed to extract feedbacks array")
    if not hp_list_match:
        print("Failed to extract hp_list array")
    if not jump_list_match:
        print("Failed to extract jump_list array")
        
    if not all([halfplanes_match, feedbacks_match, hp_list_match, jump_list_match]):
        # Print the first few hundred characters of the C file for debugging
        print("First 300 characters of C file:")
        print(c_content[:300])
        raise ValueError("Could not find all required arrays in C file")
    
    # Check if prefix in arrays matches prefix from header
    array_prefix = halfplanes_match.group(1)
    if array_prefix != prefix:
        print(f"Warning: Prefix mismatch between header ({prefix}) and arrays ({array_prefix})")
        print(f"Using array prefix: {array_prefix}")
        prefix = array_prefix
    
    halfplanes = extract_float_values(halfplanes_match.group(2))
    feedbacks = extract_float_values(feedbacks_match.group(2))
    hp_list = extract_int_values(hp_list_match.group(2))
    jump_list = extract_int_values(jump_list_match.group(2))
    
    return {
        'prefix': prefix,
        'n_parameter': n_parameter,
        'n_solution': n_solution,
        'halfplanes': halfplanes,
        'feedbacks': feedbacks,
        'hp_list': hp_list,
        'jump_list': jump_list
    }


def generate_notebook(data):
    """
    Generate a Jupyter notebook for benchmarking the PDAQP implementation
    """
    prefix = data['prefix']
    n_parameter = data['n_parameter']
    n_solution = data['n_solution']
    halfplanes = data['halfplanes']
    feedbacks = data['feedbacks']
    hp_list = data['hp_list']
    jump_list = data['jump_list']
    
    # Create notebook cells
    cells = []
    
    # Title and imports cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": f"# {prefix.upper()} Algorithm Benchmarking\n\n"
                  f"This notebook benchmarks the performance of the {prefix.upper()} "
                  f"algorithm implementation on CPU."
    })
    
    # Imports cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": "import numpy as np\n"
                  "import time\n"
                  "import matplotlib.pyplot as plt\n"
                  "import statistics  # For calculating standard deviation\n"
    })
    
    # Constants and arrays cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": f"# Constants (based on the algorithm structure)\n"
                  f"{prefix.upper()}_N_PARAMETER = {n_parameter}  # Input parameter dimension\n"
                  f"{prefix.upper()}_N_SOLUTION = {n_solution}   # Output solution dimension\n\n"
                  f"# Define arrays from the C code implementation\n"
                  f"# These are the coefficients for the halfplanes defining the decision regions\n"
                  f"{prefix}_halfplanes = np.array([\n    "
                  f"{', '.join(str(x) for x in halfplanes)}\n"
                  f"], dtype=np.float64)\n\n"
                  f"# Feedback coefficients for the affine functions at the leaf nodes\n"
                  f"{prefix}_feedbacks = np.array([\n    "
                  f"{', '.join(str(x) for x in feedbacks)}\n"
                  f"], dtype=np.float64)\n\n"
                  f"# Indices for halfplanes and jumps in the decision tree\n"
                  f"{prefix}_hp_list = np.array([\n    "
                  f"{', '.join(str(x) for x in hp_list)}\n"
                  f"], dtype=np.int32)\n\n"
                  f"{prefix}_jump_list = np.array([\n    "
                  f"{', '.join(str(x) for x in jump_list)}\n"
                  f"], dtype=np.int32)\n"
    })
    
    # Algorithm implementation cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": f"def {prefix}_evaluate(parameter, solution):\n"
                  f"    \"\"\"\n"
                  f"    Standard implementation of the {prefix.upper()} algorithm.\n"
                  f"    Direct translation of the C function to Python.\n"
                  f"    \n"
                  f"    Args:\n"
                  f"        parameter: Input vector of dimension {prefix.upper()}_N_PARAMETER\n"
                  f"        solution: Output vector of dimension {prefix.upper()}_N_SOLUTION\n"
                  f"    \n"
                  f"    Returns:\n"
                  f"        None (solution is modified in-place)\n"
                  f"    \"\"\"\n"
                  f"    id = 0\n"
                  f"    next_id = id + {prefix}_jump_list[id]\n"
                  f"    \n"
                  f"    # Traverse the decision tree until a leaf node is reached\n"
                  f"    while next_id != id:\n"
                  f"        # Compute halfplane value\n"
                  f"        disp = {prefix}_hp_list[id] * ({prefix.upper()}_N_PARAMETER + 1)\n"
                  f"        val = 0\n"
                  f"        for i in range({prefix.upper()}_N_PARAMETER):\n"
                  f"            val += parameter[i] * {prefix}_halfplanes[disp + i]\n"
                  f"        \n"
                  f"        # Determine which branch to take based on halfplane evaluation\n"
                  f"        id = next_id + (1 if val <= {prefix}_halfplanes[disp + {prefix.upper()}_N_PARAMETER] else 0)\n"
                  f"        next_id = id + {prefix}_jump_list[id]\n"
                  f"    \n"
                  f"    # Leaf node reached -> evaluate affine function\n"
                  f"    disp = {prefix}_hp_list[id] * ({prefix.upper()}_N_PARAMETER + 1) * {prefix.upper()}_N_SOLUTION\n"
                  f"    for i in range({prefix.upper()}_N_SOLUTION):\n"
                  f"        val = 0\n"
                  f"        for j in range({prefix.upper()}_N_PARAMETER):\n"
                  f"            val += parameter[j] * {prefix}_feedbacks[disp + j]\n"
                  f"        val += {prefix}_feedbacks[disp + {prefix.upper()}_N_PARAMETER]\n"
                  f"        solution[i] = val\n"
                  f"        disp += {prefix.upper()}_N_PARAMETER + 1\n\n"
                  f"def {prefix}_evaluate_vectorized(parameter, solution):\n"
                  f"    \"\"\"\n"
                  f"    Optimized version of the {prefix.upper()} algorithm using NumPy vectorization.\n"
                  f"    \n"
                  f"    Args:\n"
                  f"        parameter: Input vector of dimension {prefix.upper()}_N_PARAMETER\n"
                  f"        solution: Output vector of dimension {prefix.upper()}_N_SOLUTION\n"
                  f"    \n"
                  f"    Returns:\n"
                  f"        None (solution is modified in-place)\n"
                  f"    \"\"\"\n"
                  f"    id = 0\n"
                  f"    next_id = id + {prefix}_jump_list[id]\n"
                  f"    \n"
                  f"    # Traverse the decision tree until a leaf node is reached\n"
                  f"    while next_id != id:\n"
                  f"        # Compute halfplane value using vectorized dot product\n"
                  f"        disp = {prefix}_hp_list[id] * ({prefix.upper()}_N_PARAMETER + 1)\n"
                  f"        coeffs = {prefix}_halfplanes[disp:disp+{prefix.upper()}_N_PARAMETER]\n"
                  f"        constant = {prefix}_halfplanes[disp+{prefix.upper()}_N_PARAMETER]\n"
                  f"        val = np.dot(parameter, coeffs)\n"
                  f"        \n"
                  f"        # Determine which branch to take based on halfplane evaluation\n"
                  f"        id = next_id + (1 if val <= constant else 0)\n"
                  f"        next_id = id + {prefix}_jump_list[id]\n"
                  f"    \n"
                  f"    # Leaf node reached -> evaluate affine function using vectorized operations\n"
                  f"    disp = {prefix}_hp_list[id] * ({prefix.upper()}_N_PARAMETER + 1) * {prefix.upper()}_N_SOLUTION\n"
                  f"    \n"
                  f"    for i in range({prefix.upper()}_N_SOLUTION):\n"
                  f"        offset = disp + i*({prefix.upper()}_N_PARAMETER + 1)\n"
                  f"        coeffs = {prefix}_feedbacks[offset:offset+{prefix.upper()}_N_PARAMETER]\n"
                  f"        constant = {prefix}_feedbacks[offset+{prefix.upper()}_N_PARAMETER]\n"
                  f"        solution[i] = np.dot(parameter, coeffs) + constant\n"
    })
    
    # Performance testing functions cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": f"def test_performance(n_tests=10000, use_vectorized=False):\n"
                  f"    \"\"\"\n"
                  f"    Test CPU performance of the algorithm.\n"
                  f"    \n"
                  f"    Args:\n"
                  f"        n_tests: Number of random inputs to test\n"
                  f"        use_vectorized: Whether to use the vectorized implementation\n"
                  f"    \n"
                  f"    Returns:\n"
                  f"        avg_time: Average execution time per evaluation (seconds)\n"
                  f"        total_time: Total execution time for all evaluations (seconds)\n"
                  f"        solutions: Array of computed solutions\n"
                  f"    \"\"\"\n"
                  f"    # Generate random parameter vectors\n"
                  f"    parameters = np.random.randn(n_tests, {prefix.upper()}_N_PARAMETER)\n"
                  f"    solutions = np.zeros((n_tests, {prefix.upper()}_N_SOLUTION))\n"
                  f"    \n"
                  f"    # Select implementation\n"
                  f"    evaluate_fn = {prefix}_evaluate_vectorized if use_vectorized else {prefix}_evaluate\n"
                  f"    \n"
                  f"    # Warm up to avoid measuring JIT compilation time\n"
                  f"    for i in range(10):\n"
                  f"        evaluate_fn(parameters[i % 10], solutions[i % 10])\n"
                  f"    \n"
                  f"    # Measure performance\n"
                  f"    start_time = time.time()\n"
                  f"    for i in range(n_tests):\n"
                  f"        evaluate_fn(parameters[i], solutions[i])\n"
                  f"    end_time = time.time()\n"
                  f"    \n"
                  f"    total_time = end_time - start_time\n"
                  f"    avg_time = total_time / n_tests\n"
                  f"    \n"
                  f"    return avg_time, total_time, solutions\n\n"
                  f"def benchmark_and_compare():\n"
                  f"    \"\"\"\n"
                  f"    Run benchmarks and compare standard vs vectorized implementations.\n"
                  f"    \n"
                  f"    Returns:\n"
                  f"        std_avg_time: Average execution time for standard implementation\n"
                  f"        vec_avg_time: Average execution time for vectorized implementation\n"
                  f"    \"\"\"\n"
                  f"    n_tests = 10000\n"
                  f"    print(f\"Running performance tests with {{n_tests}} random inputs...\")\n"
                  f"    \n"
                  f"    # Test standard implementation\n"
                  f"    std_avg_time, std_total_time, std_solutions = test_performance(n_tests, use_vectorized=False)\n"
                  f"    print(f\"Standard implementation:\")\n"
                  f"    print(f\"  Average time: {{std_avg_time * 1e6:.2f}} microseconds per evaluation\")\n"
                  f"    print(f\"  Total time: {{std_total_time:.4f}} seconds for {{n_tests}} evaluations\")\n"
                  f"    \n"
                  f"    # Test vectorized implementation\n"
                  f"    vec_avg_time, vec_total_time, vec_solutions = test_performance(n_tests, use_vectorized=True)\n"
                  f"    print(f\"Vectorized implementation:\")\n"
                  f"    print(f\"  Average time: {{vec_avg_time * 1e6:.2f}} microseconds per evaluation\")\n"
                  f"    print(f\"  Total time: {{vec_total_time:.4f}} seconds for {{n_tests}} evaluations\")\n"
                  f"    \n"
                  f"    # Calculate speedup\n"
                  f"    speedup = std_avg_time / vec_avg_time\n"
                  f"    print(f\"Speedup: {{speedup:.2f}}x\")\n"
                  f"    \n"
                  f"    # Verify both implementations give the same results\n"
                  f"    if np.allclose(std_solutions, vec_solutions):\n"
                  f"        print(\"Both implementations produce identical results ✓\")\n"
                  f"    else:\n"
                  f"        print(\"WARNING: Results differ between implementations!\")\n"
                  f"    \n"
                  f"    return std_avg_time, vec_avg_time\n"
    })
    
    # Extended benchmark function cell (simplified without power/energy estimation)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": f"def extended_benchmark():\n"
                  f"    \"\"\"\n"
                  f"    Extended benchmark with detailed timing metrics.\n"
                  f"    Measures execution time, throughput, and latency jitter.\n"
                  f"    \n"
                  f"    Returns:\n"
                  f"        metrics: Dictionary containing measured performance metrics\n"
                  f"    \"\"\"\n"
                  f"    n_tests = 10000\n"
                  f"    print(f\"Running extended performance tests with {{n_tests}} random inputs...\")\n"
                  f"    \n"
                  f"    # Generate random parameter vectors\n"
                  f"    parameters = np.random.randn(n_tests, {prefix.upper()}_N_PARAMETER)\n"
                  f"    std_solutions = np.zeros((n_tests, {prefix.upper()}_N_SOLUTION))\n"
                  f"    vec_solutions = np.zeros((n_tests, {prefix.upper()}_N_SOLUTION))\n"
                  f"    \n"
                  f"    # Measure initialization overhead\n"
                  f"    start_time = time.time()\n"
                  f"    # Simulate initialization process\n"
                  f"    dummy = np.zeros_like({prefix}_halfplanes)\n"
                  f"    dummy2 = np.zeros_like({prefix}_feedbacks)\n"
                  f"    init_time = (time.time() - start_time) * 1e6  # microseconds\n"
                  f"    \n"
                  f"    # Warm up phase\n"
                  f"    for i in range(100):\n"
                  f"        {prefix}_evaluate(parameters[i % 100], std_solutions[i % 100])\n"
                  f"        {prefix}_evaluate_vectorized(parameters[i % 100], vec_solutions[i % 100])\n"
                  f"    \n"
                  f"    # Measure standard implementation with detailed timing\n"
                  f"    std_times = []\n"
                  f"    start_time = time.time()\n"
                  f"    \n"
                  f"    for i in range(n_tests):\n"
                  f"        iter_start = time.time()\n"
                  f"        {prefix}_evaluate(parameters[i], std_solutions[i])\n"
                  f"        std_times.append((time.time() - iter_start) * 1e6)  # microseconds\n"
                  f"    \n"
                  f"    std_total_time = time.time() - start_time\n"
                  f"    \n"
                  f"    # Measure vectorized implementation\n"
                  f"    vec_times = []\n"
                  f"    start_time = time.time()\n"
                  f"    \n"
                  f"    for i in range(n_tests):\n"
                  f"        iter_start = time.time()\n"
                  f"        {prefix}_evaluate_vectorized(parameters[i], vec_solutions[i])\n"
                  f"        vec_times.append((time.time() - iter_start) * 1e6)  # microseconds\n"
                  f"    \n"
                  f"    vec_total_time = time.time() - start_time\n"
                  f"    \n"
                  f"    # Calculate performance metrics\n"
                  f"    std_avg_time = sum(std_times) / n_tests\n"
                  f"    vec_avg_time = sum(vec_times) / n_tests\n"
                  f"    \n"
                  f"    # Calculate throughput in samples/second and convert to KOPS\n"
                  f"    std_throughput = 1e6 / std_avg_time  # samples per second\n"
                  f"    std_throughput_kops = std_throughput / 1e3  # convert to KOPS (thousands of samples/s)\n"
                  f"    \n"
                  f"    vec_throughput = 1e6 / vec_avg_time\n"
                  f"    vec_throughput_kops = vec_throughput / 1e3\n"
                  f"    \n"
                  f"    std_jitter = statistics.stdev(std_times)\n"
                  f"    vec_jitter = statistics.stdev(vec_times)\n"
                  f"    \n"
                  f"    # Verify results consistency\n"
                  f"    results_match = np.allclose(std_solutions, vec_solutions)\n"
                  f"    if not results_match:\n"
                  f"        print(\"WARNING: Results differ between implementations!\")\n"
                  f"    \n"
                  f"    # Print detailed results table\n"
                  f"    print(\"\\n============ Detailed Performance Metrics ============\")\n"
                  f"    print(f\"Standard Implementation:\")\n"
                  f"    print(f\"  Execution time: {{std_avg_time:.2f}} µs\")\n"
                  f"    print(f\"  Throughput: {{std_throughput_kops:.2f}} KOPS ({{std_throughput:.2f}} samples/s)\")\n"
                  f"    print(f\"  Latency jitter: ±{{std_jitter:.2f}} µs\")\n"
                  f"    print(f\"\\nVectorized Implementation:\")\n"
                  f"    print(f\"  Execution time: {{vec_avg_time:.2f}} µs\")\n"
                  f"    print(f\"  Throughput: {{vec_throughput_kops:.2f}} KOPS ({{vec_throughput:.2f}} samples/s)\")\n"
                  f"    print(f\"  Latency jitter: ±{{vec_jitter:.2f}} µs\")\n"
                  f"    print(f\"\\nComparison:\")\n"
                  f"    print(f\"  Speedup: {{std_avg_time/vec_avg_time:.2f}}x\")\n"
                  f"    print(f\"  Initialization overhead: {{init_time:.2f}} µs\")\n"
                  f"    print(\"=====================================================\\n\")\n"
                  f"    \n"
                  f"    return {{\n"
                  f"        \"std_exec_time\": std_avg_time,\n"
                  f"        \"std_throughput\": std_throughput_kops,\n"
                  f"        \"std_jitter\": std_jitter,\n"
                  f"        \"vec_exec_time\": vec_avg_time,\n"
                  f"        \"vec_throughput\": vec_throughput_kops,\n"
                  f"        \"vec_jitter\": vec_jitter,\n"
                  f"        \"init_time\": init_time,\n"
                  f"        \"speedup\": std_avg_time/vec_avg_time\n"
                  f"    }}\n"
    })
    
    # Visualization function cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": f"def visualize_decision_boundaries(resolution=50):\n"
                  f"    \"\"\"\n"
                  f"    Visualize the decision boundaries of the algorithm.\n"
                  f"    Works for 2D parameter space with multiple outputs.\n"
                  f"    \n"
                  f"    Args:\n"
                  f"        resolution: Number of points along each axis in the grid\n"
                  f"    \"\"\"\n"
                  f"    # Only supports 2D parameter space\n"
                  f"    if {prefix.upper()}_N_PARAMETER != 2:\n"
                  f"        print(f\"Visualization only supported for 2D parameter space, but N_PARAMETER = {{{prefix.upper()}_N_PARAMETER}}\")\n"
                  f"        return\n"
                  f"    \n"
                  f"    # Generate a grid of parameter values\n"
                  f"    x = np.linspace(-3, 3, resolution)\n"
                  f"    y = np.linspace(-3, 3, resolution)\n"
                  f"    X, Y = np.meshgrid(x, y)\n"
                  f"    \n"
                  f"    # Evaluate the function at each grid point\n"
                  f"    Z = np.zeros((resolution, resolution, {prefix.upper()}_N_SOLUTION))\n"
                  f"    solution = np.zeros({prefix.upper()}_N_SOLUTION)\n"
                  f"    \n"
                  f"    print(\"Generating visualization (this may take a moment)...\")\n"
                  f"    for i in range(resolution):\n"
                  f"        for j in range(resolution):\n"
                  f"            parameter = np.array([X[i, j], Y[i, j]])\n"
                  f"            {prefix}_evaluate(parameter, solution)\n"
                  f"            Z[i, j] = solution.copy()\n"
                  f"    \n"
                  f"    # Plot the decision boundaries\n"
                  f"    fig, axes = plt.subplots(1, {prefix.upper()}_N_SOLUTION, figsize=(15, 5))\n"
                  f"    \n"
                  f"    # Handle case with only one output (axes not being iterable)\n"
                  f"    if {prefix.upper()}_N_SOLUTION == 1:\n"
                  f"        axes = [axes]\n"
                  f"    \n"
                  f"    for i in range({prefix.upper()}_N_SOLUTION):\n"
                  f"        im = axes[i].imshow(Z[:, :, i], extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')\n"
                  f"        axes[i].set_title(f'Solution component {{{{i+1}}}}')\n"
                  f"        axes[i].set_xlabel('Parameter 1')\n"
                  f"        axes[i].set_ylabel('Parameter 2')\n"
                  f"        plt.colorbar(im, ax=axes[i])\n"
                  f"    \n"
                  f"    plt.tight_layout()\n"
                  f"    plt.show()\n"
    })
    
    # Performance summary function
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": f"def generate_performance_summary(metrics):\n"
                  f"    \"\"\"\n"
                  f"    Generate a performance summary table.\n"
                  f"    \n"
                  f"    Args:\n"
                  f"        metrics: Dictionary of performance metrics\n"
                  f"    \"\"\"\n"
                  f"    print(\"\\n======= Performance Summary =======\")\n"
                  f"    print(\"| Implementation | Execution Time (µs) | Throughput (KOPS) | Jitter (µs) |\")\n"
                  f"    print(\"|----------------|--------------------:|------------------:|------------:|\")\n"
                  f"    print(f\"| Standard       | {{metrics['std_exec_time']:18.2f}} | {{metrics['std_throughput']:16.2f}} | ±{{metrics['std_jitter']:9.2f}} |\")\n"
                  f"    print(f\"| Vectorized     | {{metrics['vec_exec_time']:18.2f}} | {{metrics['vec_throughput']:16.2f}} | ±{{metrics['vec_jitter']:9.2f}} |\")\n"
                  f"    print(f\"| **Speedup**    | {{metrics['speedup']:17.2f}}x | {{metrics['speedup']:15.2f}}x |             |\")\n"
                  f"    print(\"\\nNotes:\")\n"
                  f"    print(\"- KOPS = Thousands of samples processed per second\")\n"
                  f"    print(\"- Jitter = Standard deviation of execution times\")\n"
                  f"    print(\"- All measurements performed on CPU\")\n"
    })
    
    # Main execution cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": f"# Main execution block\n"
                  f"print(\"{prefix.upper()} Algorithm Benchmark\")\n"
                  f"print(\"==================================\")\n"
                  f"print(f\"Implementation details:\")\n"
                  f"print(f\"- Algorithm parameters: {{{prefix.upper()}_N_PARAMETER}} inputs, {{{prefix.upper()}_N_SOLUTION}} outputs\")\n"
                  f"print(f\"- Standard and vectorized implementations\")\n"
                  f"print(\"==================================\")\n"
                  f"\n"
                  f"# Run basic benchmarks\n"
                  f"std_time, vec_time = benchmark_and_compare()\n"
                  f"\n"
                  f"# Run extended benchmarks\n"
                  f"cpu_metrics = extended_benchmark()\n"
                  f"\n"
                  f"# Generate performance summary\n"
                  f"generate_performance_summary(cpu_metrics)\n"
    })
    
    # Visualization execution cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": f"# Generate visualization of decision boundaries\n"
                  f"if {prefix.upper()}_N_PARAMETER == 2:  # Only for 2D parameter space\n"
                  f"    print(\"\\nGenerating decision boundary visualization...\")\n"
                  f"    visualize_decision_boundaries(resolution=50)\n"
                  f"else:\n"
                  f"    print(\"\\nVisualization not available for parameter dimension > 2\")\n"
    })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def main():
    parser = argparse.ArgumentParser(description='Generate benchmark ipynb file from PDAQP C code')
    parser.add_argument('-c', '--c-file', required=True, help='Path to C file containing array definitions')
    parser.add_argument('-H', '--h-file', required=True, help='Path to H file containing parameter definitions')
    parser.add_argument('-o', '--output-dir', default='.', help='Output directory for generated files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)
    
    # Extract data from C file
    try:
        data = extract_data_from_c_file(args.c_file, args.h_file)
        print(f"Successfully extracted data from {args.c_file} and {args.h_file}")
        
        # Generate notebook
        notebook = generate_notebook(data)
        
        # Save notebook
        base_name = os.path.splitext(os.path.basename(args.c_file))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_benchmark.ipynb")
        
        with open(output_path, 'w') as f:
            json.dump(notebook, f, indent=2)
        
        print(f"Generated benchmark notebook saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()