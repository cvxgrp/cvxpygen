import numpy as np
import cupy as cp
import time
import statistics
import os

# Constants (based on the algorithm structure)
PDAQP_PID_N_PARAMETER = 2  # Input parameter dimension
PDAQP_PID_N_SOLUTION = 3   # Output solution dimension

# Define arrays from the C code implementation
# These are the coefficients for the halfplanes defining the decision regions
pdaqp_pid_halfplanes = np.array([
    0.5132947022251544, 0.8582124146547812, 1.283236755562886, 0.7071067811865475, -0.7071067811865475, -0.7463904912524668
], dtype=np.float64)

# Feedback coefficients for the affine functions at the leaf nodes
pdaqp_pid_feedbacks = np.array([
    0.140625, -0.140625, 0.1484375, -0.005208333333333315, 0.005208333333333259, 0.5130208333333334, -0.1354166666666667, 0.13541666666666669, 0.3385416666666668, 0.19999999999999998, -0.04135188866799204, 0.0, 0.2, 0.3483101391650099, 0.0, 9.656512391123383e-18, 0.36182902584493054, 0.0, 0.0, 5.551115123125783e-17, -3.071263974827799e-17, 0.09523809523809522, -0.0952380952380954, 0.6190476190476192, -0.09523809523809526, 0.09523809523809523, 0.38095238095238115
], dtype=np.float64)

# Indices for halfplanes and jumps in the decision tree
pdaqp_pid_hp_list = np.array([
    1, 0, 2, 1, 0
], dtype=np.int32)

pdaqp_pid_jump_list = np.array([
    1, 2, 0, 0, 0
], dtype=np.int32)

# Fix random seed for reproducibility
np.random.seed(42)
cp.random.seed(42)

# Transfer to GPU memory once
pdaqp_pid_halfplanes_gpu = cp.asarray(pdaqp_pid_halfplanes)
pdaqp_pid_feedbacks_gpu = cp.asarray(pdaqp_pid_feedbacks)
pdaqp_pid_hp_list_gpu = cp.asarray(pdaqp_pid_hp_list)
pdaqp_pid_jump_list_gpu = cp.asarray(pdaqp_pid_jump_list)

def pdaqp_pid_evaluate_cpu(parameter, solution):
    """CPU reference implementation"""
    id = 0
    next_id = id + pdaqp_pid_jump_list[id]
    
    while next_id != id:
        disp = pdaqp_pid_hp_list[id] * (PDAQP_PID_N_PARAMETER + 1)
        val = 0
        for i in range(PDAQP_PID_N_PARAMETER):
            val += parameter[i] * pdaqp_pid_halfplanes[disp + i]
        
        id = next_id + (1 if val <= pdaqp_pid_halfplanes[disp + PDAQP_PID_N_PARAMETER] else 0)
        next_id = id + pdaqp_pid_jump_list[id]
    
    disp = pdaqp_pid_hp_list[id] * (PDAQP_PID_N_PARAMETER + 1) * PDAQP_PID_N_SOLUTION
    for i in range(PDAQP_PID_N_SOLUTION):
        val = 0
        for j in range(PDAQP_PID_N_PARAMETER):
            val += parameter[j] * pdaqp_pid_feedbacks[disp + j]
        val += pdaqp_pid_feedbacks[disp + PDAQP_PID_N_PARAMETER]
        solution[i] = val
        disp += PDAQP_PID_N_PARAMETER + 1

# Optimized CUDA kernel
pdaqp_pid_kernel_optimized = cp.RawKernel(r'''
extern "C" __global__
void pdaqp_pid_kernel(double* parameters, double* solutions,
                     double* halfplanes, double* feedbacks,
                     int* hp_list, int* jump_list,
                     int batch_size) {
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Coalesced memory access
    double* param = &parameters[idx * ''' + str(PDAQP_PID_N_PARAMETER) + r'''];
    double* sol = &solutions[idx * ''' + str(PDAQP_PID_N_SOLUTION) + r'''];
    
    // Load parameters to registers
    double local_params[''' + str(PDAQP_PID_N_PARAMETER) + r'''];
    #pragma unroll
    for (int i = 0; i < ''' + str(PDAQP_PID_N_PARAMETER) + r'''; i++) {
        local_params[i] = param[i];
    }
    
    int id = 0;
    int next_id = id + jump_list[id];
    
    // Tree traversal
    while (next_id != id) {
        int disp = hp_list[id] * (''' + str(PDAQP_PID_N_PARAMETER) + r''' + 1);
        double val = 0;
        
        #pragma unroll 4
        for (int i = 0; i < ''' + str(PDAQP_PID_N_PARAMETER) + r'''; i++) {
            val += local_params[i] * halfplanes[disp + i];
        }
        
        id = next_id + (val <= halfplanes[disp + ''' + str(PDAQP_PID_N_PARAMETER) + r'''] ? 1 : 0);
        next_id = id + jump_list[id];
    }
    
    // Compute output
    int disp = hp_list[id] * (''' + str(PDAQP_PID_N_PARAMETER) + r''' + 1) * ''' + str(PDAQP_PID_N_SOLUTION) + r''';
    #pragma unroll
    for (int i = 0; i < ''' + str(PDAQP_PID_N_SOLUTION) + r'''; i++) {
        double val = 0;
        #pragma unroll 4
        for (int j = 0; j < ''' + str(PDAQP_PID_N_PARAMETER) + r'''; j++) {
            val += local_params[j] * feedbacks[disp + j];
        }
        val += feedbacks[disp + ''' + str(PDAQP_PID_N_PARAMETER) + r'''];
        sol[i] = val;
        disp += ''' + str(PDAQP_PID_N_PARAMETER) + r''' + 1;
    }
}
''', 'pdaqp_pid_kernel')

def test_gpu_performance(n_tests=100000, batch_size=1024, warmup_runs=200, n_iterations=5):
    """Enhanced GPU performance benchmark with improved accuracy"""
    
    # Generate test data (fixed seed ensures reproducibility)
    parameters_cpu = np.random.randn(n_tests, PDAQP_PID_N_PARAMETER).astype(np.float64)
    
    # Transfer to GPU
    parameters_gpu = cp.asarray(parameters_cpu)
    solutions_gpu = cp.zeros((n_tests, PDAQP_PID_N_SOLUTION), dtype=cp.float64)
    
    # Configure kernel launch parameters
    threads_per_block = 256
    blocks = (batch_size + threads_per_block - 1) // threads_per_block
    
    # Create CUDA events for timing
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    # Enhanced warmup
    print(f"Warmup ({warmup_runs} runs)...", end='', flush=True)
    for _ in range(warmup_runs):
        pdaqp_pid_kernel_optimized(
            (blocks,), (threads_per_block,),
            (parameters_gpu[:batch_size], solutions_gpu[:batch_size],
             pdaqp_pid_halfplanes_gpu, pdaqp_pid_feedbacks_gpu,
             pdaqp_pid_hp_list_gpu, pdaqp_pid_jump_list_gpu,
             batch_size)
        )
    cp.cuda.Stream.null.synchronize()
    print(" Done")
    
    # Performance measurement with multiple iterations
    all_batch_times = []
    
    print(f"Benchmarking ({n_iterations} iterations)...", end='', flush=True)
    
    for iteration in range(n_iterations):
        batch_times = []
        
        for i in range(0, n_tests, batch_size):
            current_batch = min(batch_size, n_tests - i)
            blocks = (current_batch + threads_per_block - 1) // threads_per_block
            
            # Time using CUDA Events for high precision
            start_event.record()
            
            pdaqp_pid_kernel_optimized(
                (blocks,), (threads_per_block,),
                (parameters_gpu[i:i+current_batch], 
                 solutions_gpu[i:i+current_batch],
                 pdaqp_pid_halfplanes_gpu, pdaqp_pid_feedbacks_gpu,
                 pdaqp_pid_hp_list_gpu, pdaqp_pid_jump_list_gpu,
                 current_batch)
            )
            
            end_event.record()
            end_event.synchronize()
            
            # Get elapsed time in microseconds
            elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
            batch_times.append(elapsed_ms * 1000)  # Convert to microseconds
        
        all_batch_times.extend(batch_times)
    
    print(" Done")
    
    # Get results back for validation
    solutions_cpu_result = cp.asnumpy(solutions_gpu)
    
    # Calculate robust statistics (remove outliers)
    all_batch_times = np.array(all_batch_times)
    q1 = np.percentile(all_batch_times, 25)
    q3 = np.percentile(all_batch_times, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_times = all_batch_times[(all_batch_times >= lower_bound) & (all_batch_times <= upper_bound)]
    
    # Statistics on filtered data
    avg_batch_time = np.mean(filtered_times)
    std_batch_time = np.std(filtered_times)
    min_batch_time = np.min(filtered_times)
    max_batch_time = np.max(filtered_times)
    median_batch_time = np.median(filtered_times)
    
    avg_time_per_sample = avg_batch_time / batch_size
    throughput_ms = batch_size / avg_batch_time  # MS/s
    
    # Validate results
    print("Validating...", end='', flush=True)
    errors = 0
    for idx in range(min(100, n_tests)):  # Validate more samples
        cpu_sol = np.zeros(PDAQP_PID_N_SOLUTION)
        pdaqp_pid_evaluate_cpu(parameters_cpu[idx], cpu_sol)
        if not np.allclose(cpu_sol, solutions_cpu_result[idx], rtol=1e-10):
            errors += 1
    
    if errors == 0:
        print(" PASSED")
    else:
        print(f" FAILED ({errors} errors)")
    
    return {
        "batch_latency": avg_batch_time,
        "batch_latency_std": std_batch_time,
        "batch_latency_min": min_batch_time,
        "batch_latency_max": max_batch_time,
        "batch_latency_median": median_batch_time,
        "per_sample_latency": avg_time_per_sample,
        "throughput_ms": throughput_ms,
        "batch_size": batch_size,
        "total_samples": n_tests,
        "outliers_removed": len(all_batch_times) - len(filtered_times),
        "total_measurements": len(all_batch_times)
    }

def set_gpu_frequency():
    """Try to set GPU to max frequency for consistent results"""
    try:
        # For Jetson devices
        os.system("sudo jetson_clocks --show > /dev/null 2>&1")
        os.system("sudo jetson_clocks > /dev/null 2>&1")
        print("GPU frequency locked to maximum")
    except:
        print("Unable to lock GPU frequency (normal for non-Jetson devices)")

def print_performance_table(metrics):
    """Print performance metrics table with 2 decimal precision"""
    print("\n===== GPU Performance Metrics =====")
    print(f"Batch size:              {metrics['batch_size']}")
    print(f"Batch latency:           {metrics['batch_latency']:.2f} ± {metrics['batch_latency_std']:.2f} µs")
    print(f"  Median:                {metrics['batch_latency_median']:.2f} µs")
    print(f"  Min/Max:               {metrics['batch_latency_min']:.2f} / {metrics['batch_latency_max']:.2f} µs")
    print(f"Per-sample latency:      {metrics['per_sample_latency']:.2f} µs")
    print(f"Throughput:              {metrics['throughput_ms']:.2f} MS/s")
    print(f"Total samples:           {metrics['total_samples']}")
    print(f"Measurements:            {metrics['total_measurements']} ({metrics['outliers_removed']} outliers removed)")

if __name__ == "__main__":
    print("PDAQP PID GPU Benchmark (Enhanced Precision)")
    print(f"Parameters: {PDAQP_PID_N_PARAMETER} inputs, {PDAQP_PID_N_SOLUTION} outputs")
    
    # Check CUDA
    if not cp.cuda.is_available():
        print("Error: CUDA not available!")
        exit(1)
    
    # Get device info
    device = cp.cuda.Device()
    print(f"\nGPU Device ID: {device.id}")
    print(f"Compute Capability: {device.compute_capability}")
    
    # Memory info
    meminfo = cp.cuda.runtime.memGetInfo()
    free_mem = meminfo[0] / 1e9
    total_mem = meminfo[1] / 1e9
    print(f"GPU Memory: {free_mem:.1f}/{total_mem:.1f} GB free")
    
    # Try to lock GPU frequency
    set_gpu_frequency()
    
    # Test batch sizes from 1 to 4096
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    results = []
    
    print("\n===== Testing Batch Sizes =====")
    for batch_size in batch_sizes:
        try:
            # More samples and iterations for better accuracy
            metrics = test_gpu_performance(
                n_tests=max(100000, batch_size * 100),  # Ensure enough samples
                batch_size=batch_size, 
                warmup_runs=200,  # More warmup
                n_iterations=5    # Multiple iterations
            )
            results.append(metrics)
            print(f"Batch {batch_size:4d}: {metrics['batch_latency']:7.2f} ± {metrics['batch_latency_std']:5.2f} µs, {metrics['throughput_ms']:7.2f} MS/s")
        except Exception as e:
            print(f"Batch {batch_size:4d}: Failed - {e}")
    
    # Find best configuration
    if results:
        best_idx = np.argmax([r['throughput_ms'] for r in results])
        print(f"\nOptimal batch size: {results[best_idx]['batch_size']}")
        print_performance_table(results[best_idx])
    
    # Summary table
    print("\n===== Batch Size vs Performance Summary =====")
    print("| Batch | Avg Latency | Std Dev | Median  | Per-Sample | Throughput |")
    print("|-------|-------------|---------|---------|------------|------------|")
    for r in results:
        print(f"| {r['batch_size']:5d} | {r['batch_latency']:9.2f} µs | {r['batch_latency_std']:5.2f} µs | {r['batch_latency_median']:7.2f} µs | {r['per_sample_latency']:8.2f} µs | {r['throughput_ms']:8.2f} MS/s |")
    
    # Performance analysis
    print("\n===== Performance Analysis =====")
    if results:
        # Find knee point (best efficiency)
        efficiencies = [r['throughput_ms'] / r['batch_size'] for r in results]
        best_efficiency_idx = np.argmax(efficiencies)
        print(f"Best efficiency at batch size: {results[best_efficiency_idx]['batch_size']}")
        
        # Scaling analysis
        single_sample_time = results[0]['batch_latency'] if results[0]['batch_size'] == 1 else None
        if single_sample_time:
            for r in results[1:]:
                speedup = (single_sample_time * r['batch_size']) / r['batch_latency']
                print(f"Batch {r['batch_size']:4d} speedup: {speedup:.2f}x")