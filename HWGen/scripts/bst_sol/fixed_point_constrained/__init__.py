# fixed_point_constrained/__init__.py

"""
Fixed-Point Constrained Optimization Module

Implements four DSP resource allocation strategies for computing
x* = F*θ + f on resource-constrained FPGAs:

1. Fully Parallel: D >= n*p (full parallelism)
2. Solution-TDM: p <= D < n*p (time-multiplex solutions)
3. Parameter-TDM: 0 < D < p (time-multiplex parameters)
4. Sequential SM: D = 0 or forced (pure sequential)

Key Components:
    - Strategy selector: Automatic strategy selection
    - Individual strategy implementations
    - Schedulers and resource estimators
    - Visualization and reporting tools
"""

from .strategy_selector import (
    DSPStrategy,
    StrategyConfig,
    select_strategy,
    print_strategy_report
)

from .solution_tdm import (
    SolutionTDMConfig,
    configure_solution_tdm,
    SolutionTDMScheduler,
    print_solution_tdm_report
)

from .parameter_tdm import (
    ParameterTDMConfig,
    configure_parameter_tdm,
    ParameterTDMScheduler,
    print_parameter_tdm_report
)

from .sequential_sm import (
    SequentialSMConfig,
    configure_sequential_sm,
    SequentialSMScheduler,
    print_sequential_sm_report
)


__all__ = [
    # Strategy selection
    'DSPStrategy',
    'StrategyConfig',
    'select_strategy',
    'print_strategy_report',
    
    # Solution-TDM
    'SolutionTDMConfig',
    'configure_solution_tdm',
    'SolutionTDMScheduler',
    'print_solution_tdm_report',
    
    # Parameter-TDM
    'ParameterTDMConfig',
    'configure_parameter_tdm',
    'ParameterTDMScheduler',
    'print_parameter_tdm_report',
    
    # Sequential SM
    'SequentialSMConfig',
    'configure_sequential_sm',
    'SequentialSMScheduler',
    'print_sequential_sm_report',
]


__version__ = '1.0.0'
__author__ = 'FPGA Architecture Design Team'
__description__ = 'DSP Resource Allocation Strategies for Fixed-Point Constrained Optimization'


def get_recommended_strategy(n: int, p: int, available_dsp: int) -> str:
    """
    Get recommended strategy with explanation
    
    Args:
        n: Number of variables
        p: Number of parameters
        available_dsp: Available DSP blocks
    
    Returns:
        Human-readable recommendation string
    """
    import math
    
    config = select_strategy(n, p, bst_depth=1, available_dsp=available_dsp)
    
    recommendations = {
        DSPStrategy.FULLY_PARALLEL: (
            f"Recommended: Fully Parallel\n"
            f"You have sufficient DSP resources ({available_dsp} >= {n*p}).\n"
            f"All {n} solutions can be computed in parallel in just 1 cycle.\n"
            f"This provides maximum throughput with minimal latency."
        ),
        DSPStrategy.SOLUTION_TDM: (
            f"Recommended: Solution-Level Time-Division Multiplexing\n"
            f"With {available_dsp} DSPs, you can process {config.group_size} solutions in parallel.\n"
            f"Total computation time: {config.t_solution} cycles across "
            f"{math.ceil(n / config.group_size)} time batches.\n"
            f"Good balance between parallelism and resource usage."
        ),
        DSPStrategy.PARAMETER_TDM: (
            f"Recommended: Parameter-Level Time-Division Multiplexing\n"
            f"With {available_dsp} DSPs (less than {p} parameters), parameters are processed in "
            f"{config.batch_count} batches.\n"
            f"Total computation time: {config.t_solution} cycles.\n"
            f"Optimized for limited DSP resources."
        ),
        DSPStrategy.SEQUENTIAL_SM: (
            f"Recommended: Sequential State Machine\n"
            f"With minimal DSP resources, all operations are performed sequentially.\n"
            f"Total computation time: {config.t_solution} cycles.\n"
            f"Minimum resource footprint, suitable for area-constrained designs."
        )
    }
    
    return recommendations.get(config.strategy, "Unknown strategy")


def compare_strategies(n: int, p: int, dsp_options: list) -> None:
    """
    Compare multiple DSP allocation options
    
    Args:
        n: Number of variables
        p: Number of parameters
        dsp_options: List of DSP counts to compare
    """
    import math
    
    print("=" * 80)
    print(f"Strategy Comparison for n={n}, p={p}")
    print("=" * 80)
    print(f"{'DSP Count':<12} {'Strategy':<20} {'Cycles':<10} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 80)
    
    baseline_cycles = None
    
    for dsp_count in sorted(dsp_options):
        config = select_strategy(n, p, bst_depth=1, available_dsp=dsp_count)
        
        if baseline_cycles is None:
            baseline_cycles = config.t_solution
        
        speedup = baseline_cycles / config.t_solution if config.t_solution > 0 else 0
        
        # Calculate DSP efficiency (useful work / total DSP-cycles)
        total_macs = n * p
        if dsp_count > 0 and config.t_solution > 0:
            efficiency = total_macs / (dsp_count * config.t_solution)
        else:
            efficiency = 0
        
        print(f"{dsp_count:<12} {config.strategy.name:<20} {config.t_solution:<10} "
              f"{speedup:<10.2f} {efficiency:<12.2%}")
    
    print("=" * 80)


def visualize_timeline(config: StrategyConfig, n: int, p: int, max_cycles: int = 50) -> None:
    """
    Visualize computation timeline for selected strategy
    
    Args:
        config: Strategy configuration
        n: Number of variables
        p: Number of parameters
        max_cycles: Maximum cycles to display
    """
    import math
    
    print("\n" + "=" * 80)
    print(f"Computation Timeline: {config.strategy.name}")
    print("=" * 80)
    
    cycles_to_show = min(config.t_solution, max_cycles)
    
    if config.strategy == DSPStrategy.FULLY_PARALLEL:
        print("Cycle 0: [All solutions computed in parallel]")
        print(f"         {' '.join([f'x{i}' for i in range(min(n, 10))])}", end='')
        if n > 10:
            print(f" ... x{n-1}")
        else:
            print()
        print("Cycle 1: [Results ready]")
    
    elif config.strategy == DSPStrategy.SOLUTION_TDM:
        G = config.group_size
        batches = math.ceil(n / G)
        for batch in range(min(batches, max_cycles // 3)):
            start_idx = batch * G
            end_idx = min(start_idx + G, n)
            cycle = batch * 3
            print(f"Cycles {cycle}-{cycle+2}: Solutions {start_idx} to {end_idx-1}")
    
    elif config.strategy == DSPStrategy.PARAMETER_TDM:
        B = config.batch_count
        print(f"Each solution requires {B} parameter batches × 3 cycles = {3*B} cycles")
        for sol in range(min(n, 3)):
            base = sol * 3 * B
            print(f"Solution {sol}: Cycles {base} to {base + 3*B - 1}")
        if n > 3:
            print(f"... (total {n} solutions)")
    
    elif config.strategy == DSPStrategy.SEQUENTIAL_SM:
        print(f"Processing {n} solutions × {p} parameters sequentially")
        print(f"Total {n * p} MAC operations")
        for i in range(min(5, n)):
            print(f"Solution {i}: MACs for all {p} parameters")
        if n > 5:
            print(f"... ({n - 5} more solutions)")
    
    print("=" * 80)


# Convenience function for quick analysis
def analyze_design_point(n: int, p: int, available_dsp: int, verbose: bool = True) -> StrategyConfig:
    """
    Perform complete analysis for a design point
    
    Args:
        n: Number of variables
        p: Number of parameters
        available_dsp: Available DSP blocks
        verbose: Print detailed report
    
    Returns:
        StrategyConfig with selected strategy
    """
    config = select_strategy(n, p, bst_depth=1, available_dsp=available_dsp)
    
    if verbose:
        print_strategy_report(config, n, p)
        print("\n" + get_recommended_strategy(n, p, available_dsp))
        visualize_timeline(config, n, p)
    
    return config