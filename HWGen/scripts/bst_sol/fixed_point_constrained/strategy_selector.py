"""
Algorithm 1: Adaptive DSP Resource Allocation Strategy Selector

Automatically selects the optimal architecture based on available DSP resources.
Can be used as:
1. Standalone analysis tool (--analyze mode)
2. Hardware generator entry point (default mode)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import math
import sys

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent
BST_SOL_DIR = SCRIPT_DIR.parent
SCRIPTS_DIR = BST_SOL_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))


class DSPStrategy(Enum):
    """DSP allocation strategies"""
    FULLY_PARALLEL = 1     # D >= n*p (fully parallel)
    SOLUTION_TDM = 2       # p <= D < n*p
    PARAMETER_TDM = 3      # 0 < D < p
    SEQUENTIAL_SM = 4      # D = 0 or forced


@dataclass
class StrategyConfig:
    """Strategy selection result with timing characteristics"""
    strategy: DSPStrategy
    dsp_allocated: int
    t_solution: int
    pipeline_depth: int
    
    # Strategy-specific parameters
    group_size: int = None
    batch_count: int = None
    multipliers: int = 1


def select_strategy(
    n: int,
    p: int,
    bst_depth: int,
    available_dsp: int,
    force_sequential: bool = False,
    multipliers: int = 1
) -> StrategyConfig:
    """
    Algorithm 1: Adaptive DSP Resource Allocation
    
    Args:
        n: Number of QP variables (solutions)
        p: Number of QP parameters
        bst_depth: Binary search tree depth
        available_dsp: Available DSP48 blocks (D)
        force_sequential: Force sequential mode
        multipliers: Multiplier count for sequential mode (M)
    
    Returns:
        StrategyConfig with selected strategy and timing
    """
    
    R_required = n * p
    
    assert n > 0 and p > 0, "n and p must be positive"
    assert bst_depth > 0, "BST depth must be positive"
    assert available_dsp >= 0, "Available DSP cannot be negative"
    assert multipliers > 0, "Multipliers must be positive"
    
    # Case 4: Sequential State Machine
    if force_sequential or available_dsp == 0:
        t_solution = math.ceil(n * p / multipliers)
        return StrategyConfig(
            strategy=DSPStrategy.SEQUENTIAL_SM,
            dsp_allocated=0,
            t_solution=t_solution,
            pipeline_depth=bst_depth + t_solution + 1,
            multipliers=multipliers
        )
    
    # Case 1: Fully Parallel
    if available_dsp >= R_required:
        return StrategyConfig(
            strategy=DSPStrategy.FULLY_PARALLEL,
            dsp_allocated=R_required,
            t_solution=1,
            pipeline_depth=bst_depth + 2
        )
    
    # Case 2: Solution-Level TDM
    if p <= available_dsp < R_required:
        G = available_dsp // p
        t_solution = 3 * math.ceil(n / G)
        return StrategyConfig(
            strategy=DSPStrategy.SOLUTION_TDM,
            dsp_allocated=available_dsp,
            t_solution=t_solution,
            pipeline_depth=bst_depth + t_solution + 1,
            group_size=G
        )
    
    # Case 3: Parameter-Level TDM
    else:  # 0 < available_dsp < p
        B = math.ceil(p / available_dsp)
        t_solution = 3 * n * B
        return StrategyConfig(
            strategy=DSPStrategy.PARAMETER_TDM,
            dsp_allocated=available_dsp,
            t_solution=t_solution,
            pipeline_depth=bst_depth + t_solution + 1,
            batch_count=B
        )


def print_strategy_report(config: StrategyConfig, n: int, p: int) -> None:
    """Print strategy selection report"""
    print("=" * 60)
    print(f"Strategy: {config.strategy.name}")
    print(f"Problem: n={n}, p={p} | DSP: {config.dsp_allocated}")
    print(f"Timing: t_solution={config.t_solution}, depth={config.pipeline_depth}")
    
    if config.group_size:
        print(f"Solution-TDM: group_size={config.group_size}")
    elif config.batch_count:
        print(f"Parameter-TDM: batches={config.batch_count}")
    elif config.multipliers > 1:
        print(f"Sequential: multipliers={config.multipliers}")
    
    print("=" * 60)


# ============================================================================
# Hardware Generation
# ============================================================================

def generate_hardware(
    config_file: Path,
    output_dir: Path,
    available_dsp: int = None,
    force_sequential: bool = False,
    multipliers: int = 1,
    verbose: bool = False
) -> int:
    """
    Generate hardware with adaptive DSP allocation
    
    Returns: 0 on success, 1 on failure
    """
    try:
        # Import common parser
        from scripts.bst_sol.common.config_parser import VerilogConfigParser
        
        # Parse configuration
        parser = VerilogConfigParser(str(config_file))
        cfg = parser.get_config()
        
        n = cfg.n_solutions
        p = cfg.n_parameters
        bst_depth = cfg.estimated_bst_depth
        
        if verbose:
            print(f"Problem: n={n}, p={p}, depth={bst_depth}")
        
        # Default: unlimited DSP
        if available_dsp is None:
            available_dsp = n * p
        
        # Select strategy
        strategy_config = select_strategy(
            n=n,
            p=p,
            bst_depth=bst_depth,
            available_dsp=available_dsp,
            force_sequential=force_sequential,
            multipliers=multipliers
        )
        
        print_strategy_report(strategy_config, n, p)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate architecture based on strategy
        if strategy_config.strategy == DSPStrategy.FULLY_PARALLEL:
            # Use standard fixed-point generator for full parallelism
            from scripts.bst_sol.fixed_point.generate_fixed import generate_bst_solver
            result = generate_bst_solver(
                config_file=str(config_file),
                output_dir=str(output_dir),
                verbose=verbose
            )
        
        elif strategy_config.strategy == DSPStrategy.SOLUTION_TDM:
            from scripts.bst_sol.fixed_point_constrained.solution_tdm import generate_solution_tdm_solver
            result = generate_solution_tdm_solver(
                config_file=config_file,
                output_dir=output_dir,
                dsp_count=strategy_config.dsp_allocated,
                verbose=verbose
            )
        
        elif strategy_config.strategy == DSPStrategy.PARAMETER_TDM:
            from scripts.bst_sol.fixed_point_constrained.parameter_tdm import generate_parameter_tdm_solver
            result = generate_parameter_tdm_solver(
                config_file=config_file,
                output_dir=output_dir,
                dsp_count=strategy_config.dsp_allocated,
                verbose=verbose
            )
        
        elif strategy_config.strategy == DSPStrategy.SEQUENTIAL_SM:
            from scripts.bst_sol.fixed_point_constrained.sequential_sm import generate_sequential_sm_solver
            result = generate_sequential_sm_solver(
                config_file=config_file,
                output_dir=output_dir,
                verbose=verbose
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_config.strategy}")
        
        if result != 0:
            return 1
        
        # Update timing file
        timing_file = output_dir.parent / 'include' / 'pdaqp_timing.vh'
        _update_timing_file(timing_file, strategy_config)
        
        print(f"✓ Generated {strategy_config.strategy.name}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def _update_timing_file(timing_file: Path, config: StrategyConfig):
    """Update timing.vh with actual strategy timing"""
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(timing_file, 'w') as f:
        f.write(f"// {config.strategy.name} | DSP: {config.dsp_allocated}\n")
        f.write(f"`define PDAQP_T_SOLUTION {config.t_solution}\n")
        f.write(f"`define PDAQP_PIPELINE_DEPTH {config.pipeline_depth}\n")
        
        if config.group_size is not None:
            f.write(f"`define PDAQP_SOLUTION_GROUP_SIZE {config.group_size}\n")
        if config.batch_count is not None:
            f.write(f"`define PDAQP_PARAMETER_BATCHES {config.batch_count}\n")
        if config.multipliers is not None:
            f.write(f"`define PDAQP_MULTIPLIERS {config.multipliers}\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description='DSP Strategy Selector & Hardware Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy_selector.py config.vh -o rtl --dsp 50
  python strategy_selector.py --analyze -n 10 -p 20 --dsp 50
  python strategy_selector.py config.vh -o rtl --sequential
        """
    )
    
    parser.add_argument('config_file', type=Path, nargs='?')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('-n', '--variables', type=int)
    parser.add_argument('-p', '--parameters', type=int)
    parser.add_argument('--bst-depth', type=int, default=10)
    parser.add_argument('--dsp', type=int, default=None)
    parser.add_argument('--sequential', action='store_true')
    parser.add_argument('--multipliers', type=int, default=1)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if args.analyze:
        if args.variables is None or args.parameters is None:
            parser.error("Analysis mode requires -n and -p")
        
        n = args.variables
        p = args.parameters
        available_dsp = args.dsp if args.dsp else n * p
        
        config = select_strategy(
            n=n, p=p, bst_depth=args.bst_depth,
            available_dsp=available_dsp,
            force_sequential=args.sequential,
            multipliers=args.multipliers
        )
        
        if args.json:
            output = {
                'problem': {'n': n, 'p': p},
                'strategy': config.strategy.name,
                'dsp': config.dsp_allocated,
                't_solution': config.t_solution,
                'pipeline_depth': config.pipeline_depth
            }
            print(json.dumps(output, indent=2))
        else:
            print_strategy_report(config, n, p)
        
        return 0
    
    else:
        if args.config_file is None or args.output is None:
            parser.error("Generation mode requires config_file and -o")
        
        return generate_hardware(
            config_file=args.config_file,
            output_dir=args.output,
            available_dsp=args.dsp,
            force_sequential=args.sequential,
            multipliers=args.multipliers,
            verbose=args.verbose
        )


if __name__ == '__main__':
    sys.exit(main())