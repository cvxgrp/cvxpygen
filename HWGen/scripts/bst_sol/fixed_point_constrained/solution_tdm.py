# fixed_point_constrained/solution_tdm.py

"""
Strategy 2: Solution-Level Time-Division Multiplexing

Partitions D DSP blocks into G = floor(D/p) groups, where each group
processes one complete solution in parallel. Processes n solutions
across ceil(n/G) time batches.

Applicable when: p <= D < n*p
"""

from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path
import math


@dataclass
class SolutionTDMConfig:
    """Solution-Level TDM architecture configuration"""
    n: int                      # Number of variables
    p: int                      # Number of parameters
    dsp_count: int              # Available DSP blocks (D)
    group_size: int             # G = floor(D/p)
    num_batches: int            # ceil(n/G)
    cycles_per_batch: int       # 3 (DSP48 latency)
    total_cycles: int           # 3 * num_batches
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        assert self.p <= self.dsp_count < self.n * self.p, \
            f"Invalid DSP count {self.dsp_count} for Solution-TDM"
        assert self.group_size > 0, "Group size must be positive"
        assert self.num_batches > 0, "Number of batches must be positive"


def configure_solution_tdm(n: int, p: int, dsp_count: int) -> SolutionTDMConfig:
    """
    Configure Solution-Level Time-Division Multiplexing
    
    Strategy Overview:
        - Partition D DSPs into G = floor(D/p) groups
        - Each group computes one solution: x[i] = F[i,:] * θ + f[i]
        - Process G solutions in parallel per time batch
        - Require ceil(n/G) time batches to complete all solutions
    
    Timing:
        - Each batch takes 3 cycles (DSP48 pipeline depth)
        - Total cycles: 3 * ceil(n/G)
    
    Args:
        n: Number of variables
        p: Number of parameters  
        dsp_count: Available DSP blocks (D), must satisfy p <= D < n*p
    
    Returns:
        SolutionTDMConfig with resource allocation and timing
    
    Example:
        n=10, p=5, D=30
        G = 30//5 = 6 groups
        num_batches = ceil(10/6) = 2
        total_cycles = 3 * 2 = 6 cycles
    """
    
    # Validate input constraints
    assert n > 0 and p > 0, "n and p must be positive"
    assert p <= dsp_count < n * p, \
        f"Invalid DSP count {dsp_count} for Solution-TDM (need {p} <= D < {n*p})"
    
    # Calculate resource partitioning
    G = dsp_count // p                      # Groups processing in parallel
    num_batches = math.ceil(n / G)          # Time batches needed
    cycles_per_batch = 3                    # DSP48 pipeline latency
    total_cycles = cycles_per_batch * num_batches
    
    return SolutionTDMConfig(
        n=n,
        p=p,
        dsp_count=dsp_count,
        group_size=G,
        num_batches=num_batches,
        cycles_per_batch=cycles_per_batch,
        total_cycles=total_cycles
    )


class SolutionTDMScheduler:
    """
    Scheduler for Solution-Level Time-Division
    
    Implements batched computation: x* = F*θ + f
    where F is (n x p) and we process G rows at a time.
    
    Timeline for each batch (3 cycles):
        Cycle 0: Load F rows and θ
        Cycle 1: DSP multiply (internal pipeline)
        Cycle 2: Accumulate and add f offset
    """
    
    def __init__(self, config: SolutionTDMConfig):
        """
        Initialize scheduler
        
        Args:
            config: Solution-TDM configuration
        """
        self.config = config
        self.current_batch = 0
    
    def get_batch_schedule(self, batch_idx: int) -> Tuple[int, int]:
        """
        Get solution row range for batch_idx
        
        Args:
            batch_idx: Batch index (0 to num_batches-1)
        
        Returns:
            (start_row, end_row): Row indices to process in this batch
            
        Example:
            n=10, G=6
            batch 0: (0, 6)
            batch 1: (6, 10)
        """
        assert 0 <= batch_idx < self.config.num_batches, \
            f"Invalid batch index {batch_idx}"
        
        G = self.config.group_size
        start_row = batch_idx * G
        end_row = min(start_row + G, self.config.n)
        return start_row, end_row
    
    def get_active_groups(self, batch_idx: int) -> int:
        """
        Get number of active groups in batch_idx
        
        Last batch may have fewer than G active groups
        
        Args:
            batch_idx: Batch index
            
        Returns:
            Number of active solution groups
        """
        start_row, end_row = self.get_batch_schedule(batch_idx)
        return end_row - start_row
    
    def generate_control_signals(self) -> List[dict]:
        """
        Generate cycle-by-cycle control signals
        
        Returns:
            List of control signal dictionaries per cycle
            
        Control Signal Fields:
            - cycle: Global cycle number
            - operation: 'load', 'multiply', or 'accumulate_add'
            - f_row_start, f_row_end: F matrix row range
            - active_rows: Number of solutions being processed
            - output_valid: Whether result is valid this cycle
        """
        signals = []
        
        for batch_idx in range(self.config.num_batches):
            start_row, end_row = self.get_batch_schedule(batch_idx)
            active_rows = end_row - start_row
            base_cycle = batch_idx * 3
            
            # Cycle 0: Load F rows and theta vector
            signals.append({
                'cycle': base_cycle,
                'operation': 'load',
                'batch_idx': batch_idx,
                'f_row_start': start_row,
                'f_row_end': end_row,
                'active_rows': active_rows,
                'theta_load': True,
                'output_valid': False
            })
            
            # Cycle 1: DSP multiply (internal pipeline stage)
            signals.append({
                'cycle': base_cycle + 1,
                'operation': 'multiply',
                'stage': 'dsp_pipeline',
                'batch_idx': batch_idx,
                'output_valid': False
            })
            
            # Cycle 2: Accumulate partial products and add f offset
            signals.append({
                'cycle': base_cycle + 2,
                'operation': 'accumulate_add',
                'batch_idx': batch_idx,
                'f_offset_start': start_row,
                'f_offset_end': end_row,
                'result_idx_start': start_row,
                'result_idx_end': end_row,
                'output_valid': True
            })
        
        return signals
    
    def generate_memory_access_pattern(self) -> dict:
        """
        Generate memory access pattern for efficient memory banking
        
        Returns:
            Dictionary with memory access patterns for F, theta, and f
        """
        access_pattern = {
            'F_matrix': [],
            'theta_vector': [],
            'f_offset': []
        }
        
        for batch_idx in range(self.config.num_batches):
            start_row, end_row = self.get_batch_schedule(batch_idx)
            
            # F matrix: Read rows [start_row:end_row, :]
            for row_idx in range(start_row, end_row):
                for col_idx in range(self.config.p):
                    access_pattern['F_matrix'].append({
                        'batch': batch_idx,
                        'cycle': batch_idx * 3,
                        'row': row_idx,
                        'col': col_idx,
                        'address': row_idx * self.config.p + col_idx
                    })
            
            # Theta vector: Read all p elements (broadcast to all groups)
            for col_idx in range(self.config.p):
                access_pattern['theta_vector'].append({
                    'batch': batch_idx,
                    'cycle': batch_idx * 3,
                    'idx': col_idx
                })
            
            # f offset: Read elements [start_row:end_row]
            for idx in range(start_row, end_row):
                access_pattern['f_offset'].append({
                    'batch': batch_idx,
                    'cycle': batch_idx * 3 + 2,
                    'idx': idx
                })
        
        return access_pattern
    
    def estimate_resources(self) -> dict:
        """
        Estimate FPGA resource usage
        
        Returns:
            Dictionary with estimated resource counts
        """
        return {
            'dsp_blocks': self.config.dsp_count,
            'dsp_utilization': self.config.dsp_count,
            'adder_trees': self.config.group_size,  # One per group
            'f_matrix_banks': min(self.config.group_size, 4),  # Memory banking
            'theta_broadcast': 1,
            'registers_per_group': self.config.p + 2,  # Accumulator + control
            'total_registers': self.config.group_size * (self.config.p + 2)
        }
    
    def get_latency_breakdown(self) -> dict:
        """
        Get detailed latency breakdown
        
        Returns:
            Dictionary with latency components
        """
        return {
            'cycles_per_batch': self.config.cycles_per_batch,
            'num_batches': self.config.num_batches,
            'solutions_per_batch': self.config.group_size,
            'total_cycles': self.config.total_cycles,
            'dsp_pipeline_depth': 3,
            'accumulator_depth': 1
        }


def print_solution_tdm_report(config: SolutionTDMConfig) -> None:
    """
    Print detailed Solution-TDM configuration report
    
    Args:
        config: Solution-TDM configuration
    """
    print("=" * 60)
    print("Solution-Level Time-Division Multiplexing Configuration")
    print("=" * 60)
    print(f"Problem Size: n={config.n}, p={config.p}")
    print(f"Available DSP: {config.dsp_count}")
    print(f"\nResource Partitioning:")
    print(f"  - Group Size (G): {config.group_size}")
    print(f"  - Solutions per Batch: {config.group_size}")
    print(f"  - DSP per Group: {config.p}")
    print(f"\nTiming:")
    print(f"  - Number of Batches: {config.num_batches}")
    print(f"  - Cycles per Batch: {config.cycles_per_batch}")
    print(f"  - Total Cycles: {config.total_cycles}")
    print(f"\nEfficiency:")
    total_macs = config.n * config.p
    dsp_efficiency = total_macs / (config.dsp_count * config.total_cycles)
    print(f"  - Total MACs: {total_macs}")
    print(f"  - DSP Efficiency: {dsp_efficiency:.2%}")
    print(f"  - Throughput: {config.n / config.total_cycles:.4f} solutions/cycle")
    print(f"  - Latency: {config.total_cycles} cycles for {config.n} solutions")
    print("=" * 60)


def compare_with_full_parallel(sol_config: SolutionTDMConfig) -> None:
    """
    Compare Solution-TDM with hypothetical Full-Parallel
    
    Args:
        sol_config: Solution-TDM configuration
    """
    print("\n" + "=" * 60)
    print("Strategy Comparison: Solution-TDM vs Full-Parallel")
    print("=" * 60)
    
    # Solution-TDM stats
    print(f"\nSolution-TDM (Current, D={sol_config.dsp_count}):")
    print(f"  - Total Cycles: {sol_config.total_cycles}")
    print(f"  - Solutions per Batch: {sol_config.group_size}")
    print(f"  - Number of Batches: {sol_config.num_batches}")
    
    # Hypothetical Full-Parallel (if we had D >= n*p)
    required_dsps = sol_config.n * sol_config.p
    if sol_config.dsp_count >= required_dsps:
        print(f"\nFull-Parallel (Hypothetical, D>={required_dsps}):")
        print(f"  - Total Cycles: 3")
        print(f"  - Speedup: {sol_config.total_cycles / 3:.2f}x")
    else:
        print(f"\nFull-Parallel: Not applicable (D < n*p = {required_dsps})")
        print(f"  - Would require: {required_dsps} DSP blocks")
        print(f"  - Current DSP: {sol_config.dsp_count}")
    
    print("=" * 60)


# ============================================================================
# Main Entry Point for Strategy Selector
# ============================================================================

def generate_solution_tdm_solver(
    config_file: Path,
    output_dir: Path,
    dsp_count: int = None,
    verbose: bool = False
) -> int:
    """
    Generate Solution-Level TDM BST solver hardware
    
    This is the main entry point called by strategy_selector.py
    
    Args:
        config_file: Path to Verilog config file
        output_dir: Output directory for RTL
        dsp_count: Number of DSP blocks (D), auto-calculated if None
        verbose: Enable verbose output
    
    Returns:
        0 on success, 1 on failure
    
    Constraints:
        - Requires: p <= D < n*p
        - Strategy: Partition DSPs into G=floor(D/p) groups
        - Timing: 3*ceil(n/G) total cycles
    """
    try:
        import sys
        
        # Add project root to Python path
        SCRIPT_DIR = Path(__file__).parent
        PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        
        from scripts.bst_sol.common.config_parser import VerilogConfigParser
        from scripts.bst_sol.fixed_point_constrained.solution_tdm_generator import SolutionTDMGenerator
        
        # Parse configuration file
        parser = VerilogConfigParser(str(config_file))
        cfg = parser.get_config()
        
        n = cfg.n_solutions
        p = cfg.n_parameters
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Solution-Level TDM Solver Generation")
            print(f"{'='*60}")
            print(f"Input Config: {config_file}")
            print(f"Problem Size: n={n}, p={p}")
        
        # Calculate DSP count if not provided
        if dsp_count is None:
            # Default strategy: use 2*p DSP blocks for 2 parallel solutions
            dsp_count = min(n * p - 1, 2 * p)
            if verbose:
                print(f"Auto DSP Count: {dsp_count} (2 groups of {p} DSPs)")
        
        # Validate DSP count for Solution-TDM
        if not (p <= dsp_count < n * p):
            print(f"❌ Invalid DSP count {dsp_count} for Solution-TDM")
            print(f"   Requirement: {p} <= D < {n * p}")
            print(f"   Suggested range: [{p}, {n * p - 1}]")
            return 1
        
        # Configure Solution-TDM architecture
        tdm_config = configure_solution_tdm(n, p, dsp_count)
        
        # Print detailed report
        if verbose:
            print_solution_tdm_report(tdm_config)
        
        # Generate RTL using SolutionTDMGenerator
        generator = SolutionTDMGenerator(tdm_config, cfg)
        success = generator.generate_rtl(output_dir)
        
        if success:
            print(f"✓ Solution-TDM RTL generated successfully")
            print(f"  Output Directory: {output_dir}")
            print(f"  DSP Blocks: {dsp_count}")
            print(f"  Parallel Groups: {tdm_config.group_size}")
            print(f"  Total Cycles: {tdm_config.total_cycles}")
            return 0
        else:
            print(f"❌ Solution-TDM RTL generation failed")
            return 1
        
    except Exception as e:
        print(f"❌ Solution-TDM generation error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


# ============================================================================
# Command-line Interface (for standalone testing)
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Solution-Level TDM BST Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto DSP count (2 parallel groups)
  python solution_tdm.py config.v output/

  # Manual DSP count for 3 parallel groups
  python solution_tdm.py config.v output/ --dsp-count 15

  # Verbose mode
  python solution_tdm.py config.v output/ -v
        """
    )
    
    parser.add_argument('config_file', type=Path,
                        help='Path to Verilog config file')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for RTL')
    parser.add_argument('--dsp-count', type=int, default=None,
                        help='Number of DSP blocks (auto if not specified)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate solver
    exit_code = generate_solution_tdm_solver(
        args.config_file,
        args.output_dir,
        args.dsp_count,
        args.verbose
    )
    
    exit(exit_code)