# fixed_point_constrained/sequential_sm.py

"""
Strategy 4: Sequential State Machine

Processes one multiply-accumulate (MAC) operation per cycle using a single
DSP block. Computes all n solutions sequentially with full parameter sweep.

Applicable when: D = 1 (single DSP block available)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import math


@dataclass
class SequentialSMConfig:
    """Sequential State Machine architecture configuration"""
    n: int                      # Number of variables
    p: int                      # Number of parameters
    dsp_count: int              # Available DSP blocks (D = 1)
    macs_per_solution: int      # p MACs per solution
    cycles_per_mac: int         # 3 (DSP48 pipeline depth)
    cycles_per_solution: int    # 3 * p + 1 (MACs + offset add)
    total_cycles: int           # n * (3*p + 1)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        assert self.dsp_count == 1, \
            f"Sequential SM requires exactly 1 DSP, got {self.dsp_count}"
        assert self.macs_per_solution == self.p, "Invalid MAC count"
        assert self.cycles_per_solution == 3 * self.p + 1, "Invalid cycle count"


def configure_sequential_sm(n: int, p: int) -> SequentialSMConfig:
    """
    Configure Sequential State Machine
    
    Strategy Overview:
        - Single DSP block processes one MAC per 3 cycles
        - For each solution x[i]:
          * Perform p MACs: F[i,0]*theta[0] + ... + F[i,p-1]*theta[p-1]
          * Accumulate in register
          * Add offset f[i] in final cycle
        - Process all n solutions sequentially
    
    Timing:
        - Each MAC takes 3 cycles (DSP48 pipeline)
        - Each solution: p MACs + 1 offset add = 3*p + 1 cycles
        - Total: n * (3*p + 1) cycles
    
    State Machine States:
        IDLE -> LOAD_PARAMS -> MAC_0 -> MAC_1 -> ... -> MAC_{p-1} -> 
        ADD_OFFSET -> OUTPUT -> (next solution)
    
    Args:
        n: Number of variables
        p: Number of parameters
    
    Returns:
        SequentialSMConfig with state machine timing
    
    Example:
        n=10, p=5
        cycles_per_solution = 3*5 + 1 = 16
        total_cycles = 10 * 16 = 160 cycles
    """
    
    # Validate input constraints
    assert n > 0 and p > 0, "n and p must be positive"
    
    # Calculate timing
    dsp_count = 1
    macs_per_solution = p
    cycles_per_mac = 3                      # DSP48 pipeline depth
    cycles_per_solution = cycles_per_mac * p + 1  # +1 for offset add
    total_cycles = n * cycles_per_solution
    
    return SequentialSMConfig(
        n=n,
        p=p,
        dsp_count=dsp_count,
        macs_per_solution=macs_per_solution,
        cycles_per_mac=cycles_per_mac,
        cycles_per_solution=cycles_per_solution,
        total_cycles=total_cycles
    )


class SequentialSMScheduler:
    """
    Sequential State Machine Scheduler
    
    Implements FSM-based sequential computation:
        x[i] = (sum_{j=0}^{p-1} F[i,j] * theta[j]) + f[i]
    
    FSM State Transitions:
        IDLE: Waiting for start
        LOAD: Load F[i,j] and theta[j]
        MAC_STAGE1: DSP multiply (cycle 1)
        MAC_STAGE2: DSP multiply (cycle 2)  
        MAC_STAGE3: Accumulate product (cycle 3)
        ADD_OFFSET: Add f[i] to accumulator
        OUTPUT: Write x[i] to output
    """
    
    # State definitions
    STATE_IDLE = 'IDLE'
    STATE_LOAD = 'LOAD'
    STATE_MAC_S1 = 'MAC_STAGE1'
    STATE_MAC_S2 = 'MAC_STAGE2'
    STATE_MAC_S3 = 'MAC_STAGE3'
    STATE_ADD_OFFSET = 'ADD_OFFSET'
    STATE_OUTPUT = 'OUTPUT'
    
    def __init__(self, config: SequentialSMConfig):
        """
        Initialize sequential state machine scheduler
        
        Args:
            config: Sequential SM configuration
        """
        self.config = config
        self.current_state = self.STATE_IDLE
        self.current_solution = 0
        self.current_param = 0
        self.cycle_counter = 0
    
    def get_state_at_cycle(self, cycle: int) -> Tuple[str, int, int]:
        """
        Get FSM state at given cycle
        
        Args:
            cycle: Global cycle number
        
        Returns:
            (state, solution_idx, param_idx): Current state and indices
        """
        if cycle >= self.config.total_cycles:
            return (self.STATE_IDLE, -1, -1)
        
        # Determine which solution we're processing
        sol_idx = cycle // self.config.cycles_per_solution
        cycle_in_solution = cycle % self.config.cycles_per_solution
        
        # Determine state within solution
        if cycle_in_solution < self.config.p * 3:
            # MAC operations
            mac_idx = cycle_in_solution // 3
            mac_stage = cycle_in_solution % 3
            
            if mac_stage == 0:
                state = self.STATE_LOAD
            elif mac_stage == 1:
                state = self.STATE_MAC_S1
            else:  # mac_stage == 2
                state = self.STATE_MAC_S3
            
            return (state, sol_idx, mac_idx)
        else:
            # Offset addition and output
            return (self.STATE_ADD_OFFSET, sol_idx, -1)
    
    def generate_fsm_transitions(self) -> List[dict]:
        """
        Generate complete FSM state transition table
        
        Returns:
            List of state transitions with control signals
        """
        transitions = []
        
        # Initial idle state
        transitions.append({
            'cycle': -1,
            'current_state': self.STATE_IDLE,
            'next_state': self.STATE_LOAD,
            'condition': 'start_signal',
            'solution_idx': 0,
            'param_idx': 0
        })
        
        for sol_idx in range(self.config.n):
            for param_idx in range(self.config.p):
                base_cycle = sol_idx * self.config.cycles_per_solution + param_idx * 3
                
                # LOAD -> MAC_S1
                transitions.append({
                    'cycle': base_cycle,
                    'current_state': self.STATE_LOAD,
                    'next_state': self.STATE_MAC_S1,
                    'solution_idx': sol_idx,
                    'param_idx': param_idx,
                    'action': f'load F[{sol_idx},{param_idx}] and theta[{param_idx}]'
                })
                
                # MAC_S1 -> MAC_S2
                transitions.append({
                    'cycle': base_cycle + 1,
                    'current_state': self.STATE_MAC_S1,
                    'next_state': self.STATE_MAC_S2,
                    'solution_idx': sol_idx,
                    'param_idx': param_idx,
                    'action': 'DSP pipeline stage 1'
                })
                
                # MAC_S2 -> MAC_S3 (or LOAD for next param)
                is_last_param = (param_idx == self.config.p - 1)
                next_state = self.STATE_ADD_OFFSET if is_last_param else self.STATE_LOAD
                
                transitions.append({
                    'cycle': base_cycle + 2,
                    'current_state': self.STATE_MAC_S2,
                    'next_state': next_state,
                    'solution_idx': sol_idx,
                    'param_idx': param_idx,
                    'action': 'accumulate product',
                    'accumulator_update': True,
                    'accumulator_clear': (param_idx == 0)
                })
            
            # ADD_OFFSET -> OUTPUT (or LOAD for next solution)
            offset_cycle = sol_idx * self.config.cycles_per_solution + self.config.p * 3
            is_last_solution = (sol_idx == self.config.n - 1)
            next_state = self.STATE_IDLE if is_last_solution else self.STATE_LOAD
            
            transitions.append({
                'cycle': offset_cycle,
                'current_state': self.STATE_ADD_OFFSET,
                'next_state': next_state,
                'solution_idx': sol_idx,
                'param_idx': -1,
                'action': f'add f[{sol_idx}] and output x[{sol_idx}]',
                'output_valid': True
            })
        
        return transitions
    
    def generate_control_signals(self) -> List[dict]:
        """
        Generate cycle-by-cycle control signals
        
        Returns:
            List of control signal dictionaries per cycle
        """
        signals = []
        
        for cycle in range(self.config.total_cycles):
            state, sol_idx, param_idx = self.get_state_at_cycle(cycle)
            
            # Base signal structure
            signal = {
                'cycle': cycle,
                'state': state,
                'solution_idx': sol_idx,
                'param_idx': param_idx
            }
            
            # State-specific signals
            if state == self.STATE_LOAD:
                signal.update({
                    'f_read_enable': True,
                    'f_address': sol_idx * self.config.p + param_idx,
                    'theta_read_enable': True,
                    'theta_address': param_idx,
                    'dsp_load': True
                })
            
            elif state == self.STATE_MAC_S1:
                signal.update({
                    'dsp_multiply': True,
                    'pipeline_stage': 1
                })
            
            elif state == self.STATE_MAC_S3:
                signal.update({
                    'dsp_multiply': True,
                    'pipeline_stage': 2,
                    'accumulator_enable': True,
                    'accumulator_clear': (param_idx == 0)
                })
            
            elif state == self.STATE_ADD_OFFSET:
                signal.update({
                    'f_offset_read_enable': True,
                    'f_offset_address': sol_idx,
                    'adder_enable': True,
                    'output_enable': True,
                    'output_address': sol_idx
                })
            
            signals.append(signal)
        
        return signals
    
    def generate_memory_access_pattern(self) -> dict:
        """
        Generate memory access pattern
        
        Returns:
            Dictionary with sequential memory access patterns
        """
        access_pattern = {
            'F_matrix': [],
            'theta_vector': [],
            'f_offset': []
        }
        
        for sol_idx in range(self.config.n):
            base_cycle = sol_idx * self.config.cycles_per_solution
            
            # F matrix and theta: One access per parameter
            for param_idx in range(self.config.p):
                cycle = base_cycle + param_idx * 3
                
                access_pattern['F_matrix'].append({
                    'cycle': cycle,
                    'solution': sol_idx,
                    'param': param_idx,
                    'address': sol_idx * self.config.p + param_idx,
                    'row': sol_idx,
                    'col': param_idx
                })
                
                access_pattern['theta_vector'].append({
                    'cycle': cycle,
                    'solution': sol_idx,
                    'param': param_idx,
                    'address': param_idx
                })
            
            # f offset: One access per solution
            offset_cycle = base_cycle + self.config.p * 3
            access_pattern['f_offset'].append({
                'cycle': offset_cycle,
                'solution': sol_idx,
                'address': sol_idx
            })
        
        return access_pattern
    
    def generate_register_map(self) -> dict:
        """
        Generate register allocation map
        
        Returns:
            Dictionary describing register usage
        """
        return {
            'accumulator': {
                'width': 32,
                'description': 'Accumulates partial products for current solution',
                'reset_condition': 'Start of new solution (param_idx == 0)'
            },
            'f_register': {
                'width': 16,
                'description': 'Holds current F[i,j] value'
            },
            'theta_register': {
                'width': 16,
                'description': 'Holds current theta[j] value'
            },
            'solution_counter': {
                'width': math.ceil(math.log2(self.config.n + 1)),
                'description': 'Tracks current solution index',
                'range': f'0 to {self.config.n - 1}'
            },
            'param_counter': {
                'width': math.ceil(math.log2(self.config.p + 1)),
                'description': 'Tracks current parameter index',
                'range': f'0 to {self.config.p - 1}'
            },
            'state_register': {
                'width': 3,
                'description': 'FSM state encoding',
                'states': {
                    0: self.STATE_IDLE,
                    1: self.STATE_LOAD,
                    2: self.STATE_MAC_S1,
                    3: self.STATE_MAC_S2,
                    4: self.STATE_ADD_OFFSET
                }
            }
        }
    
    def estimate_resources(self) -> dict:
        """
        Estimate FPGA resource usage
        
        Returns:
            Dictionary with estimated resource counts
        """
        return {
            'dsp_blocks': 1,
            'dsp_utilization': 1,
            'accumulator_registers': 1,
            'accumulator_width': 32,
            'input_registers': 2,  # F and theta
            'counter_registers': 2,  # solution_idx and param_idx
            'state_register': 1,
            'adder': 1,  # For f offset addition
            'total_logic_cells': 100,  # Approximate FSM logic
            'memory_bandwidth': {
                'F_reads_per_cycle': 1.0 / 3,  # One read every 3 cycles
                'theta_reads_per_cycle': 1.0 / 3,
                'f_reads_per_cycle': 1.0 / self.config.cycles_per_solution
            }
        }
    
    def get_critical_path(self) -> dict:
        """
        Identify critical timing paths
        
        Returns:
            Dictionary describing critical paths
        """
        return {
            'longest_path': 'DSP multiply -> Accumulator add',
            'path_stages': [
                'Memory read (F, theta)',
                'DSP multiplier (3 pipeline stages)',
                'Accumulator addition',
                'Register write'
            ],
            'estimated_delay_ns': 5.0,  # Typical for modern FPGAs
            'max_frequency_mhz': 200
        }
    
    def get_latency_breakdown(self) -> dict:
        """
        Get detailed latency breakdown
        
        Returns:
            Dictionary with latency components
        """
        return {
            'cycles_per_mac': self.config.cycles_per_mac,
            'macs_per_solution': self.config.macs_per_solution,
            'cycles_per_solution': self.config.cycles_per_solution,
            'offset_add_cycles': 1,
            'total_solutions': self.config.n,
            'total_cycles': self.config.total_cycles,
            'dsp_pipeline_depth': 3,
            'fsm_overhead_cycles': self.config.n  # One offset add per solution
        }


def print_sequential_sm_report(config: SequentialSMConfig) -> None:
    """
    Print detailed Sequential SM configuration report
    
    Args:
        config: Sequential SM configuration
    """
    print("=" * 60)
    print("Sequential State Machine Configuration")
    print("=" * 60)
    print(f"Problem Size: n={config.n}, p={config.p}")
    print(f"Available DSP: {config.dsp_count}")
    print(f"\nSequential Processing:")
    print(f"  - MACs per Solution: {config.macs_per_solution}")
    print(f"  - Cycles per MAC: {config.cycles_per_mac}")
    print(f"  - Cycles per Solution: {config.cycles_per_solution}")
    print(f"    = {config.p} MACs × 3 cycles + 1 offset add")
    print(f"\nTotal Timing:")
    print(f"  - Total Solutions: {config.n}")
    print(f"  - Total Cycles: {config.total_cycles}")
    print(f"  - Total MACs: {config.n * config.p}")
    print(f"\nEfficiency:")
    total_macs = config.n * config.p
    theoretical_cycles = total_macs * 3
    overhead = config.total_cycles - theoretical_cycles
    print(f"  - Theoretical MAC Cycles: {theoretical_cycles}")
    print(f"  - Offset Add Overhead: {overhead} cycles ({config.n} additions)")
    print(f"  - DSP Utilization: {theoretical_cycles / config.total_cycles:.2%}")
    print(f"  - Throughput: {config.n / config.total_cycles:.6f} solutions/cycle")
    print(f"\nResource Footprint:")
    print(f"  - Extremely minimal (1 DSP + basic FSM)")
    print(f"  - Ideal for area-constrained designs")
    print("=" * 60)


def compare_all_strategies(n: int, p: int, sequential_config: SequentialSMConfig) -> None:
    """
    Compare Sequential SM with other strategies
    
    Args:
        n: Number of variables
        p: Number of parameters
        sequential_config: Sequential SM configuration
    """
    print("\n" + "=" * 60)
    print("Strategy Comparison Summary")
    print("=" * 60)
    
    print(f"\nProblem: n={n}, p={p}")
    print(f"\n1. Sequential SM (D=1):")
    print(f"   Cycles: {sequential_config.total_cycles}")
    print(f"   DSP: 1")
    print(f"   Area: Minimal")
    
    # Fully parallel (if D >= n*p)
    fully_parallel_dsps = n * p
    print(f"\n2. Fully Parallel (D={fully_parallel_dsps}):")
    print(f"   Cycles: 3")
    print(f"   DSP: {fully_parallel_dsps}")
    print(f"   Speedup: {sequential_config.total_cycles / 3:.1f}x")
    print(f"   Area: Maximum")
    
    # Solution-TDM (example: 2 parallel solutions)
    if p > 0:
        D_sol = 2 * p
        if D_sol < n * p:
            try:
                from scripts.bst_sol.fixed_point_constrained.solution_tdm import configure_solution_tdm
                sol_config = configure_solution_tdm(n, p, D_sol)
                print(f"\n3. Solution-TDM (D={D_sol}):")
                print(f"   Cycles: {sol_config.total_cycles}")
                print(f"   DSP: {D_sol}")
                print(f"   Speedup vs Sequential: {sequential_config.total_cycles / sol_config.total_cycles:.1f}x")
                print(f"   Area: Medium")
            except:
                pass
    
    # Parameter-TDM (example: half parameters)
    if p >= 2:
        D_param = max(1, p // 2)
        if D_param < p:
            try:
                from scripts.bst_sol.fixed_point_constrained.parameter_tdm import configure_parameter_tdm
                param_config = configure_parameter_tdm(n, p, D_param)
                print(f"\n4. Parameter-TDM (D={D_param}):")
                print(f"   Cycles: {param_config.total_cycles}")
                print(f"   DSP: {D_param}")
                print(f"   Speedup vs Sequential: {sequential_config.total_cycles / param_config.total_cycles:.1f}x")
                print(f"   Area: Low-Medium")
            except:
                pass
    
    print("\n" + "=" * 60)
    print("Trade-off Summary:")
    print("  Sequential SM: Minimal area, maximum latency")
    print("  Parameter-TDM: Low area, reduced latency")
    print("  Solution-TDM: Medium area, good latency")
    print("  Fully Parallel: Maximum area, minimum latency")
    print("=" * 60)


# ============================================================================
# Main Entry Point for Strategy Selector
# ============================================================================

def generate_sequential_sm_solver(
    config_file: Path,
    output_dir: Path,
    verbose: bool = False
) -> int:
    """
    Generate Sequential State Machine BST solver hardware
    
    This is the main entry point called by strategy_selector.py
    
    Args:
        config_file: Path to Verilog config file
        output_dir: Output directory for RTL
        verbose: Enable verbose output
    
    Returns:
        0 on success, 1 on failure
    
    Constraints:
        - Requires: D = 1 (single DSP block)
        - Strategy: Sequential FSM processing
        - Timing: n * (3*p + 1) total cycles
    """
    try:
        import sys
        
        # Add project root to Python path
        SCRIPT_DIR = Path(__file__).parent
        PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        
        from scripts.bst_sol.common.config_parser import VerilogConfigParser
        from scripts.bst_sol.fixed_point_constrained.sequential_sm_generator import SequentialSMGenerator
        
        # Parse configuration file
        parser = VerilogConfigParser(str(config_file))
        cfg = parser.get_config()
        
        n = cfg.n_solutions
        p = cfg.n_parameters
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Sequential State Machine Solver Generation")
            print(f"{'='*60}")
            print(f"Input Config: {config_file}")
            print(f"Problem Size: n={n}, p={p}")
        
        # Configure Sequential SM architecture
        sm_config = configure_sequential_sm(n, p)
        
        # Print detailed report
        if verbose:
            print_sequential_sm_report(sm_config)
            compare_all_strategies(n, p, sm_config)
        
        # Generate RTL using SequentialSMGenerator
        generator = SequentialSMGenerator(sm_config, cfg)
        success = generator.generate_rtl(output_dir)
        
        if success:
            print(f"✓ Sequential SM RTL generated successfully")
            print(f"  Output Directory: {output_dir}")
            print(f"  DSP Blocks: 1")
            print(f"  Total Cycles: {sm_config.total_cycles}")
            print(f"  Cycles per Solution: {sm_config.cycles_per_solution}")
            return 0
        else:
            print(f"❌ Sequential SM RTL generation failed")
            return 1
        
    except Exception as e:
        print(f"❌ Sequential SM generation error: {e}")
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
        description="Generate Sequential State Machine BST Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation
  python sequential_sm.py config.v output/

  # Verbose mode with strategy comparison
  python sequential_sm.py config.v output/ -v

  # Generate and show FSM details
  python sequential_sm.py config.v output/ --show-fsm
        """
    )
    
    parser.add_argument('config_file', type=Path,
                        help='Path to Verilog config file')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for RTL')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--show-fsm', action='store_true',
                        help='Display FSM state transitions')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate solver
    exit_code = generate_sequential_sm_solver(
        args.config_file,
        args.output_dir,
        args.verbose
    )
    
    # Optional: Show FSM details
    if args.show_fsm and exit_code == 0:
        import sys
        SCRIPT_DIR = Path(__file__).parent
        PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        
        from scripts.bst_sol.common.config_parser import VerilogConfigParser
        parser = VerilogConfigParser(str(args.config_file))
        cfg = parser.get_config()
        
        config = configure_sequential_sm(cfg.n_solutions, cfg.n_parameters)
        scheduler = SequentialSMScheduler(config)
        
        print("\n" + "="*60)
        print("FSM State Transitions (first 10 cycles)")
        print("="*60)
        for i in range(min(10, config.total_cycles)):
            state, sol, param = scheduler.get_state_at_cycle(i)
            print(f"Cycle {i:3d}: {state:12s} | Solution {sol:2d} | Param {param:2d}")
    
    exit(exit_code)