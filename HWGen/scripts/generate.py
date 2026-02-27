# scripts/generate.py

"""
Unified Hardware Generation Manager
Manages all generation stages: config -> interface -> bst_sol (fixed/float)
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional
import subprocess
import os
import re
import json

# PROJECT_ROOT is parent of scripts/ directory
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class GenerationStage:
    """Base class for generation stages"""
    
    def __init__(self, name: str, script_path: Path, description: str = ""):
        self.name = name
        self.script_path = script_path
        self.description = description
    
    def run(self, args: List[str], verbose: bool = False) -> int:
        """Run this stage with given arguments"""
        cmd = [sys.executable, str(self.script_path)] + args
        
        print(f"\n{'='*70}")
        print(f"Stage: {self.name}")
        if self.description:
            print(f"Description: {self.description}")
        if verbose:
            print(f"Command: {' '.join(cmd)}")
        print(f"{'='*70}\n")
        
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT)}
        )
        return result.returncode


class HardwareGenerator:
    """Unified hardware generation manager"""
    
    def __init__(self):
        self.stages = {
            'config': GenerationStage(
                name='Configuration Generation',
                script_path=SCRIPTS_DIR / 'config/generate_config.py',
                description='Parse C/H files and generate Verilog config + memory files'
            ),
            'interface': GenerationStage(
                name='Interface Generation',
                script_path=SCRIPTS_DIR / 'interface/generate_interface.py',
                description='Generate AXI4-Stream top module with adaptive packing'
            ),
            'bst_fixed': GenerationStage(
                name='BST Solver Generation (Fixed-Point)',
                script_path=SCRIPTS_DIR / 'bst_sol/fixed_point/generate_fixed.py',
                description='Generate BST LUT core with fixed-point arithmetic'
            ),
            'bst_fixed_constrained': GenerationStage(
                name='BST Solver Generation (Fixed-Point DSP-Constrained)',
                script_path=SCRIPTS_DIR / 'bst_sol/fixed_point_constrained/strategy_selector.py',
                description='Generate BST LUT core with adaptive DSP allocation'
            ),
            'bst_float': GenerationStage(
                name='BST Solver Generation (Floating-Point)',
                script_path=SCRIPTS_DIR / 'bst_sol/floating_point/generate_float.py',
                description='Generate BST LUT core with FP32/FP16 arithmetic'
            ),
        }
        
        self.stage_order = ['config', 'interface', 'bst_solver']
    
    def run_stage(self, stage_name: str, args: List[str], verbose: bool = False) -> int:
        """Run a single stage"""
        if stage_name not in self.stages:
            print(f"Error: Unknown stage '{stage_name}'")
            print(f"Available stages: {', '.join(self.stages.keys())}")
            return 1
        
        return self.stages[stage_name].run(args, verbose=verbose)
    
    def _detect_data_format(self, config_file: Path) -> str:
        """
        Detect data format from generated config file
        Returns: 'fixed', 'fp16', or 'fp32'
        """
        try:
            if not config_file.exists():
                print(f"Warning: Config file not found: {config_file}")
                return 'fixed'
            
            content = config_file.read_text()
            
            # Method 1: Check PDAQP_USE_* flags (new format)
            use_fp32_match = re.search(r'`define\s+PDAQP_USE_FP32\s+(\d+)', content)
            use_fp16_match = re.search(r'`define\s+PDAQP_USE_FP16\s+(\d+)', content)
            use_fix16_match = re.search(r'`define\s+PDAQP_USE_FIX16\s+(\d+)', content)
            
            if use_fp32_match and use_fp32_match.group(1) == '1':
                return 'fp32'
            if use_fp16_match and use_fp16_match.group(1) == '1':
                return 'fp16'
            if use_fix16_match and use_fix16_match.group(1) == '1':
                return 'fixed'
            
            # Method 2: Check old DATA_FORMAT string
            format_match = re.search(r'`define\s+DATA_FORMAT\s+"([^"]+)"', content)
            if format_match:
                format_str = format_match.group(1).lower().strip()
                if format_str in ['fp32', 'float32', 'float_32']:
                    return 'fp32'
                elif format_str in ['fp16', 'float16', 'float_16']:
                    return 'fp16'
                elif format_str in ['fix16', 'fixed16', 'fixed_16', 'fixed']:
                    return 'fixed'
            
            # Method 3: Check for FIX16_MODE flag
            if re.search(r'`define\s+PDAQP_FIX16_MODE\s+1', content):
                return 'fixed'
            
            # Method 4: Fallback to generic USE_FP* defines
            if re.search(r'`define\s+\w*USE_FP32\s+1', content):
                return 'fp32'
            if re.search(r'`define\s+\w*USE_FP16\s+1', content):
                return 'fp16'
            
            # Default to fixed point
            print("  Warning: Could not detect format, defaulting to fixed-point")
            return 'fixed'
            
        except Exception as e:
            print(f"Warning: Error reading config file {config_file}: {e}")
            return 'fixed'
    
    def _extract_problem_size(self, config_file: Path) -> tuple:
        """
        Extract n (variables) and p (parameters) from config file
        Returns: (n, p) tuple
        """
        try:
            if not config_file.exists():
                return (None, None)
            
            content = config_file.read_text()
            
            # Look for various naming patterns
            patterns = [
                # New naming
                (r'`define\s+PDAQP_N_SOLUTION\s+(\d+)', r'`define\s+PDAQP_N_PARAMETER\s+(\d+)'),
                # Old naming
                (r'`define\s+PDAQP_N\s+(\d+)', r'`define\s+PDAQP_P\s+(\d+)'),
                # Alternative patterns
                (r'`define\s+N_SOLUTION\s+(\d+)', r'`define\s+N_PARAMETER\s+(\d+)'),
                (r'`define\s+NUM_VARS\s+(\d+)', r'`define\s+NUM_PARAMS\s+(\d+)'),
            ]
            
            for n_pattern, p_pattern in patterns:
                n_match = re.search(n_pattern, content)
                p_match = re.search(p_pattern, content)
                
                if n_match and p_match:
                    n = int(n_match.group(1))
                    p = int(p_match.group(1))
                    return (n, p)
            
            # If still not found, print debug info
            print("  Debug: Searching for problem size in config file:")
            for line in content.split('\n'):
                if 'define' in line and any(x in line for x in ['_N', '_P', 'SOLUTION', 'PARAMETER']):
                    print(f"    {line.strip()}")
            
            return (None, None)
            
        except Exception as e:
            print(f"Warning: Error extracting problem size: {e}")
            import traceback
            traceback.print_exc()
            return (None, None)
    
    def _run_strategy_selector(self, config_file: Path, available_dsp: int, 
                               verbose: bool = False) -> Optional[dict]:
        """
        Run DSP strategy selector using Python module invocation
        
        Returns:
            Strategy configuration dict or None if failed
        """
        try:
            # Extract problem size
            n, p = self._extract_problem_size(config_file)
            
            if n is None or p is None:
                print("  Warning: Could not extract problem size, skipping strategy selection")
                return None
            
            # Use Python module invocation to ensure proper imports
            cmd = [
                sys.executable, '-m',
                'scripts.bst_sol.fixed_point_constrained.strategy_selector',
                '--analyze',
                '-n', str(n),
                '-p', str(p),
                '--dsp', str(available_dsp),
                '--json'
            ]
            
            if verbose:
                print(f"\n{'='*70}")
                print("DSP Strategy Selection")
                print(f"{'='*70}")
                print(f"Problem size: n={n}, p={p}")
                print(f"Available DSP: {available_dsp}")
                print(f"Command: {' '.join(cmd)}")
                print(f"{'='*70}\n")
            
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT)}
            )
            
            if result.returncode != 0:
                print(f"  Warning: Strategy selector failed")
                if verbose:
                    print(f"  STDERR: {result.stderr}")
                    print(f"  STDOUT: {result.stdout}")
                return None
            
            # Parse JSON output
            try:
                strategy_config = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse strategy selector output")
                if verbose:
                    print(f"  Output: {result.stdout}")
                    print(f"  Error: {e}")
                return None
            
            if verbose:
                print("\nSelected Strategy:")
                print(f"  Strategy: {strategy_config.get('selected_strategy', 'N/A')}")
                print(f"  DSP Allocated: {strategy_config.get('dsp_allocated', 'N/A')}")
                if 'timing' in strategy_config:
                    print(f"  Solution Cycles: {strategy_config['timing'].get('t_solution', 'N/A')}")
                    print(f"  Pipeline Depth: {strategy_config['timing'].get('pipeline_depth', 'N/A')}")
            
            return strategy_config
            
        except Exception as e:
            print(f"  Warning: Strategy selection error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def run_all(self, 
                c_file: str,
                h_file: str,
                output_dir: Optional[str] = None,
                data_format: str = 'auto',
                max_iob: Optional[int] = None,
                no_input_buffer: bool = False,
                available_dsp: Optional[int] = None,
                force_sequential: bool = False,
                multipliers: int = 1,
                verbose: bool = False) -> int:
        """Run all stages in sequence"""
        c_path = Path(c_file).resolve()
        h_path = Path(h_file).resolve()
        
        # Validate input files
        if not c_path.exists():
            print(f"Error: C file not found: {c_path}")
            return 1
        
        if not h_path.exists():
            print(f"Error: Header file not found: {h_path}")
            return 1
        
        # Determine output directory
        if output_dir:
            out_dir = Path(output_dir).resolve()
        else:
            project_name = c_path.stem
            out_dir = Path.cwd() / f"codegen_{project_name}"
        
        # Display format mapping
        format_display = {
            'auto': 'FIX16 (auto Q-format)',
            'fixed': 'FIX16 (manual Q-format)',
            'fp32': 'FP32 (IEEE-754 single)',
            'fp16': 'FP16 (IEEE-754 half)'
        }.get(data_format, data_format)
        
        # DSP constraint info
        dsp_info = "Unlimited (Full Parallelism)"
        if force_sequential:
            dsp_info = f"Sequential (M={multipliers} multipliers)"
        elif available_dsp is not None:
            dsp_info = f"{available_dsp} blocks (Adaptive)"
        
        print(f"\n{'='*70}")
        print(f"UNIFIED HARDWARE GENERATION")
        print(f"{'='*70}")
        print(f"Input C:     {c_path}")
        print(f"Input H:     {h_path}")
        print(f"Output:      {out_dir}")
        print(f"Data format: {format_display}")
        print(f"DSP config:  {dsp_info}")
        if max_iob:
            print(f"Max IOB:     {max_iob}")
        print(f"Stages:      config -> interface -> bst_solver")
        print(f"{'='*70}\n")
        
        # Stage 1: Config generation
        print("▶ Stage 1/3: Configuration Generation")
        config_args = [
            '-c', str(c_path),
            '-H', str(h_path),
            '-o', str(out_dir / 'include')
        ]
        
        # Add format flag
        if data_format == 'fp32':
            config_args.append('--fp32')
        elif data_format == 'fp16':
            config_args.append('--fp16')
        elif data_format == 'fixed':
            config_args.append('--fixed')
        # 'auto' mode: no flag, let config generator auto-detect
        
        if verbose:
            config_args.append('--verbose')
        
        if self.run_stage('config', config_args, verbose=verbose) != 0:
            print("\n❌ Config generation failed")
            return 1
        
        print("✓ Stage 1 completed\n")
        
        # Stage 2: Interface generation
        print("▶ Stage 2/3: Interface Generation")
        config_file = out_dir / 'include' / f'{c_path.stem}_config.vh'
        
        interface_args = [
            '-c', str(config_file),
            '-o', str(out_dir)
        ]
        
        if max_iob:
            interface_args.extend(['--max-iob', str(max_iob)])
        
        if no_input_buffer:
            interface_args.append('--no-input-buffer')
        
        if verbose:
            interface_args.append('--verbose')
        
        if self.run_stage('interface', interface_args, verbose=verbose) != 0:
            print("\n❌ Interface generation failed")
            return 1
        
        print("✓ Stage 2 completed\n")
        
        # Stage 3: BST solver generation - AUTO-DETECT FORMAT & DSP STRATEGY
        print("▶ Stage 3/3: BST Solver Generation")
        
        # Detect actual format from generated config file
        actual_format = self._detect_data_format(config_file)
        
        if verbose:
            print(f"  Detected format from config: {actual_format}")
        
        # Determine BST generator based on format and DSP constraints
        strategy_config = None
        
        # Only use DSP-constrained generator for fixed-point with DSP limits
        if actual_format == 'fixed' and (available_dsp is not None or force_sequential):
            bst_stage = 'bst_fixed_constrained'
            print(f"  Using DSP-constrained fixed-point generator\n")
            
            # Run strategy selector in analysis mode if DSP limit specified
            if available_dsp is not None and not force_sequential:
                strategy_config = self._run_strategy_selector(
                    config_file, available_dsp, verbose=verbose
                )
            
        elif actual_format in ['fp32', 'fp16']:
            bst_stage = 'bst_float'
            print(f"  Using floating-point generator ({actual_format.upper()})\n")
            
        else:
            bst_stage = 'bst_fixed'
            print(f"  Using standard fixed-point generator (FIX16)\n")
        
        # Build BST arguments
        bst_args = [
            str(config_file),
            '-o', str(out_dir / 'rtl')
        ]
        
        # Add DSP constraint parameters if using constrained generator
        if bst_stage == 'bst_fixed_constrained':
            if available_dsp is not None:
                bst_args.extend(['--dsp', str(available_dsp)])
            if force_sequential:
                bst_args.append('--sequential')
                bst_args.extend(['--multipliers', str(multipliers)])
        
        if verbose:
            bst_args.append('-v')
        
        if self.run_stage(bst_stage, bst_args, verbose=verbose) != 0:
            print("\n❌ BST solver generation failed")
            return 1
        
        print("✓ Stage 3 completed\n")
        
        self._print_success_summary(
            out_dir, c_path.stem, actual_format, 
            strategy_config=strategy_config,
            available_dsp=available_dsp,
            force_sequential=force_sequential,
            multipliers=multipliers
        )
        
        return 0
    
    def _print_success_summary(self, out_dir: Path, project_name: str, 
                              data_format: str, strategy_config: Optional[dict] = None,
                              available_dsp: Optional[int] = None,
                              force_sequential: bool = False,
                              multipliers: int = 1):
        """Print success summary"""
        format_info = {
            'fixed': 'Fixed-Point (FIX16)',
            'fp32': 'Floating-Point (FP32)',
            'fp16': 'Floating-Point (FP16)'
        }.get(data_format, data_format.upper())
        
        print(f"\n{'='*70}")
        print(f"✓ ALL STAGES COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"\nGenerated files in: {out_dir}")
        print(f"Data format: {format_info}")
        
        # Print DSP strategy info if available
        if strategy_config:
            print(f"\nDSP Allocation Strategy:")
            print(f"  Selected: {strategy_config.get('selected_strategy', 'N/A')}")
            print(f"  DSP Used: {strategy_config.get('dsp_allocated', 'N/A')}")
            if 'timing' in strategy_config:
                print(f"  Solution Cycles: {strategy_config['timing'].get('t_solution', 'N/A')}")
                print(f"  Pipeline Depth: {strategy_config['timing'].get('pipeline_depth', 'N/A')}")
        elif force_sequential:
            print(f"\nDSP Allocation: Sequential Mode")
            print(f"  Multipliers: {multipliers}")
            print(f"  DSP Used: 0")
        elif available_dsp is not None:
            print(f"\nDSP Allocation: {available_dsp} blocks")
        else:
            print(f"\nDSP Allocation: Unlimited (Full Parallelism)")
        
        print(f"\nDirectory structure:")
        print(f"  {out_dir.name}/")
        print(f"  ├── include/")
        print(f"  │   ├── {project_name}_config.vh")
        print(f"  │   ├── {project_name}_timing.vh")
        print(f"  │   └── *.mem")
        print(f"  └── rtl/")
        print(f"      ├── {project_name}_top.v")
        print(f"      └── {project_name}_bst_lut.v")
        
        if data_format in ['fp32', 'fp16']:
            print(f"\n⚠  Floating-point design requires FP operator modules")
            print(f"   (fp_mult.v, fp_add.v, fp_compare.v, etc.)")
        
        print(f"{'='*70}\n")
    
    def list_stages(self):
        """List all available stages"""
        print("\nAvailable stages:")
        print(f"{'='*70}")
        
        for i, (key, stage) in enumerate(self.stages.items(), 1):
            print(f"\n{i}. {key}")
            print(f"   Name: {stage.name}")
            print(f"   Script: {stage.script_path.relative_to(PROJECT_ROOT)}")
            print(f"   Desc: {stage.description}")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Hardware Generation Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full parallelism (auto Q-format)
  python -m scripts.generate all -c solver.c -H solver.h
  
  # DSP-constrained design with 100 DSP blocks
  python -m scripts.generate all -c solver.c -H solver.h --dsp 100
  
  # Sequential mode with 2 multipliers
  python -m scripts.generate all -c solver.c -H solver.h --sequential --multipliers 2
  
  # Floating-point with DSP limit (DSP limit applies to fixed-point only)
  python -m scripts.generate all -c solver.c -H solver.h --fp32
  
  # Manual fixed-point Q-format with DSP constraint
  python -m scripts.generate all -c solver.c -H solver.h --fixed --dsp 50
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # All command - runs config -> interface -> bst_solver
    all_parser = subparsers.add_parser('all', help='Run all generation stages')
    all_parser.add_argument('-c', '--c-file', required=True, help='Input C source file')
    all_parser.add_argument('-H', '--header-file', required=True, help='Input header file')
    all_parser.add_argument('-o', '--output', help='Output directory')
    
    format_group = all_parser.add_mutually_exclusive_group()
    format_group.add_argument('--fp32', action='store_true', help='Use FP32 format')
    format_group.add_argument('--fp16', action='store_true', help='Use FP16 format')
    format_group.add_argument('--fixed', action='store_true', help='Use manual FIX16')
    
    all_parser.add_argument('--max-iob', type=int, help='Maximum IOB pins')
    all_parser.add_argument('--no-input-buffer', action='store_true', help='Disable input buffer')
    
    # DSP constraint options
    dsp_group = all_parser.add_mutually_exclusive_group()
    dsp_group.add_argument('--dsp', type=int, help='Available DSP48 blocks (enables adaptive allocation)')
    dsp_group.add_argument('--sequential', action='store_true', help='Force sequential mode (no DSP)')
    
    all_parser.add_argument('--multipliers', type=int, default=1, 
                           help='Number of multipliers for sequential mode (default: 1)')
    
    all_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Config command - stage 1 only
    config_parser = subparsers.add_parser('config', help='Generate config only')
    config_parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments for config stage')
    
    # Interface command - stage 2 only
    interface_parser = subparsers.add_parser('interface', help='Generate interface only')
    interface_parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments for interface stage')
    
    # BST fixed command - stage 3 fixed-point
    bst_fixed_parser = subparsers.add_parser('bst_fixed', help='Generate fixed-point BST solver')
    bst_fixed_parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments for bst_fixed stage')
    
    # BST fixed constrained command - stage 3 DSP-constrained fixed-point
    bst_fixed_constrained_parser = subparsers.add_parser('bst_fixed_constrained', 
                                                         help='Generate DSP-constrained fixed-point BST solver')
    bst_fixed_constrained_parser.add_argument('args', nargs=argparse.REMAINDER, 
                                              help='Arguments for bst_fixed_constrained stage')
    
    # BST float command - stage 3 floating-point
    bst_float_parser = subparsers.add_parser('bst_float', help='Generate floating-point BST solver')
    bst_float_parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments for bst_float stage')
    
    # List command - show all available stages
    list_parser = subparsers.add_parser('list', help='List all stages')
    
    args = parser.parse_args()
    
    generator = HardwareGenerator()
    
    # Show help if no command
    if not args.command:
        parser.print_help()
        return 1
    
    # List stages
    if args.command == 'list':
        generator.list_stages()
        return 0
    
    # Run all stages
    if args.command == 'all':
        # Determine data format from flags
        if args.fp32:
            data_format = 'fp32'
        elif args.fp16:
            data_format = 'fp16'
        elif args.fixed:
            data_format = 'fixed'
        else:
            data_format = 'auto'  # Auto-detect Q-format
        
        return generator.run_all(
            c_file=args.c_file,
            h_file=args.header_file,
            output_dir=args.output,
            data_format=data_format,
            max_iob=args.max_iob,
            no_input_buffer=args.no_input_buffer,
            available_dsp=args.dsp,
            force_sequential=args.sequential,
            multipliers=args.multipliers,
            verbose=args.verbose
        )
    
    # Handle single stage commands - pass through all args to stage script
    stage_map = {
        'config': 'config',
        'interface': 'interface',
        'bst_fixed': 'bst_fixed',
        'bst_fixed_constrained': 'bst_fixed_constrained',
        'bst_float': 'bst_float'
    }
    
    if args.command in stage_map:
        return generator.run_stage(stage_map[args.command], args.args, verbose=False)
    
    return 1


if __name__ == '__main__':
    sys.exit(main())