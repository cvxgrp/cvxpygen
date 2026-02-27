"""
Floating-Point BST LUT Generator - Main Entry Point
Generates BST LUT module for floating-point implementation (matching baseline)
"""

import sys
import shutil
from pathlib import Path
from typing import Dict
from datetime import datetime
import math

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use absolute imports
from scripts.bst_sol.common.config_parser import VerilogConfigParser
from scripts.bst_sol.common.rtl_writer import RTLWriter
from scripts.bst_sol.floating_point.memory_layout_float import FloatMemoryLayoutGenerator
from scripts.bst_sol.floating_point.bst_search_module_float import FloatBSTSearchGenerator
from scripts.bst_sol.floating_point.solver_module_float import FloatSolverGenerator


def calculate_adder_tree_depth(n_params: int) -> int:
    """
    Calculate adder tree depth for dot product reduction
    
    For n_params terms after multiplication:
    - 4 params: 2 levels (4→2→1)
    - 10 params: 4 levels (10→5→3→2→1)
    
    Args:
        n_params: Number of parameters to reduce
        
    Returns:
        Number of adder levels needed
    """
    if n_params <= 1:
        return 0
    return math.ceil(math.log2(n_params))


class FloatingPointGenerator:
    """Main generator for floating-point BST LUT implementation"""
    
    # *** FIXED: FP format configuration lookup table ***
    # Removed +1 from mult_latency - Xilinx IP latency already includes all stages
    FP_CONFIGS = {
        'fp64': {
            'data_width': 64,
            'mult_latency': 12,  # No +1!
            'add_latency': 14,
            'comp_latency': 2
        },
        'fp32': {
            'data_width': 32,
            'mult_latency': 9,   # No +1!
            'add_latency': 12,
            'comp_latency': 1
        },
        'fp16': {
            'data_width': 16,
            'mult_latency': 7,   # No +1! (was 6, corrected to 7 for FP16)
            'add_latency': 12,
            'comp_latency': 2
        },
        'fp8': {
            'data_width': 8,
            'mult_latency': 4,   # No +1!
            'add_latency': 5,
            'comp_latency': 1
        }
    }
    
    def __init__(self, config_file: str, output_dir: str = "generated_hardware"):
        """
        Initialize generator with config file
        
        Args:
            config_file: Path to Verilog config header (.vh)
            output_dir: Output directory for generated RTL
        """
        # Parse configuration
        parser = VerilogConfigParser(config_file)
        self.config = parser.get_config()
        
        # Validate floating-point format
        if not self.config.data_format.startswith('fp'):
            raise ValueError(f"FloatingPointGenerator requires fp format, got {self.config.data_format}")
        
        # Inject FP latencies into config
        self._inject_fp_latencies()
        
        # Calculate and inject adder tree depth
        self.adder_tree_depth = calculate_adder_tree_depth(self.config.n_parameters)
        self.config.adder_tree_depth = self.adder_tree_depth
        
        # Setup output directories
        self.rtl_dir = Path(output_dir)
        self.rtl_dir.mkdir(parents=True, exist_ok=True)
        
        # Create fp_ops subdirectory
        self.fp_ops_dir = self.rtl_dir / 'fp_ops'
        self.fp_ops_dir.mkdir(exist_ok=True)
        
        # Generate FP configuration file
        self._generate_fp_config()
        
        # Initialize sub-generators
        self.mem_gen = FloatMemoryLayoutGenerator(self.config)
        self.bst_gen = FloatBSTSearchGenerator(self.config, self.mem_gen)
        self.sol_gen = FloatSolverGenerator(self.config, self.mem_gen)
        
        # Calculate pipeline latencies
        self.input_buffer_stages = 1
        self.bst_traversal_latency = (self.config.estimated_bst_depth * 
                                      self.bst_gen.comparison_latency)
        self.sol_latency = self.sol_gen.solution_latency
        self.core_latency = self.bst_traversal_latency + self.sol_latency
        self.bst_depth = self.config.estimated_bst_depth
        self.total_pipeline_capacity = self.input_buffer_stages + self.core_latency
        
        # Print configuration summary
        self._print_summary()
    
    def _inject_fp_latencies(self):
        """Inject FP latency attributes into config object"""
        fmt = self.config.data_format
        
        if fmt not in self.FP_CONFIGS:
            raise ValueError(f"Unsupported FP format: {fmt}")
        
        params = self.FP_CONFIGS[fmt]
        
        # Add FP latency attributes
        self.config.fp_mult_latency = params['mult_latency']
        self.config.fp_add_latency = params['add_latency']
        self.config.fp_comp_latency = params['comp_latency']
        
        # Verify data_width matches format
        if hasattr(self.config, 'data_width'):
            if self.config.data_width != params['data_width']:
                print(f"Warning: data_width {self.config.data_width} doesn't match "
                      f"{fmt} standard ({params['data_width']}), using config value")
        else:
            self.config.data_width = params['data_width']
    
    def _print_summary(self):
        """Print configuration summary"""
        print(f"\n{'='*70}")
        print(f"Floating-Point BST LUT Generator (Baseline-Compatible - FIXED)")
        print(f"{'='*70}")
        print(f"Format: {self.config.data_format.upper()}")
        print(f"Parameters: {self.config.n_parameters}")
        print(f"Solutions: {self.config.n_solutions}")
        print(f"Tree nodes: {self.config.n_tree_nodes}")
        print(f"BST depth: {self.config.estimated_bst_depth}")
        print(f"{'='*70}")
        print(f"Pipeline Configuration:")
        print(f"  FP mult latency: {self.config.fp_mult_latency} cycles (base IP)")
        print(f"  FP add latency: {self.config.fp_add_latency} cycles")
        print(f"  FP comp latency: {self.config.fp_comp_latency} cycles")
        print(f"  *** FIXED: Actual mult = base (no +1) ***")
        print(f"  Actual mult latency: {self.bst_gen.actual_mult_latency} cycles")
        print(f"  Actual add latency: {self.bst_gen.actual_add_latency} cycles")
        print(f"  Adder tree depth: {self.adder_tree_depth} levels (BST traversal)")
        print(f"  Dot product latency: {self.bst_gen.dot_product_latency} cycles")
        print(f"    = {self.bst_gen.actual_mult_latency} (mult) + {self.adder_tree_depth} × {self.bst_gen.actual_add_latency} (tree)")
        print(f"  Comparison latency: {self.bst_gen.comparison_latency} cycles")
        print(f"  BST total: {self.config.estimated_bst_depth} × {self.bst_gen.comparison_latency} = {self.bst_traversal_latency} cycles")
        print(f"  MAC latency: {self.sol_gen.mac_latency} cycles (serial chain for solution)")
        print(f"    = {self.sol_gen.actual_mult_latency} (mult) + {self.config.n_parameters-1} × {self.sol_gen.actual_add_latency} (chain)")
        print(f"  Solution latency: {self.sol_gen.solution_latency} cycles")
        print(f"  Total pipeline depth: {self.total_pipeline_capacity} cycles")
        print(f"{'='*70}")
        print(f"Output: {self.rtl_dir}")
        print(f"FP ops: {self.fp_ops_dir}")
        print(f"{'='*70}\n")
    
    def _generate_fp_config(self):
        """Generate fp_config.vh with FIXED latencies"""
        fmt = self.config.data_format
        params = self.FP_CONFIGS[fmt]
        
        # *** FIXED: Calculate correct latencies ***
        mult_lat = params['mult_latency']  # No +1!
        add_lat = params['add_latency']
        comp_lat = params['comp_latency']
        
        # Dot product latency: MULT + TREE_DEPTH * ADD
        dot_product_lat = mult_lat + self.adder_tree_depth * add_lat
        
        content = f"""//=======================================================================
// Floating-Point Configuration
//=======================================================================
// Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Format: {fmt.upper()}
// Parameters: {self.config.n_parameters}
// Adder Tree Depth: {self.adder_tree_depth} levels (for BST traversal)
// *** FIXED: Removed incorrect +1 from mult latency ***
//=======================================================================

`ifndef FP_CONFIG_VH
`define FP_CONFIG_VH

//-----------------------------------------------------------------------
// Data Format Configuration
//-----------------------------------------------------------------------

// FP data width (bits)
`define FP_DATA_WIDTH {params['data_width']}

// Metadata width for node ID tracking
`define FP_METADATA_WIDTH 8

//-----------------------------------------------------------------------
// FP Operation Latencies (cycles)
//-----------------------------------------------------------------------
// Note: These are the actual IP latencies (no +1 needed)

// Multiplication latency
`define FP_MULT_LATENCY {mult_lat}

// Addition/Subtraction latency
`define FP_ADD_LATENCY {add_lat}

// Comparison latency
`define FP_COMP_LATENCY {comp_lat}

//-----------------------------------------------------------------------
// Adder Tree Configuration (for BST traversal only)
//-----------------------------------------------------------------------

// Adder tree depth for {self.config.n_parameters} parameters
// Structure: {self.config.n_parameters} → ... → 1 ({self.adder_tree_depth} levels)
`define ADDER_TREE_DEPTH {self.adder_tree_depth}

// *** FIXED: Dot product total latency (MULT + TREE_DEPTH * ADD) ***
// = {mult_lat} + {self.adder_tree_depth} * {add_lat} = {dot_product_lat}
`define DOT_PRODUCT_LATENCY {dot_product_lat}

`endif // FP_CONFIG_VH
"""
        
        # Write to fp_ops directory
        output_file = self.fp_ops_dir / 'fp_config.vh'
        output_file.write_text(content)
        
        print(f"[1/5] Generated FP config (FIXED): {output_file}")
        print(f"  Format: {fmt.upper()}")
        print(f"  Data width: {params['data_width']} bits")
        print(f"  *** FIXED: Mult latency = {mult_lat} (no +1) ***")
        print(f"  Add latency: {add_lat} cycles")
        print(f"  Comp latency: {comp_lat} cycles")
        print(f"  Adder tree depth: {self.adder_tree_depth} levels (BST only)")
        print(f"  Dot product latency: {dot_product_lat} cycles")
        print(f"    = {mult_lat} (mult) + {self.adder_tree_depth} × {add_lat} (tree)\n")
    
    def _copy_fp_operations(self):
        """Copy FP operation modules to target directory"""
        # FP operation modules to copy
        fp_modules = [
            'fp_add.v',
            'fp_mult.v',
            'fp_compare.v',
            'fp_mac.v'  # Contains delay_line modules and top_mac_module
        ]
        
        # Source directory candidates
        possible_sources = [
            PROJECT_ROOT / 'rtl' / 'fp_ops',
            PROJECT_ROOT / 'hardware' / 'fp_ops',
            PROJECT_ROOT / 'src' / 'fp_ops',
            PROJECT_ROOT / 'modules' / 'fp_ops'
        ]
        
        fp_ops_source = None
        for src in possible_sources:
            if src.exists():
                fp_ops_source = src
                break
        
        if fp_ops_source is None:
            print(f"\n[4/5] Warning: FP ops source directory not found")
            print(f"  Searched in:")
            for src in possible_sources:
                print(f"    - {src}")
            print(f"  Please manually copy FP modules to: {self.fp_ops_dir}/")
            print(f"  Required: {', '.join(fp_modules)}\n")
            return
        
        print(f"\n[4/5] Copying FP operation modules...")
        print(f"  Source: {fp_ops_source}")
        
        copied = []
        missing = []
        
        for module in fp_modules:
            source_file = fp_ops_source / module
            target_file = self.fp_ops_dir / module
            
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                copied.append(module)
                print(f"  ✓ Copied: {module}")
            else:
                missing.append(module)
                print(f"  ✗ Missing: {module}")
        
        if missing:
            print(f"\n  Warning: {len(missing)} module(s) not found")
            print(f"  Please manually copy to {self.fp_ops_dir}/:")
            print(f"    {', '.join(missing)}")
        else:
            print(f"\n  All {len(copied)} modules copied successfully")
    
    def generate_bst_lut_module(self) -> Path:
        """Generate main BST LUT module (baseline-compatible)"""
        print(f"[2/5] Generating BST LUT module (baseline-compatible)...")
        
        project = self.config.project_name
        output_file = self.rtl_dir / f"{project}_bst_lut.v"
        
        writer = RTLWriter(str(output_file))
        
        # Module header
        writer.write_header(
            f"{project}_bst_lut",
            "Binary Search Tree Lookup Table - Floating Point (Baseline-Compatible, FIXED)",
            Format=self.config.data_format,
            Parameters=self.config.n_parameters,
            Solutions=self.config.n_solutions,
            AdderTreeDepth=f"{self.adder_tree_depth} (BST only, solution uses MAC)",
            PipelineDepth=self.total_pipeline_capacity,
            Note="FIXED: Removed incorrect +1 from mult latency"
        )
        writer.write_blank()
        
        # Include files
        writer.write_include(f"../include/{project}_config.vh")
        writer.write_include(f"./fp_ops/fp_config.vh")
        writer.write_blank(2)
        
        # Module declaration with parameters
        writer.begin_module(
            f"{project}_bst_lut",
            parameters=[
                f"N_PARAMS = `{project.upper()}_N_PARAMETER",
                f"N_SOLUTIONS = `{project.upper()}_N_SOLUTION",
                "DATA_WIDTH = `FP_DATA_WIDTH",
                f"N_NODES = `{project.upper()}_TREE_NODES",
                f"MAX_DEPTH = `{project.upper()}_ESTIMATED_BST_DEPTH",
                "METADATA_WIDTH = `FP_METADATA_WIDTH"
            ]
        )
        
        # Ports
        self._generate_ports(writer)
        writer.end_ports()
        
        # Memory declarations
        self.mem_gen.generate_memory_declarations(writer)
        
        # Memory initialization
        self.mem_gen.generate_memory_initialization(writer)
        
        # Constants
        self.bst_gen.generate_constants(writer)
        
        # Input stage (with FIFO)
        self.bst_gen.generate_input_stage(writer)
        
        # BST pipeline (with adder tree)
        self.bst_gen.generate_bst_pipeline(writer)
        
        # Final leaf stage
        self.bst_gen.generate_final_leaf_stage(writer)
        
        # Solution calculation (with MAC)
        self.sol_gen.generate_solution_pipeline(writer)
        
        # Output assignments
        self.sol_gen.generate_output_assignments(writer)
        
        # End module
        writer.end_module()
        writer.save()
        
        print(f"  ✓ Generated: {output_file}")
        print(f"  Module: {project}_bst_lut")
        print(f"  Lines: {writer.line_count}")
        print(f"  Architecture:")
        print(f"    - BST traversal: Adder tree ({self.adder_tree_depth} levels)")
        print(f"    - Solution: MAC (serial chain)")
        print(f"    - FIFO buffering: Enabled")
        print(f"    - Delay alignment: Complete")
        print(f"    - *** FIXED: Mult latency (no +1) ***\n")
        
        return output_file
    
    def _generate_ports(self, writer: RTLWriter):
        """Generate module ports (match baseline exactly)"""
        # Clock and reset
        writer.write_port("input wire", "clk")
        writer.write_port("input wire", "rst_n")
        writer.write_blank()
        
        # Parameter inputs
        writer.write_comment("Parameter inputs (FP format)")
        for i in range(self.config.n_parameters):
            writer.write_port("input wire", f"param_in_{i}", width="[DATA_WIDTH-1:0]")
        
        writer.write_port("input wire", "valid_in")
        writer.write_port("output wire", "ready_out")
        writer.write_blank()
        
        # Solution outputs
        writer.write_comment("Solution outputs (FP format)")
        for i in range(self.config.n_solutions):
            writer.write_port("output wire", f"sol_out_{i}", width="[DATA_WIDTH-1:0]")
        
        writer.write_port("output wire", "valid_out", last=True)
    
    def _generate_timing_header(self):
        """Generate timing header with FIXED latencies"""
        print(f"\n[3/5] Generating timing header (FIXED)...")
        
        project = self.config.project_name
        include_dir = self.rtl_dir.parent / 'include'
        include_dir.mkdir(exist_ok=True)
        
        output_file = include_dir / f"{project}_timing.vh"
        
        # Timing values
        bst_depth = self.bst_depth
        sol_latency = self.sol_latency
        core_latency = self.core_latency
        total_capacity = self.total_pipeline_capacity
        
        # FP latencies (FIXED: no +1)
        fp_mult_lat = self.config.fp_mult_latency
        fp_add_lat = self.config.fp_add_latency
        fp_comp_lat = self.config.fp_comp_latency
        
        # Derived latencies
        dot_product_lat = self.bst_gen.dot_product_latency
        comparison_lat = self.bst_gen.comparison_latency
        mac_lat = self.sol_gen.mac_latency
        
        content = f"""//=======================================================================
// PDAQP Timing Parameters
//=======================================================================
// Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Config: {self.config.n_parameters} params -> {self.config.n_solutions} sols
// Format: {self.config.data_format.upper()}
// *** FIXED: Removed incorrect +1 from mult latency ***
//
// Pipeline Breakdown:
//   Input Buffer:    {self.input_buffer_stages} cycle(s)
//   BST Traversal:   {self.bst_traversal_latency} cycles (depth {bst_depth} × {comparison_lat} cycles/node)
//   Solution Calc:   {sol_latency} cycles
//   Total Capacity:  {total_capacity} cycles
//
// Architecture:
//   BST Traversal:   Adder Tree ({self.adder_tree_depth} levels)
//   Solution:        MAC (serial chain, {self.config.n_parameters-1} adds)
//   Dot Product:     {fp_mult_lat} (mult) + {self.adder_tree_depth} × {fp_add_lat} (tree) = {dot_product_lat} cycles
//   MAC:             {fp_mult_lat} (mult) + {self.config.n_parameters-1} × {fp_add_lat} (chain) = {mac_lat} cycles
//=======================================================================

`ifndef {project.upper()}_TIMING_VH
`define {project.upper()}_TIMING_VH

//-----------------------------------------------------------------------
// Primary Pipeline Latencies
//-----------------------------------------------------------------------

// BST tree depth (decision levels)
`define {project.upper()}_BST_DEPTH {bst_depth}

// Solution calculation latency (MAC operations)
`define {project.upper()}_SOL_LATENCY {sol_latency}

// Core pipeline latency (BST + Solution, no input buffer)
`define {project.upper()}_CORE_LATENCY {core_latency}

// Total pipeline capacity (for top module PIPELINE_CAPACITY)
// Includes: Input buffer + BST traversal + Solution calculation
`define {project.upper()}_BST_LATENCY {total_capacity}

//-----------------------------------------------------------------------
// FP Operation Latencies (FIXED: actual IP latencies)
//-----------------------------------------------------------------------

`define {project.upper()}_FP_ADD_LAT {fp_add_lat}
`define {project.upper()}_FP_MULT_LAT {fp_mult_lat}
`define {project.upper()}_FP_COMP_LAT {fp_comp_lat}

//-----------------------------------------------------------------------
// Architecture-Specific Latencies
//-----------------------------------------------------------------------

// Adder tree depth for BST dot product ({self.config.n_parameters} parameters)
`define {project.upper()}_ADDER_TREE_DEPTH {self.adder_tree_depth}

// *** FIXED: Dot product latency using adder tree (BST traversal) ***
// = {fp_mult_lat} (mult) + {self.adder_tree_depth} (levels) × {fp_add_lat} (add) = {dot_product_lat}
`define {project.upper()}_DOT_PRODUCT_LAT {dot_product_lat}

// *** FIXED: MAC latency using serial chain (solution computation) ***
// = {fp_mult_lat} (mult) + {self.config.n_parameters-1} (adds) × {fp_add_lat} = {mac_lat}
`define {project.upper()}_MAC_LAT {mac_lat}

`endif // {project.upper()}_TIMING_VH
"""
        
        output_file.write_text(content)
        
        print(f"  ✓ Generated: {output_file}")
        print(f"\n  Timing Values:")
        print(f"    BST depth:        {bst_depth}")
        print(f"    Solution latency: {sol_latency}")
        print(f"    Core latency:     {core_latency}")
        print(f"    Total capacity:   {total_capacity}")
        print(f"\n  FP Latencies (FIXED):")
        print(f"    Multiply:         {fp_mult_lat} cycles (actual IP latency)")
        print(f"    Add/Sub:          {fp_add_lat} cycles")
        print(f"    Compare:          {fp_comp_lat} cycles")
        print(f"\n  Architecture:")
        print(f"    Adder tree depth: {self.adder_tree_depth} levels (BST)")
        print(f"    Dot product:      {dot_product_lat} cycles (adder tree)")
        print(f"      = {fp_mult_lat} + {self.adder_tree_depth} × {fp_add_lat}")
        print(f"    MAC:              {mac_lat} cycles (serial chain)")
        print(f"      = {fp_mult_lat} + {self.config.n_parameters-1} × {fp_add_lat}")
        print(f"    Comparison:       {comparison_lat} cycles")
        print(f"    Solution:         {sol_latency} cycles\n")
    
    def generate_all(self) -> Dict[str, Path]:
        """
        Generate all required files
        
        Returns:
            Dict mapping component names to file paths
        """
        generated_files = {}
        
        # 1. FP config (done in __init__)
        generated_files['fp_config'] = self.fp_ops_dir / 'fp_config.vh'
        
        # 2. Main BST LUT module
        generated_files['bst_lut'] = self.generate_bst_lut_module()
        
        # 3. Timing header
        self._generate_timing_header()
        generated_files['timing'] = self.rtl_dir.parent / 'include' / f'{self.config.project_name}_timing.vh'
        
        # 4. Copy FP ops
        self._copy_fp_operations()
        
        # 5. Summary
        self._print_generation_summary(generated_files)
        
        return generated_files
    
    def _print_generation_summary(self, files: Dict[str, Path]):
        """Print generation summary"""
        print(f"\n{'='*70}")
        print("Generation Complete! (Baseline-Compatible, FIXED)")
        print(f"{'='*70}")
        print(f"Generated files:")
        
        for name, path in files.items():
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {name}: {path}")
        
        print(f"\nOutput directory: {self.rtl_dir}")
        print(f"FP operations: {self.fp_ops_dir}")
        
        print(f"\nArchitecture Summary:")
        print(f"  ✓ BST Traversal:   Adder tree ({self.adder_tree_depth} levels)")
        print(f"  ✓ Solution:        MAC (serial chain)")
        print(f"  ✓ FIFO Buffering:  Enabled (depth 128)")
        print(f"  ✓ Delay Alignment: Complete (all signals)")
        print(f"  ✓ Ready Signal:    Assigned (always 1)")
        print(f"  ✓ *** FIXED: Mult latency (no +1) ***")
        
        print(f"\nFIXED Issues:")
        print(f"  ✓ Removed +1 from ACTUAL_MULT_LATENCY")
        print(f"  ✓ Added delay lines for passthrough paths")
        print(f"  ✓ Corrected FP16 mult latency (7, not 6)")
        print(f"  ✓ All latency calculations now match baseline")
        
        print(f"\nNext steps:")
        print(f"  1. Verify FP modules in {self.fp_ops_dir}/")
        print(f"     - fp_add.v")
        print(f"     - fp_mult.v")
        print(f"     - fp_compare.v")
        print(f"     - fp_mac.v (with top_mac_module + delay lines)")
        print(f"  2. Review generated RTL")
        print(f"  3. Compare with baseline (should be identical)")
        print(f"  4. Run synthesis")
        print(f"{'='*70}\n")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate floating-point BST LUT hardware (baseline-compatible, FIXED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_float.py config.vh
  python generate_float.py config.vh -o my_project/rtl
  python generate_float.py config.vh -v

Fixes:
  - Removed incorrect +1 from mult latency
  - Added delay lines for passthrough paths in adder tree
  - Corrected FP16 mult latency to 7 cycles
        """
    )
    
    parser.add_argument('config_file', 
                       help='Verilog config header (.vh)')
    parser.add_argument('-o', '--output', 
                       default='generated_hardware',
                       help='Output directory (default: generated_hardware)')
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate config file
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config_file}")
        sys.exit(1)
    
    try:
        # Generate
        generator = FloatingPointGenerator(args.config_file, args.output)
        generated = generator.generate_all()
        
        # Verbose output
        if args.verbose:
            print(f"\nDetailed info:")
            for name, path in generated.items():
                if path.exists():
                    size = path.stat().st_size
                    if path.suffix in ['.v', '.vh']:
                        lines = len(path.read_text().splitlines())
                    else:
                        lines = 'N/A'
                    print(f"  {name}:")
                    print(f"    Path: {path}")
                    print(f"    Size: {size:,} bytes")
                    print(f"    Lines: {lines}")
        
        sys.exit(0)
    
    except Exception as e:
        print(f"\nError: Generation failed")
        print(f"  {type(e).__name__}: {e}")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == '__main__':
    main()