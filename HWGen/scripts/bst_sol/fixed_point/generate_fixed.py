"""
Fixed-Point BST LUT Generator - Main Entry Point
Generates BST LUT module only (top module handled separately)
"""

import sys
import math
from pathlib import Path
from typing import Dict

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use absolute imports
from scripts.bst_sol.common.config_parser import VerilogConfigParser
from scripts.bst_sol.common.rtl_writer.rtl_writer import RTLWriter
from scripts.bst_sol.fixed_point.memory_layout import MemoryLayoutGenerator
from scripts.bst_sol.fixed_point.bst_search_module import BSTSearchGenerator
from scripts.bst_sol.fixed_point.solver_module import SolverGenerator


class FixedPointGenerator:
    """Main generator for fixed-point BST LUT implementation"""
    
    def __init__(self, config_file: str, output_dir: str = "generated_hardware"):
        # Parse configuration
        parser = VerilogConfigParser(config_file)
        self.config = parser.get_config()
        
        # Validate fixed-point format
        if not self.config.data_format.startswith('fix'):
            raise ValueError(f"FixedPointGenerator requires fixed-point format, got {self.config.data_format}")
        
        # Setup output directory
        self.rtl_dir = Path(output_dir)
        self.rtl_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-generators
        self.mem_gen = MemoryLayoutGenerator(self.config)
        self.bst_gen = BSTSearchGenerator(self.config, self.mem_gen)
        self.sol_gen = SolverGenerator(self.config, self.mem_gen)
        
        # Calculate total pipeline depth
        self.pipe_depth = self.config.estimated_bst_depth + self.sol_gen.sol_stages
        
        print(f"\n{'='*70}")
        print(f"Fixed-Point BST LUT Generator")
        print(f"{'='*70}")
        print(f"Config: {self.config}")
        print(f"Memory: {self.mem_gen.get_memory_config()}")
        print(f"Pipeline depth: {self.pipe_depth} stages")
        print(f"  - BST: {self.config.estimated_bst_depth}")
        print(f"  - Solution: {self.sol_gen.sol_stages}")
        print(f"Output directory: {self.rtl_dir}")
        print(f"{'='*70}\n")
    
    def generate_bst_lut_module(self) -> Path:
        """Generate main BST LUT module"""
        project = self.config.project_name
        output_file = self.rtl_dir / f"{project}_bst_lut.v"
        
        writer = RTLWriter(output_file)
        
        # Header
        writer.write_header(
            f"{project}_bst_lut",
            "Binary Search Tree Lookup Table - Fixed Point Implementation",
            Format=self.config.data_format,
            Parameters=self.config.n_parameters,
            Solutions=self.config.n_solutions,
            PipelineDepth=self.pipe_depth
        )
        writer.write_blank()
        
        # Include configuration file
        writer.write_include(f"../include/{project}_config.vh")
        writer.write_blank(2)
        
        # Module declaration
        writer.begin_module(f"{project}_bst_lut")
        
        # Ports
        self._generate_ports(writer)
        writer.end_ports()
        
        # Memory declarations
        self.mem_gen.generate_memory_declarations(writer)
        
        # Memory initialization
        self.mem_gen.generate_memory_initialization(writer)
        
        # Pipeline registers
        self.bst_gen.generate_pipeline_registers(writer)
        self.sol_gen.generate_pipeline_registers(writer)
        
        # Combinational logic
        self.bst_gen.generate_combinational_logic(writer)
        
        # Integer loop variables
        writer.write_comment("Loop variables")
        writer.write_line("integer i, j;")
        writer.write_blank(2)
        
        # Main always block
        writer.begin_always("posedge clk or negedge rst_n")
        
        # Reset logic
        writer.begin_if("!rst_n")
        self._generate_reset_logic(writer)
        
        # Normal operation
        writer.begin_else()
        
        # BST traversal stages
        self.bst_gen.generate_stage0_input(writer)
        self.bst_gen.generate_bst_traversal_stages(writer)
        
        # Solution computation stages
        self.sol_gen.generate_stage_prepare(writer)
        self.sol_gen.generate_stage_mac(writer)
        self.sol_gen.generate_stage_offset(writer)
        self.sol_gen.generate_stage_output(writer)
        
        writer.end_if()
        writer.end_always()
        
        # Output assignments (outside always block)
        self._generate_output_assignment(writer)
        
        # End module
        writer.end_module()
        
        writer.save()
        return output_file
    
    def _generate_ports(self, writer: RTLWriter):
        """Generate module ports"""
        # Clock and reset
        writer.write_port("input", "clk")
        writer.write_port("input", "rst_n")
        writer.write_blank()
        
        # Parameter inputs
        writer.write_comment("Parameter inputs")
        for i in range(self.config.n_parameters):
            data_range = f"{self.config.data_width-1}:0"
            writer.write_port("input", f"param_in_{i}", width=data_range, signed=True, last=False)
        
        writer.write_port("input", "valid_in")
        writer.write_blank()
        
        # Solution outputs
        writer.write_comment("Solution outputs")
        for i in range(self.config.n_solutions):
            data_range = f"{self.config.data_width-1}:0"
            writer.write_port("output", f"solution_{i}", width=data_range, signed=True)
        
        writer.write_port("output", "valid_out", last=True)
    
    def _generate_output_assignment(self, writer: RTLWriter):
        """Generate output port assignments"""
        writer.write_blank()
        writer.write_section("Output Assignments")
        writer.write_blank()
        
        writer.write_comment("Connect internal registers to output ports")
        for i in range(self.config.n_solutions):
            writer.write_line(f"assign solution_{i} = sol_out_{i};")
        
        # Use final stage valid signal
        final_stage = self.config.estimated_bst_depth + self.sol_gen.sol_stages
        writer.write_line(f"assign valid_out = valid_pipe_sol[{final_stage}];")
        writer.write_blank()
    
    def _generate_reset_logic(self, writer: RTLWriter):
        """Generate reset logic"""
        writer.write_comment("Reset all pipeline stages")
        self.bst_gen.generate_reset_logic(writer)
        self.sol_gen.generate_reset_logic(writer)
    
    def generate_all(self) -> Dict[str, Path]:
        """Generate BST LUT module only"""
        print("Generating BST LUT module...")
        
        bst_file = self.generate_bst_lut_module()
        
        print(f"\n{'='*70}")
        print("✓ Generation complete!")
        print(f"{'='*70}")
        print(f"Output file:")
        print(f"  ✓ {bst_file}")
        
        return {
            'bst_lut': bst_file
        }


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fixed-point BST LUT hardware')
    parser.add_argument('config_file', help='Configuration .vh file')
    parser.add_argument('-o', '--output', default='generated_hardware', help='Output directory (rtl folder)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        generator = FixedPointGenerator(args.config_file, args.output)
        generated = generator.generate_all()
        
        if args.verbose:
            print(f"\nDetailed info:")
            for name, path in generated.items():
                size = path.stat().st_size
                print(f"  {name}: {size} bytes")
        
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())