"""
Fixed-Point Memory Layout Generator
Manages ROM organization for BST data and feedback coefficients
"""

import math
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter


@dataclass
class MemoryConfig:
    """Memory configuration for BST LUT"""
    # Halfplane memory
    halfplane_entries: int       # Total halfplane coefficient entries (from PDAQP_HALFPLANES)
    halfplane_stride: int        # Elements per halfplane (from PDAQP_HALFPLANE_STRIDE)
    halfplane_width: int         # Bit width (from INPUT_DATA_WIDTH)
    
    # Feedback memory
    feedback_entries: int        # Total feedback coefficient entries (from PDAQP_FEEDBACKS)
    feedback_stride: int         # Elements per solution set (from PDAQP_SOL_PER_NODE)
    feedback_width: int          # Bit width (from OUTPUT_DATA_WIDTH)
    
    # BST structure memory
    hp_list_size: int           # Number of tree nodes (from PDAQP_TREE_NODES)
    jump_list_size: int         # Number of tree nodes (from PDAQP_TREE_NODES)
    index_width: int            # Bit width for addressing all arrays
    
    def __str__(self):
        return (f"Memory: HP={self.halfplane_entries}×{self.halfplane_width}b, "
                f"FB={self.feedback_entries}×{self.feedback_width}b, "
                f"BST={self.hp_list_size} nodes")


class MemoryLayoutGenerator:
    """Generate memory declarations and initialization for fixed-point"""
    
    def __init__(self, config: PDQAPConfig):
        self.config = config
        self.mem_config = self._build_memory_config()
    
    def _build_memory_config(self) -> MemoryConfig:
        """Build memory configuration directly from parsed config defines"""
        
        # All sizes come directly from config.vh defines - no calculation
        halfplane_entries = self.config.n_halfplanes     # PDAQP_HALFPLANES
        halfplane_stride = self.config.halfplane_stride   # PDAQP_HALFPLANE_STRIDE
        feedback_entries = self.config.n_feedbacks        # PDAQP_FEEDBACKS
        feedback_stride = self.config.sol_per_node        # PDAQP_SOL_PER_NODE
        
        hp_list_size = self.config.n_tree_nodes          # PDAQP_TREE_NODES
        jump_list_size = self.config.n_tree_nodes        # PDAQP_TREE_NODES
        
        # Index width must accommodate the largest array
        max_entries = max(
            halfplane_entries,
            feedback_entries,
            hp_list_size,
            jump_list_size
        )
        index_width = max(1, math.ceil(math.log2(max_entries)))
        
        return MemoryConfig(
            halfplane_entries=halfplane_entries,
            halfplane_stride=halfplane_stride,
            halfplane_width=self.config.data_width,
            feedback_entries=feedback_entries,
            feedback_stride=feedback_stride,
            feedback_width=self.config.data_width,
            hp_list_size=hp_list_size,
            jump_list_size=jump_list_size,
            index_width=index_width
        )
    
    def generate_memory_declarations(self, writer: RTLWriter):
        """Generate ROM declarations with FPGA synthesis attributes"""
        writer.write_section("Memory Declarations")
        writer.write_blank()
        
        data_range = f"{self.config.data_width-1}:0"
        index_range = f"{self.mem_config.index_width-1}:0"
        
        # Halfplane coefficients ROM
        writer.write_comment("Halfplane coefficients (distributed ROM)")
        signed_attr = "signed " if self.config.data_format.startswith('fixed') else ""
        writer.write_line(
            f"(* rom_style = \"distributed\" *) reg {signed_attr}[{data_range}] "
            f"halfplanes [0:{self.mem_config.halfplane_entries-1}];"
        )
        writer.write_blank()
        
        # Feedback coefficients ROM
        rom_style = "distributed" if self.mem_config.feedback_entries < 2048 else "block"
        writer.write_comment(f"Feedback coefficients ({rom_style} ROM)")
        writer.write_line(
            f"(* rom_style = \"{rom_style}\" *) reg {signed_attr}[{data_range}] "
            f"feedbacks [0:{self.mem_config.feedback_entries-1}];"
        )
        writer.write_blank()
        
        # BST structure ROMs
        writer.write_comment("BST structure (distributed ROM)")
        writer.write_line(
            f"(* rom_style = \"distributed\" *) reg [{index_range}] "
            f"hp_list [0:{self.mem_config.hp_list_size-1}];"
        )
        writer.write_line(
            f"(* rom_style = \"distributed\" *) reg [{index_range}] "
            f"jump_list [0:{self.mem_config.jump_list_size-1}];"
        )
        writer.write_blank()
    
    def generate_memory_initialization(self, writer: RTLWriter):
        """Generate memory initialization from .mem files"""
        writer.write_section("Memory Initialization")
        writer.write_blank()
        
        project = self.config.project_name
        
        writer.write_line("initial begin")
        writer.indent()
        writer.write_line(f"$readmemh(\"../include/{project}_halfplanes.mem\", halfplanes);")
        writer.write_line(f"$readmemh(\"../include/{project}_feedbacks.mem\", feedbacks);")
        writer.write_line(f"$readmemh(\"../include/{project}_hp_list.mem\", hp_list);")
        writer.write_line(f"$readmemh(\"../include/{project}_jump_list.mem\", jump_list);")
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
    
    def get_halfplane_address(self, node_idx: str, param_idx: int) -> str:
        """
        Generate address expression for halfplane coefficient access
        Pattern: node_idx * HALFPLANE_STRIDE + param_idx
        
        This is generic - works for any stride value from config
        """
        return f"{node_idx}*{self.mem_config.halfplane_stride}+{param_idx}"
    
    def get_feedback_address(self, node_idx: str, solution_idx: int, param_idx: int) -> str:
        """
        Generate address expression for feedback coefficient access
        Pattern: node_idx * SOL_PER_NODE + solution_idx * (N_PARAM+1) + param_idx
        
        Note: This assumes SOL_PER_NODE = N_SOLUTION * (N_PARAMETER + 1)
        which should be validated in config parser
        """
        elements_per_solution = self.config.n_parameters + 1
        solution_offset = solution_idx * elements_per_solution
        return f"{node_idx}*{self.mem_config.feedback_stride}+{solution_offset + param_idx}"
    
    def get_memory_config(self) -> MemoryConfig:
        """Get memory configuration object"""
        return self.mem_config