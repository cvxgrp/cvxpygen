"""
Floating-Point Memory Layout Generator
Manages ROM organization for BST data and feedback coefficients (FP format)
"""

import math
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter


@dataclass
class FloatMemoryConfig:
    """Memory configuration for floating-point BST LUT"""
    # Halfplane memory
    halfplane_size: int
    halfplane_stride: int
    halfplane_width: int
    
    # Feedback memory
    feedback_size: int
    feedback_stride: int
    feedback_width: int
    
    # BST structure memory
    hp_list_size: int
    jump_list_size: int
    index_width: int
    
    def __str__(self):
        return (f"FP Memory: HP={self.halfplane_size}x{self.halfplane_width}b, "
                f"FB={self.feedback_size}x{self.feedback_width}b, "
                f"BST={self.hp_list_size} nodes")


class FloatMemoryLayoutGenerator:
    """Generate memory declarations for floating-point implementation"""
    
    def __init__(self, config: PDQAPConfig):
        self.config = config
        self.mem_config = self._calculate_memory_config()
    
    def _calculate_memory_config(self) -> FloatMemoryConfig:
        """Calculate memory dimensions for FP"""
        # Same structure as fixed-point, but different data width
        halfplane_stride = self.config.n_parameters + 1
        halfplane_size = self.config.n_tree_nodes * halfplane_stride
        
        feedback_stride = self.config.n_solutions * (self.config.n_parameters + 1)
        feedback_size = self.config.n_tree_nodes * feedback_stride
        
        hp_list_size = self.config.n_tree_nodes
        jump_list_size = self.config.n_tree_nodes
        
        index_width = max(1, math.ceil(math.log2(self.config.n_tree_nodes)))
        
        return FloatMemoryConfig(
            halfplane_size=halfplane_size,
            halfplane_stride=halfplane_stride,
            halfplane_width=self.config.data_width,
            feedback_size=feedback_size,
            feedback_stride=feedback_stride,
            feedback_width=self.config.data_width,
            hp_list_size=hp_list_size,
            jump_list_size=jump_list_size,
            index_width=index_width
        )
    
    def generate_memory_declarations(self, writer: RTLWriter):
        """Generate ROM declarations for FP data"""
        writer.write_section("Memory Declarations (Floating Point)")
        writer.write_blank()
        
        data_range = f"{self.config.data_width-1}:0"
        index_range = f"{self.mem_config.index_width-1}:0"
        
        # Halfplane ROM - no signed attribute for FP
        writer.write_comment("Halfplane coefficients (FP format, distributed ROM)")
        writer.write_line(f"(* rom_style = \"distributed\" *) reg [{data_range}] halfplanes [0:{self.mem_config.halfplane_size-1}];")
        writer.write_blank()
        
        # Feedback ROM
        rom_style = "distributed" if self.mem_config.feedback_size < 2048 else "block"
        writer.write_comment(f"Feedback coefficients (FP format, {rom_style} ROM)")
        writer.write_line(f"(* rom_style = \"{rom_style}\" *) reg [{data_range}] feedbacks [0:{self.mem_config.feedback_size-1}];")
        writer.write_blank()
        
        # BST structure ROMs
        writer.write_comment("BST structure (distributed ROM)")
        writer.write_line(f"(* rom_style = \"distributed\" *) reg [{index_range}] hp_list [0:{self.mem_config.hp_list_size-1}];")
        writer.write_line(f"(* rom_style = \"distributed\" *) reg [{index_range}] jump_list [0:{self.mem_config.jump_list_size-1}];")
        writer.write_blank()
    
    def generate_memory_initialization(self, writer: RTLWriter):
        """Generate memory initialization from .mem files"""
        writer.write_section("Memory Initialization")
        writer.write_blank()
        
        project = self.config.project_name
        
        writer.write_line("initial begin")
        writer.indent()
        writer.write_line(f"$readmemh(\"../include/{project}_halfplanes_fp.mem\", halfplanes);")
        writer.write_line(f"$readmemh(\"../include/{project}_feedbacks_fp.mem\", feedbacks);")
        writer.write_line(f"$readmemh(\"../include/{project}_hp_list.mem\", hp_list);")
        writer.write_line(f"$readmemh(\"../include/{project}_jump_list.mem\", jump_list);")
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
    
    def get_halfplane_address(self, node_idx: str, param_idx: int) -> str:
        """Generate address calculation for halfplane coefficient"""
        return f"{node_idx}*{self.mem_config.halfplane_stride}+{param_idx}"
    
    def get_feedback_address(self, node_idx: str, solution_idx: int, param_idx: int) -> str:
        """Generate address calculation for feedback coefficient"""
        sol_stride = self.config.n_parameters + 1
        offset = solution_idx * sol_stride + param_idx
        return f"{node_idx}*{self.mem_config.feedback_stride}+{offset}"
    
    def get_memory_config(self) -> FloatMemoryConfig:
        """Get memory configuration"""
        return self.mem_config