"""
Fixed-Point BST Search Module Generator
Implements binary tree traversal with halfplane decisions
"""

from typing import List
from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter
from .memory_layout import MemoryLayoutGenerator


class BSTSearchGenerator:
    """Generate BST traversal pipeline for fixed-point"""
    
    def __init__(self, config: PDQAPConfig, mem_gen: MemoryLayoutGenerator):
        self.config = config
        self.mem_gen = mem_gen
        self.mem_config = mem_gen.get_memory_config()
        
        # Pipeline configuration
        self.max_bst_depth = config.estimated_bst_depth
        self.index_width = self.mem_config.index_width
    
    def generate_pipeline_registers(self, writer: RTLWriter):
        """Generate pipeline registers for BST traversal"""
        writer.write_section("BST Traversal Pipeline Registers")
        writer.write_blank()
        
        # Parameter propagation pipeline
        writer.write_comment("Parameter pipeline")
        for i in range(self.config.n_parameters):
            data_range = f"{self.config.data_width-1}:0"
            writer.write_reg(f"param{i}_pipe", 
                           width=data_range,
                           signed=True,
                           array=f"0:{self.max_bst_depth}")
        writer.write_blank()
        
        # BST state pipeline
        writer.write_comment("BST traversal state")
        index_range = f"{self.index_width-1}:0"
        writer.write_reg("current_id_pipe", width=index_range, array=f"0:{self.max_bst_depth}")
        writer.write_reg("next_id_pipe", width=index_range, array=f"0:{self.max_bst_depth}")
        writer.write_reg("hp_idx_pipe", width=index_range, array=f"0:{self.max_bst_depth}")
        writer.write_reg("traversal_done_pipe", array=f"0:{self.max_bst_depth}")
        writer.write_blank()
        
        # Valid signal pipeline
        writer.write_comment("Valid signal pipeline")
        writer.write_reg("valid_pipe", array=f"0:{self.max_bst_depth}")
        writer.write_blank()
    
    def generate_combinational_logic(self, writer: RTLWriter):
        """Generate combinational logic for halfplane evaluation"""
        writer.write_section("Combinational Logic (Halfplane Decision)")
        writer.write_blank()
        
        # Calculate accumulator width (double precision for intermediate results)
        acc_width = 2 * self.config.data_width
        
        writer.write_comment("Halfplane computation temporaries")
        writer.write_reg("hp_val", width=f"{acc_width-1}:0", signed=True)
        writer.write_reg("hp_thresh", width=f"{acc_width-1}:0", signed=True)
        writer.write_reg("decision")
        writer.write_reg("new_id", width=f"{self.index_width-1}:0")
        writer.write_blank()
    
    def generate_stage0_input(self, writer: RTLWriter):
        """Generate Stage 0: Input capture"""
        writer.write_comment("Stage 0: Input")
        writer.write_line("valid_pipe[0] <= valid_in;")
        writer.write_blank()
        
        writer.begin_if("valid_in")
        
        # Capture input parameters
        for i in range(self.config.n_parameters):
            writer.write_line(f"param{i}_pipe[0] <= param_in_{i};")
        writer.write_blank()
        
        # Initialize BST traversal at root
        writer.write_comment("Initialize at BST root")
        writer.write_line("current_id_pipe[0] <= 0;")
        writer.write_line("next_id_pipe[0] <= jump_list[0];")
        writer.write_line("hp_idx_pipe[0] <= hp_list[0];")
        writer.write_line("traversal_done_pipe[0] <= (jump_list[0] == 0);")
        
        writer.end_if()
        writer.write_blank()
    
    def generate_bst_traversal_stages(self, writer: RTLWriter):
        """Generate BST traversal logic for stages 1 to MAX_BST_DEPTH"""
        
        writer.write_comment(f"Stages 1 to {self.max_bst_depth}: BST traversal")
        writer.begin_for("i = 0", f"i < {self.max_bst_depth}", "i = i + 1")
        
        # Pipeline control
        writer.write_line("valid_pipe[i+1] <= valid_pipe[i];")
        
        # Propagate parameters
        for param_idx in range(self.config.n_parameters):
            writer.write_line(f"param{param_idx}_pipe[i+1] <= param{param_idx}_pipe[i];")
        writer.write_blank()
        
        # Conditional logic
        writer.begin_if("valid_pipe[i]")
        
        # If already at leaf, pass through
        writer.begin_if("traversal_done_pipe[i]")
        writer.write_comment("Pass through if at leaf")
        writer.write_line("current_id_pipe[i+1] <= current_id_pipe[i];")
        writer.write_line("next_id_pipe[i+1] <= next_id_pipe[i];")
        writer.write_line("hp_idx_pipe[i+1] <= hp_idx_pipe[i];")
        writer.write_line("traversal_done_pipe[i+1] <= 1;")
        
        # Otherwise, evaluate halfplane
        writer.begin_else()
        
        # Halfplane computation using generic address
        writer.write_comment("Halfplane dot product: w^T * x")
        hp_terms = []
        for param_idx in range(self.config.n_parameters):
            addr = self.mem_gen.get_halfplane_address("hp_idx_pipe[i]", param_idx)
            hp_terms.append(
                f"($signed(param{param_idx}_pipe[i]) * $signed(halfplanes[{addr}]))"
            )
        
        hp_dot = ' + '.join(hp_terms)
        writer.write_line(f"hp_val = {hp_dot};")
        writer.write_blank()
        
        # Threshold comparison (use logical shift)
        frac_bits = self.config.feedback_frac_bits or 0
        thresh_addr = self.mem_gen.get_halfplane_address("hp_idx_pipe[i]", self.config.n_parameters)
        writer.write_comment("Scale threshold to match accumulator width")
        writer.write_line(f"hp_thresh = $signed(halfplanes[{thresh_addr}]) << {frac_bits};")
        writer.write_blank()
        
        # Decision
        writer.write_line("decision = (hp_val <= hp_thresh);")
        writer.write_blank()
        
        # Compute next node
        writer.write_comment("Compute next node ID")
        writer.write_line(
            f"new_id = next_id_pipe[i] + (decision ? {self.mem_config.index_width}'d1 : {self.mem_config.index_width}'d0);"
        )
        writer.write_blank()
        
        # Update state
        writer.write_comment("Update pipeline")
        writer.write_line("current_id_pipe[i+1] <= new_id;")
        writer.write_line("next_id_pipe[i+1] <= new_id + jump_list[new_id];")
        writer.write_line("hp_idx_pipe[i+1] <= hp_list[new_id];")
        writer.write_line("traversal_done_pipe[i+1] <= (jump_list[new_id] == 0);")
        
        writer.end_if()
        
        # Invalid data handling
        writer.begin_else()
        writer.write_comment("Reset if invalid")
        writer.write_line("current_id_pipe[i+1] <= 0;")
        writer.write_line("next_id_pipe[i+1] <= 0;")
        writer.write_line("hp_idx_pipe[i+1] <= 0;")
        writer.write_line("traversal_done_pipe[i+1] <= 0;")
        writer.end_if()
        
        writer.end_for()
        writer.write_blank()
    
    def generate_reset_logic(self, writer: RTLWriter):
        """Generate reset logic for BST pipeline"""
        writer.write_comment("Reset BST pipeline")
        writer.begin_for("i = 0", f"i <= {self.max_bst_depth}", "i = i + 1")
        writer.write_line("valid_pipe[i] <= 0;")
        writer.write_line("current_id_pipe[i] <= 0;")
        writer.write_line("next_id_pipe[i] <= 0;")
        writer.write_line("hp_idx_pipe[i] <= 0;")
        writer.write_line("traversal_done_pipe[i] <= 0;")
        writer.end_for()
        writer.write_blank()
        
        writer.begin_for("j = 0", f"j <= {self.max_bst_depth}", "j = j + 1")
        for param_idx in range(self.config.n_parameters):
            writer.write_line(f"param{param_idx}_pipe[j] <= 0;")
        writer.end_for()
        writer.write_blank()