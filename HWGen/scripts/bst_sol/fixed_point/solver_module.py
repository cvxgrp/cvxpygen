"""
Fixed-Point Solver Module Generator
Computes output solutions using feedback coefficients
"""

from typing import List
from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter
from .memory_layout import MemoryLayoutGenerator


class SolverGenerator:
    """Generate solution computation pipeline for fixed-point"""
    
    def __init__(self, config: PDQAPConfig, mem_gen: MemoryLayoutGenerator):
        self.config = config
        self.mem_gen = mem_gen
        self.mem_config = mem_gen.get_memory_config()
        
        # Solution pipeline configuration
        self.max_bst_depth = config.estimated_bst_depth
        self.sol_stages = 4  # prepare(1) + MAC(1) + offset(1) + output(1)
        
        # Calculate accumulator width (double precision for intermediate results)
        self.acc_width = 2 * config.data_width
    
    def generate_pipeline_registers(self, writer: RTLWriter):
        """Generate pipeline registers for solution computation"""
        writer.write_section("Solution Computation Pipeline Registers")
        writer.write_blank()
        
        # Halfplane index pipeline (not node ID!)
        writer.write_comment("Halfplane index pipeline for feedback addressing")
        writer.write_reg("hp_idx_sol", width=f"{self.mem_config.index_width-1}:0", array="0:2")
        writer.write_blank()
        
        # Parameter propagation
        writer.write_comment("Parameter pipeline for solver")
        for i in range(self.config.n_parameters):
            data_range = f"{self.config.data_width-1}:0"
            writer.write_reg(f"param{i}_sol", 
                           width=data_range,
                           signed=True,
                           array="0:1")
        writer.write_blank()
        
        # Solution intermediate results
        writer.write_comment("Solution computation temporaries")
        for sol_idx in range(self.config.n_solutions):
            writer.write_reg(f"sol_temp_{sol_idx}", 
                           width=f"{self.acc_width-1}:0", 
                           signed=True, 
                           array="0:1")
        writer.write_blank()
        
        # Solution output registers
        writer.write_comment("Solution outputs")
        for sol_idx in range(self.config.n_solutions):
            data_range = f"{self.config.data_width-1}:0"
            writer.write_reg(f"sol_out_{sol_idx}", width=data_range, signed=True)
        writer.write_blank()
        
        # Valid signal extension
        writer.write_comment("Valid signal for solution stages")
        total_stages = self.max_bst_depth + self.sol_stages
        writer.write_reg("valid_pipe_sol", array=f"0:{total_stages}")
        writer.write_blank()
    
    def generate_stage_prepare(self, writer: RTLWriter):
        """Generate Stage max_bst_depth+1: Prepare for solution computation"""
        stage_idx = self.max_bst_depth + 1
        prev_stage = self.max_bst_depth
        
        writer.write_comment(f"Stage {stage_idx}: Prepare for solution computation")
        writer.write_line(f"valid_pipe_sol[{stage_idx}] <= valid_pipe[{prev_stage}];")
        writer.write_blank()
        
        writer.begin_if(f"valid_pipe[{prev_stage}]")
        
        # Capture halfplane index (used to index feedback memory)
        writer.write_comment("Capture halfplane index for feedback addressing")
        writer.write_line(f"hp_idx_sol[0] <= hp_idx_pipe[{prev_stage}];")
        writer.write_blank()
        
        # Propagate parameters
        writer.write_comment("Propagate parameters")
        for param_idx in range(self.config.n_parameters):
            writer.write_line(f"param{param_idx}_sol[0] <= param{param_idx}_pipe[{prev_stage}];")
        
        writer.end_if()
        writer.write_blank()
    
    def generate_stage_mac(self, writer: RTLWriter):
        """Generate Stage max_bst_depth+2: Multiply-accumulate"""
        stage_idx = self.max_bst_depth + 2
        
        writer.write_comment(f"Stage {stage_idx}: Multiply-accumulate (w^T * x)")
        writer.write_line(f"valid_pipe_sol[{stage_idx}] <= valid_pipe_sol[{stage_idx-1}];")
        writer.write_blank()
        
        writer.begin_if(f"valid_pipe_sol[{stage_idx-1}]")
        
        # Propagate halfplane index
        writer.write_line("hp_idx_sol[1] <= hp_idx_sol[0];")
        writer.write_blank()
        
        # Propagate parameters
        for param_idx in range(self.config.n_parameters):
            writer.write_line(f"param{param_idx}_sol[1] <= param{param_idx}_sol[0];")
        writer.write_blank()
        
        # Compute dot products using generic addressing
        for sol_idx in range(self.config.n_solutions):
            writer.write_comment(f"Solution {sol_idx}: MAC")
            
            # Build MAC expression using hp_idx (not node_id)
            terms = []
            for param_idx in range(self.config.n_parameters):
                addr = self.mem_gen.get_feedback_address("hp_idx_sol[0]", sol_idx, param_idx)
                terms.append(
                    f"($signed(param{param_idx}_sol[0]) * $signed(feedbacks[{addr}]))"
                )
            
            mac_expr = ' + '.join(terms)
            writer.write_line(f"sol_temp_{sol_idx}[0] <= {mac_expr};")
            writer.write_blank()
        
        writer.end_if()
        writer.write_blank()
    
    def generate_stage_offset(self, writer: RTLWriter):
        """Generate Stage max_bst_depth+3: Add offset (bias term)"""
        stage_idx = self.max_bst_depth + 3
        
        writer.write_comment(f"Stage {stage_idx}: Add offset (bias)")
        writer.write_line(f"valid_pipe_sol[{stage_idx}] <= valid_pipe_sol[{stage_idx-1}];")
        writer.write_blank()
        
        writer.begin_if(f"valid_pipe_sol[{stage_idx-1}]")
        
        # Propagate halfplane index to stage 2
        writer.write_line("hp_idx_sol[2] <= hp_idx_sol[1];")
        writer.write_blank()
        
        # Add bias for each solution
        for sol_idx in range(self.config.n_solutions):
            # Bias stored at last position
            addr = self.mem_gen.get_feedback_address(
                "hp_idx_sol[1]", 
                sol_idx, 
                self.config.n_parameters
            )
            
            # Scale bias to match accumulator precision (use logical shift)
            frac_bits = self.config.feedback_frac_bits or 0
            writer.write_comment(f"Solution {sol_idx}: Add bias (scaled)")
            writer.write_line(
                f"sol_temp_{sol_idx}[1] <= sol_temp_{sol_idx}[0] + "
                f"($signed(feedbacks[{addr}]) << {frac_bits});"
            )
        
        writer.end_if()
        writer.write_blank()
    
    def generate_stage_output(self, writer: RTLWriter):
        """Generate Stage max_bst_depth+4: Output with truncation"""
        stage_idx = self.max_bst_depth + 4
        
        writer.write_comment(f"Stage {stage_idx}: Output (truncate to output format)")
        writer.write_line(f"valid_pipe_sol[{stage_idx}] <= valid_pipe_sol[{stage_idx-1}];")
        writer.write_blank()
        
        writer.begin_if(f"valid_pipe_sol[{stage_idx-1}]")
        
        # Calculate truncation range
        # Accumulator format: signed [31:0] with FRAC fractional bits
        # After MAC: result is at [2*FRAC] position
        # After bias add: still at [FRAC] position (bias is pre-shifted)
        # Extract: [FRAC+WIDTH-1 : FRAC]
        frac_bits = self.config.feedback_frac_bits or 0
        output_width = self.config.data_width
        
        # Correct truncation: keep [FRAC+WIDTH-1 : FRAC]
        high_bit = frac_bits + output_width - 1
        low_bit = frac_bits
        
        writer.write_comment(f"Truncate [{high_bit}:{low_bit}] to output width")
        for sol_idx in range(self.config.n_solutions):
            writer.write_line(f"sol_out_{sol_idx} <= sol_temp_{sol_idx}[1][{high_bit}:{low_bit}];")
        
        writer.end_if()
        writer.write_blank()
    
    def generate_reset_logic(self, writer: RTLWriter):
        """Generate reset logic for solver pipeline"""
        writer.write_comment("Reset solution pipeline")
        
        total_stages = self.max_bst_depth + self.sol_stages
        writer.begin_for("i = 0", f"i <= {total_stages}", "i = i + 1")
        writer.write_line("valid_pipe_sol[i] <= 0;")
        writer.end_for()
        writer.write_blank()
        
        # Reset halfplane index pipeline (0:2, three stages)
        writer.begin_for("j = 0", "j <= 2", "j = j + 1")
        writer.write_line("hp_idx_sol[j] <= 0;")
        writer.end_for()
        writer.write_blank()
        
        # Reset parameter and temp pipelines (0:1, two stages)
        writer.begin_for("j = 0", "j <= 1", "j = j + 1")
        for param_idx in range(self.config.n_parameters):
            writer.write_line(f"param{param_idx}_sol[j] <= 0;")
        for sol_idx in range(self.config.n_solutions):
            writer.write_line(f"sol_temp_{sol_idx}[j] <= 0;")
        writer.end_for()
        writer.write_blank()
        
        # Reset output registers
        for sol_idx in range(self.config.n_solutions):
            writer.write_line(f"sol_out_{sol_idx} <= 0;")
        writer.write_blank()