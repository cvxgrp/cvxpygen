"""
Floating-Point Solver Module Generator
Computes output solutions using FP feedback coefficients with MAC
"""

import math
from typing import Dict
from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter
from .memory_layout_float import FloatMemoryLayoutGenerator


class FloatSolverGenerator:
    """Generate solution computation pipeline for floating-point using MAC"""
    
    def __init__(self, config: PDQAPConfig, mem_gen: FloatMemoryLayoutGenerator):
        self.config = config
        self.mem_gen = mem_gen
        self.mem_config = mem_gen.get_memory_config()
        
        self.max_bst_depth = config.estimated_bst_depth
        
        # FP operator latencies
        self.mult_latency = config.fp_mult_latency
        self.add_latency = config.fp_add_latency
        
        # *** FIXED: Remove unnecessary +1 ***
        # Xilinx FP IP latency already includes all pipeline stages
        self.actual_mult_latency = self.mult_latency  # No +1!
        self.actual_add_latency = self.add_latency
        
        # MAC latency for solution (serial chain)
        # MAC = MULT + (N_PARAMS - 1) * ADD
        self.mac_latency = (self.actual_mult_latency + 
                           (config.n_parameters - 1) * self.actual_add_latency)
        
        # Solution = MAC + offset ADD
        self.solution_latency = self.mac_latency + self.actual_add_latency
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate solver configuration"""
        if self.config.n_solutions < 1:
            raise ValueError("Number of solutions must be >= 1")
        if self.config.n_parameters < 1:
            raise ValueError("Number of parameters must be >= 1")
    
    def get_pipeline_info(self) -> Dict[str, int]:
        """Get pipeline timing information"""
        return {
            'mac_latency': self.mac_latency,
            'solution_latency': self.solution_latency,
            'mult_latency': self.mult_latency,
            'add_latency': self.add_latency,
            'actual_mult_latency': self.actual_mult_latency,  # For debug
            'actual_add_latency': self.actual_add_latency
        }
    
    def generate_solution_pipeline(self, writer: RTLWriter):
        """Generate complete solution computation pipeline"""
        writer.write_separator("Solution Computation (MAC)")
        writer.write_blank()
        
        # Add MAC latency constant with correct formula
        writer.write_comment("Solution computation uses MAC (sequential adder chain)")
        writer.write_comment(f"MAC latency = MULT + (N_PARAMS - 1) * ADD")
        writer.write_comment(f"  = {self.actual_mult_latency} + ({self.config.n_parameters} - 1) * {self.actual_add_latency} = {self.mac_latency}")
        writer.write_line(f"localparam MAC_LATENCY = ACTUAL_MULT_LATENCY + (N_PARAMS - 1) * ACTUAL_ADD_LATENCY;")
        writer.write_blank()
        
        writer.write_comment("Solution latency = MAC + final offset ADD")
        writer.write_line(f"localparam SOLUTION_LATENCY = MAC_LATENCY + ACTUAL_ADD_LATENCY;")
        writer.write_blank()
        
        # S1 stage registers
        self._generate_s1_stage(writer)
        
        # Solution computation wires
        self._generate_solution_wires(writer)
        
        # Generate per-solution computation
        self._generate_solution_instances(writer)
        
        # Output stage
        self._generate_output_stage(writer)
    
    def _generate_s1_stage(self, writer: RTLWriter):
        """Generate S1 stage: prepare for solution computation"""
        writer.write_comment("S1 stage: prepare for solution computation")
        writer.write_reg("s1_params", width="[DATA_WIDTH-1:0]", array="0:N_PARAMS-1")
        writer.write_reg("s1_hp_idx", width="[7:0]")
        writer.write_reg("s1_valid")
        writer.write_blank()
        
        writer.begin_always("posedge clk or negedge rst_n")
        writer.begin_if("!rst_n")
        writer.write_line("s1_valid <= 1'b0;")
        writer.write_line("s1_hp_idx <= 8'd0;")
        writer.write_line("for (i = 0; i < N_PARAMS; i = i + 1) begin")
        writer.indent()
        writer.write_line("s1_params[i] <= {DATA_WIDTH{1'b0}};")
        writer.dedent()
        writer.write_line("end")
        writer.begin_else_if("final_leaf_valid")
        # Read from FIFO based on final_fifo_ptr
        for i in range(self.config.n_parameters):
            writer.write_line(f"s1_params[{i}] <= param_fifo_{i}[final_fifo_ptr];")
        writer.write_line("s1_hp_idx <= final_hp_idx;")
        writer.write_line("s1_valid <= 1'b1;")
        writer.begin_else()
        writer.write_line("s1_valid <= 1'b0;")
        writer.end_if()
        writer.end_always()
        writer.write_blank()
    
    def _generate_solution_wires(self, writer: RTLWriter):
        """Generate wires for solution computation"""
        writer.write_comment("Solution results")
        writer.write_wire("sol_result", width="[DATA_WIDTH-1:0]", array="0:N_SOLUTIONS-1")
        writer.write_wire("sol_valid", array="0:N_SOLUTIONS-1")
        writer.write_blank()
        
        # Optional: delayed offset for debug
        writer.write_comment("Delayed offset (for debug)")
        writer.write_reg("s1_offset_delayed", width="[DATA_WIDTH-1:0]", array="0:N_SOLUTIONS-1")
        writer.write_blank()
    
    def _generate_solution_instances(self, writer: RTLWriter):
        """Generate solution computation instances"""
        writer.write_comment("Generate solution computations")
        writer.begin_generate()
        writer.write_line("genvar sol_idx;")
        writer.write_line("for (sol_idx = 0; sol_idx < N_SOLUTIONS; sol_idx = sol_idx + 1) begin : sol_gen")
        writer.indent()
        writer.write_blank()
        
        # Get feedback coefficients
        self._generate_feedback_access(writer)
        
        # Pack for MAC
        self._generate_mac_packing(writer)
        
        # MAC instance
        self._generate_mac_instance(writer)
        
        # Offset delay
        self._generate_offset_delay(writer)
        
        # Add offset
        self._generate_offset_add(writer)
        
        writer.dedent()
        writer.write_line("end")
        writer.end_generate()
        writer.write_blank()
    
    def _generate_feedback_access(self, writer: RTLWriter):
        """Generate feedback coefficient access"""
        writer.write_comment("Get feedback coefficients for this solution")
        writer.write_comment("Layout: feedbacks[hp_idx][sol_idx][param_idx]")
        
        # Declare wires for scales and offset
        writer.write_wire("fb_scales", width="[DATA_WIDTH-1:0]", array="0:N_PARAMS-1")
        writer.write_wire("fb_offset", width="[DATA_WIDTH-1:0]")
        writer.write_blank()
        
        # Generate access logic
        writer.begin_generate()
        writer.write_line("genvar fb_idx;")
        writer.write_line("for (fb_idx = 0; fb_idx < N_PARAMS; fb_idx = fb_idx + 1) begin : fb_scale_gen")
        writer.indent()
        writer.write_line(
            "assign fb_scales[fb_idx] = feedbacks[s1_hp_idx * (N_PARAMS+1) * N_SOLUTIONS + "
            "sol_idx * (N_PARAMS+1) + fb_idx];"
        )
        writer.dedent()
        writer.write_line("end")
        writer.end_generate()
        
        writer.write_line(
            "assign fb_offset = feedbacks[s1_hp_idx * (N_PARAMS+1) * N_SOLUTIONS + "
            "sol_idx * (N_PARAMS+1) + N_PARAMS];"
        )
        writer.write_blank()
    
    def _generate_mac_packing(self, writer: RTLWriter):
        """Generate MAC input packing"""
        writer.write_comment("Pack parameters and feedback scales for MAC")
        writer.write_wire("mac_a_packed", width="[N_PARAMS*DATA_WIDTH-1:0]")
        writer.write_wire("mac_b_packed", width="[N_PARAMS*DATA_WIDTH-1:0]")
        writer.write_blank()
        
        # Build packed signals (MSB = highest index)
        # Standard Verilog concatenation: {highest, ..., lowest}
        a_parts = [f"s1_params[{i}]" for i in range(self.config.n_parameters-1, -1, -1)]
        b_parts = [f"fb_scales[{i}]" for i in range(self.config.n_parameters-1, -1, -1)]
        
        writer.write_line(f"assign mac_a_packed = {{{', '.join(a_parts)}}};")
        writer.write_line(f"assign mac_b_packed = {{{', '.join(b_parts)}}};")
        writer.write_blank()
    
    def _generate_mac_instance(self, writer: RTLWriter):
        """Generate MAC instance"""
        writer.write_comment("MAC: compute dot product of params and feedback scales")
        writer.write_wire("mac_result", width="[DATA_WIDTH-1:0]")
        writer.write_wire("mac_valid")
        writer.write_blank()
        
        writer.write_line("top_mac_module #(")
        writer.indent()
        writer.write_line(".N_TERMS(N_PARAMS)")
        writer.dedent()
        writer.write_line(") mac (")
        writer.indent()
        writer.write_line(".clk(clk),")
        writer.write_line(".rst_n(rst_n),")
        writer.write_line(".a_packed(mac_a_packed),")
        writer.write_line(".b_packed(mac_b_packed),")
        writer.write_line(".valid_in(s1_valid),")
        writer.write_line(".ready_in(),")
        writer.write_line(".metadata_in({METADATA_WIDTH{1'b0}}),")
        writer.write_line(".result(mac_result),")
        writer.write_line(".valid_out(mac_valid),")
        writer.write_line(".ready_out(1'b1),")
        writer.write_line(".metadata_out()")
        writer.dedent()
        writer.write_line(");")
        writer.write_blank()
    
    def _generate_offset_delay(self, writer: RTLWriter):
        """Generate offset delay lines"""
        writer.write_comment("Delay offset to align with MAC output")
        writer.write_wire("offset_delayed", width="[DATA_WIDTH-1:0]")
        writer.write_wire("offset_valid_delayed")
        writer.write_blank()
        
        # Data delay
        writer.write_line("simple_delay_line #(")
        writer.indent()
        writer.write_line(".DATA_WIDTH(DATA_WIDTH),")
        writer.write_line(".DEPTH(MAC_LATENCY)")
        writer.dedent()
        writer.write_line(") delay_offset (")
        writer.indent()
        writer.write_line(".clk(clk),")
        writer.write_line(".rst_n(rst_n),")
        writer.write_line(".data_in(fb_offset),")
        writer.write_line(".data_out(offset_delayed)")
        writer.dedent()
        writer.write_line(");")
        writer.write_blank()
        
        # Valid delay
        writer.write_line("valid_delay_line #(")
        writer.indent()
        writer.write_line(".DEPTH(MAC_LATENCY)")
        writer.dedent()
        writer.write_line(") delay_offset_valid (")
        writer.indent()
        writer.write_line(".clk(clk),")
        writer.write_line(".rst_n(rst_n),")
        writer.write_line(".valid_in(s1_valid),")
        writer.write_line(".valid_out(offset_valid_delayed)")
        writer.dedent()
        writer.write_line(");")
        writer.write_blank()
        
        # Store delayed offset for debug
        writer.write_comment("Store delayed offset for debug")
        writer.write_line("always @(posedge clk) begin")
        writer.indent()
        writer.write_line("if (offset_valid_delayed) begin")
        writer.indent()
        writer.write_line("s1_offset_delayed[sol_idx] <= offset_delayed;")
        writer.dedent()
        writer.write_line("end")
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
    
    def _generate_offset_add(self, writer: RTLWriter):
        """Generate offset addition"""
        writer.write_comment("Add offset to MAC result")
        writer.write_wire("add_offset_valid")
        writer.write_line("assign add_offset_valid = mac_valid && offset_valid_delayed;")
        writer.write_blank()
        
        writer.write_line("fp_add #(")
        writer.indent()
        writer.write_line(".DATA_WIDTH(DATA_WIDTH),")
        writer.write_line(".METADATA_WIDTH(METADATA_WIDTH)")
        writer.dedent()
        writer.write_line(") add_offset (")
        writer.indent()
        writer.write_line(".aclk(clk),")
        writer.write_line(".aresetn(rst_n),")
        writer.write_line(".s_axis_a_tdata(mac_result),")
        writer.write_line(".s_axis_a_tvalid(add_offset_valid),")
        writer.write_line(".s_axis_b_tdata(offset_delayed),")
        writer.write_line(".s_axis_b_tvalid(add_offset_valid),")
        writer.write_line(".m_axis_result_tdata(sol_result[sol_idx]),")
        writer.write_line(".m_axis_result_tvalid(sol_valid[sol_idx]),")
        writer.write_line(".m_axis_result_tready(1'b1)")
        writer.dedent()
        writer.write_line(");")
        writer.write_blank()
    
    def _generate_output_stage(self, writer: RTLWriter):
        """Generate output stage"""
        writer.write_separator("Output Stage")
        writer.write_blank()
        
        # Output registers
        writer.write_comment("Output registers")
        writer.write_reg("output_sol", width="[DATA_WIDTH-1:0]", array="0:N_SOLUTIONS-1")
        writer.write_reg("output_valid_reg")
        writer.write_blank()
        
        # Declare integer for loop
        writer.write_line("integer idx;")
        writer.write_blank()
        
        # Output assignment logic
        writer.begin_always("posedge clk or negedge rst_n")
        writer.write_line(": output_stage")  # Named block for clarity
        writer.begin_if("!rst_n")
        writer.write_line("output_valid_reg <= 1'b0;")
        writer.write_line("for (idx = 0; idx < N_SOLUTIONS; idx = idx + 1) begin")
        writer.indent()
        writer.write_line("output_sol[idx] <= {DATA_WIDTH{1'b0}};")
        writer.dedent()
        writer.write_line("end")
        writer.begin_else()
        # Assign all solutions
        for i in range(self.config.n_solutions):
            writer.write_line(f"output_sol[{i}] <= sol_result[{i}];")
        writer.write_line("output_valid_reg <= sol_valid[0];")
        writer.write_blank()
        
        # Debug print
        writer.write_line("if (sol_valid[0]) begin")
        writer.indent()
        writer.write_line('$display("[%0t] [Out] #%0d, s0=0x%h, s1=0x%h, s2=0x%h, s3=0x%h, s4=0x%h",')
        writer.indent()
        writer.write_line("$time, output_count, sol_result[0], sol_result[1],")
        writer.write_line("sol_result[2], sol_result[3], sol_result[4]);")
        writer.dedent()
        writer.dedent()
        writer.write_line("end")
        
        writer.end_if()
        writer.end_always()
        writer.write_blank()
        
        # FIFO read pointer update
        writer.write_comment("Update FIFO read pointer")
        writer.begin_always("posedge clk or negedge rst_n")
        writer.begin_if("!rst_n")
        writer.write_line("param_fifo_rd_ptr <= 8'd0;")
        writer.begin_else_if("output_valid_reg")
        writer.write_line("param_fifo_rd_ptr <= param_fifo_rd_ptr + 1'd1;")
        writer.end_if()
        writer.end_always()
        writer.write_blank()
    
    def generate_output_assignments(self, writer: RTLWriter):
        """Generate final output assignments"""
        writer.write_separator("Output Assignments")
        writer.write_blank()
        
        # Assign individual solution outputs
        for i in range(self.config.n_solutions):
            writer.write_line(f"assign sol_out_{i} = output_sol[{i}];")
        
        writer.write_line("assign valid_out = output_valid_reg;")
        writer.write_line("assign ready_out = 1'b1;")
        writer.write_blank()