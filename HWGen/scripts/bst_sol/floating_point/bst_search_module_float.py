"""
Floating-Point BST Search Module Generator
Generates BST traversal pipeline using adder tree for dot product
"""

import math
from typing import Dict, List
from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter
from .memory_layout_float import FloatMemoryLayoutGenerator


class FloatBSTSearchGenerator:
    """Generate BST search pipeline for floating-point"""
    
    def __init__(self, config: PDQAPConfig, mem_gen: FloatMemoryLayoutGenerator):
        self.config = config
        self.mem_gen = mem_gen
        self.mem_config = mem_gen.get_memory_config()
        
        self.max_bst_depth = config.estimated_bst_depth
        
        # FP operator latencies from config
        self.mult_latency = config.fp_mult_latency
        self.add_latency = config.fp_add_latency
        self.comp_latency = config.fp_comp_latency
        
        # Actual latencies (no adjustment)
        self.actual_mult_latency = self.mult_latency
        self.actual_add_latency = self.add_latency
        
        # Build adder tree structure
        self.tree_structure = self._build_adder_tree_structure(config.n_parameters)
        self.tree_depth = len(self.tree_structure)
        
        # Calculate dot product latency: MULT + tree_depth * ADD
        self.dot_product_latency = (self.actual_mult_latency + 
                                   self.tree_depth * self.actual_add_latency)
        
        # Comparison = dot_product + comp
        self.comparison_latency = self.dot_product_latency + self.comp_latency
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate BST configuration"""
        if self.max_bst_depth < 1:
            raise ValueError("BST depth must be >= 1")
        if self.config.n_parameters < 1:
            raise ValueError("Number of parameters must be >= 1")
    
    def _build_adder_tree_structure(self, n_params: int) -> List[List[Dict]]:
        """Build balanced adder tree with passthrough delay alignment"""
        tree = []
        current_count = n_params
        level_idx = 0
        
        while current_count > 1:
            level = []
            num_pairs = current_count // 2
            has_odd = (current_count % 2 == 1)
            
            # Paired adders
            for i in range(num_pairs):
                left_idx = 2 * i
                right_idx = 2 * i + 1
                
                if level_idx == 0:
                    left_signal = f'd_mult_out[depth][{left_idx}]'
                    left_valid = f'd_mult_valid[depth][{left_idx}]'
                    right_signal = f'd_mult_out[depth][{right_idx}]'
                    right_valid = f'd_mult_valid[depth][{right_idx}]'
                else:
                    prev_level = level_idx
                    left_signal = f'd_add_l{prev_level}_out[depth][{left_idx}]'
                    left_valid = f'd_add_l{prev_level}_valid[depth][{left_idx}]'
                    right_signal = f'd_add_l{prev_level}_out[depth][{right_idx}]'
                    right_valid = f'd_add_l{prev_level}_valid[depth][{right_idx}]'
                
                level.append({
                    'index': i,
                    'left_signal': left_signal,
                    'left_valid': left_valid,
                    'right_signal': right_signal,
                    'right_valid': right_valid,
                    'is_passthrough': False,
                    'needs_delay': False
                })
            
            # Passthrough for odd element (needs delay alignment)
            if has_odd:
                odd_idx = current_count - 1
                passthrough_output_idx = num_pairs
                
                if level_idx == 0:
                    source_signal = f'd_mult_out[depth][{odd_idx}]'
                    source_valid = f'd_mult_valid[depth][{odd_idx}]'
                else:
                    prev_level = level_idx
                    source_signal = f'd_add_l{prev_level}_out[depth][{odd_idx}]'
                    source_valid = f'd_add_l{prev_level}_valid[depth][{odd_idx}]'
                
                level.append({
                    'index': passthrough_output_idx,
                    'left_signal': source_signal,
                    'left_valid': source_valid,
                    'right_signal': None,
                    'right_valid': None,
                    'is_passthrough': True,
                    'needs_delay': True
                })
            
            tree.append(level)
            current_count = num_pairs + (1 if has_odd else 0)
            level_idx += 1
        
        return tree
    
    def get_pipeline_info(self) -> Dict[str, int]:
        """Get pipeline timing information"""
        return {
            'dot_product_latency': self.dot_product_latency,
            'comparison_latency': self.comparison_latency,
            'tree_depth': self.tree_depth,
            'mult_latency': self.mult_latency,
            'add_latency': self.add_latency,
            'comp_latency': self.comp_latency,
            'actual_mult_latency': self.actual_mult_latency,
            'actual_add_latency': self.actual_add_latency
        }
    
    def generate_constants(self, writer: RTLWriter):
        """Generate timing constants"""
        writer.write_separator("Timing Constants")
        writer.write_blank()
        
        writer.write_comment("FP operation latencies")
        writer.write_line(f"localparam FP_MULT_LATENCY = `FP_MULT_LATENCY;")
        writer.write_line(f"localparam FP_ADD_LATENCY = `FP_ADD_LATENCY;")
        writer.write_line(f"localparam FP_COMP_LATENCY = `FP_COMP_LATENCY;")
        writer.write_blank()
        
        writer.write_comment("Actual latencies (no adjustment)")
        writer.write_line(f"localparam ACTUAL_MULT_LATENCY = FP_MULT_LATENCY;")
        writer.write_line(f"localparam ACTUAL_ADD_LATENCY = FP_ADD_LATENCY;")
        writer.write_blank()
        
        writer.write_comment("Adder tree levels")
        writer.write_line(f"localparam ADDER_TREE_LEVELS = {self.tree_depth};")
        writer.write_blank()
        
        writer.write_comment(f"Dot product latency: MULT + TREE * ADD = {self.dot_product_latency}")
        writer.write_line(f"localparam DOT_PRODUCT_LATENCY = ACTUAL_MULT_LATENCY + ADDER_TREE_LEVELS * ACTUAL_ADD_LATENCY;")
        writer.write_blank()
        
        writer.write_comment("Comparison latency: DOT + COMP")
        writer.write_line(f"localparam COMPARISON_LATENCY = DOT_PRODUCT_LATENCY + FP_COMP_LATENCY;")
        writer.write_blank()
        
        # MAC latency
        writer.write_comment("MAC latency")
        writer.write_line(f"localparam MAC_LATENCY = ACTUAL_MULT_LATENCY + (N_PARAMS - 1) * ACTUAL_ADD_LATENCY;")
        writer.write_line(f"localparam SOLUTION_LATENCY = MAC_LATENCY + ACTUAL_ADD_LATENCY;")
        writer.write_blank()
    
    def generate_input_stage(self, writer: RTLWriter):
        """Generate input stage with FIFO"""
        writer.write_separator("Input Stage")
        writer.write_blank()
        
        # Input registers
        writer.write_comment("Input registers")
        writer.write_reg("stage0_params", width="[DATA_WIDTH-1:0]", array="0:N_PARAMS-1")
        writer.write_reg("stage0_valid")
        writer.write_blank()
        
        # Loop variable declaration
        writer.write_comment("Loop variable for input stage")
        writer.write_line("integer i;")
        writer.write_blank()
        
        # Input assignment
        writer.begin_always("posedge clk or negedge rst_n")
        writer.begin_if("!rst_n")
        writer.write_line("stage0_valid <= 1'b0;")
        writer.write_line("for (i = 0; i < N_PARAMS; i = i + 1) begin")
        writer.indent()
        writer.write_line("stage0_params[i] <= {DATA_WIDTH{1'b0}};")
        writer.dedent()
        writer.write_line("end")
        writer.begin_else()
        writer.write_line("for (i = 0; i < N_PARAMS; i = i + 1) begin")
        writer.indent()
        writer.write_line("case (i)")
        for i in range(self.config.n_parameters):
            writer.write_line(f"{i}: stage0_params[i] <= param_in_{i};")
        writer.write_line("default: stage0_params[i] <= {DATA_WIDTH{1'b0}};")
        writer.write_line("endcase")
        writer.dedent()
        writer.write_line("end")
        writer.write_line("stage0_valid <= valid_in;")
        writer.end_if()
        writer.end_always()
        writer.write_blank()
        
        # Parameter FIFO (128-deep)
        writer.write_comment("Parameter FIFO (128-deep)")
        writer.write_reg("param_fifo", width="[DATA_WIDTH-1:0]", array="0:N_PARAMS-1][0:127")
        writer.write_reg("param_fifo_wr_ptr", width="[7:0]")
        writer.write_reg("param_fifo_rd_ptr", width="[7:0]")
        writer.write_blank()
        
        # FIFO write logic
        writer.begin_always("posedge clk or negedge rst_n")
        writer.begin_if("!rst_n")
        writer.write_line("param_fifo_wr_ptr <= 8'd0;")
        writer.begin_else_if("stage0_valid")
        writer.write_line("for (i = 0; i < N_PARAMS; i = i + 1) begin")
        writer.indent()
        writer.write_line("param_fifo[i][param_fifo_wr_ptr] <= stage0_params[i];")
        writer.dedent()
        writer.write_line("end")
        writer.write_line("param_fifo_wr_ptr <= param_fifo_wr_ptr + 1'd1;")
        writer.end_if()
        writer.end_always()
        writer.write_blank()
    
    def generate_bst_pipeline(self, writer: RTLWriter):
        """Generate BST traversal pipeline"""
        writer.write_separator("BST Traversal Pipeline")
        writer.write_blank()
        
        # Pipeline state
        self._generate_pipeline_state(writer)
        
        # Depth 0 initialization (OUTSIDE generate)
        self._generate_depth0_init(writer)
        
        # Pipeline instances (INSIDE generate)
        self._generate_pipeline_instances(writer)
        
        # Next depth logic (SEPARATE generate)
        self._generate_next_depth_logic(writer)
    
    def _generate_pipeline_state(self, writer: RTLWriter):
        """Generate pipeline state registers and wires"""
        writer.write_comment("Pipeline state")
        writer.write_reg("d_node_id", width="[7:0]", array="0:MAX_DEPTH-1")
        writer.write_reg("d_hp_idx", width="[7:0]", array="0:MAX_DEPTH-1")
        writer.write_reg("d_fifo_ptr", width="[7:0]", array="0:MAX_DEPTH-1")
        writer.write_reg("d_valid", array="0:MAX_DEPTH-1")
        writer.write_reg("d_is_leaf", array="0:MAX_DEPTH-1")
        writer.write_blank()
        
        n = self.config.n_parameters
        
        # Multiplier outputs
        writer.write_comment("Multiplier outputs")
        writer.write_line(f"wire [DATA_WIDTH-1:0] d_mult_out [0:MAX_DEPTH-1][0:N_PARAMS-1];")
        writer.write_line(f"wire d_mult_valid [0:MAX_DEPTH-1][0:N_PARAMS-1];")
        writer.write_blank()
        
        # Adder tree levels
        writer.write_comment("Adder tree outputs")
        for level_idx, level in enumerate(self.tree_structure):
            num_outputs = len(level)
            writer.write_line(f"wire [DATA_WIDTH-1:0] d_add_l{level_idx+1}_out [0:MAX_DEPTH-1][0:{num_outputs-1}];")
            writer.write_line(f"wire d_add_l{level_idx+1}_valid [0:MAX_DEPTH-1][0:{num_outputs-1}];")
        writer.write_blank()
        
        # Dot product (alias to last level)
        writer.write_comment("Dot product (alias to last level)")
        writer.write_line(f"wire [DATA_WIDTH-1:0] d_dot_out [0:MAX_DEPTH-1];")
        writer.write_line(f"wire d_dot_valid [0:MAX_DEPTH-1];")
        writer.write_blank()
        
        # Comparison result
        writer.write_comment("Comparison result")
        writer.write_wire("d_cmp_out", array="0:MAX_DEPTH-1")
        writer.write_wire("d_cmp_valid", array="0:MAX_DEPTH-1")
        writer.write_blank()
        
        # Delayed metadata
        writer.write_comment("Delayed metadata")
        writer.write_wire("d_node_id_delayed", width="[7:0]", array="0:MAX_DEPTH-1")
        writer.write_wire("d_hp_idx_delayed", width="[7:0]", array="0:MAX_DEPTH-1")
        writer.write_wire("d_fifo_ptr_delayed", width="[7:0]", array="0:MAX_DEPTH-1")
        writer.write_wire("d_is_leaf_delayed", array="0:MAX_DEPTH-1")
        writer.write_wire("d_valid_delayed", array="0:MAX_DEPTH-1")
        writer.write_wire("d_threshold_delayed", width="[DATA_WIDTH-1:0]", array="0:MAX_DEPTH-1")
        writer.write_blank()
    
    def _generate_depth0_init(self, writer: RTLWriter):
        """Generate depth 0 initialization (OUTSIDE generate)"""
        writer.write_separator("Depth 0 Initialization (Root Node)")
        writer.write_blank()
        
        writer.begin_always("posedge clk or negedge rst_n")
        writer.begin_if("!rst_n")
        writer.write_line("d_node_id[0] <= 8'd0;")
        writer.write_line("d_hp_idx[0] <= 8'd0;")
        writer.write_line("d_fifo_ptr[0] <= 8'd0;")
        writer.write_line("d_valid[0] <= 1'b0;")
        writer.write_line("d_is_leaf[0] <= 1'b0;")
        writer.begin_else_if("stage0_valid")
        writer.write_line("d_node_id[0] <= 8'd0;")
        writer.write_line("d_hp_idx[0] <= hp_list[0];")
        writer.write_line("d_fifo_ptr[0] <= param_fifo_wr_ptr;")
        writer.write_line("d_valid[0] <= 1'b1;")
        writer.write_line("d_is_leaf[0] <= (jump_list[0] == 8'd0);")
        writer.begin_else()
        writer.write_line("d_valid[0] <= 1'b0;")
        writer.end_if()
        writer.end_always()
        writer.write_blank()
    
    def _generate_pipeline_instances(self, writer: RTLWriter):
        """Generate pipeline instances (FP operations)"""
        writer.write_separator("Generate Pipeline Instances (FP Operations)")
        writer.write_blank()
        
        writer.begin_generate()
        writer.write_line("genvar depth;")
        writer.write_blank()
        writer.write_line("for (depth = 0; depth < MAX_DEPTH; depth = depth + 1) begin : depth_gen")
        writer.indent()
        writer.write_blank()
        
        # Multipliers
        self._generate_multipliers(writer)
        
        # Adder tree
        self._generate_adder_tree(writer)
        
        # Dot product alias
        last_level = len(self.tree_structure)
        writer.write_comment("Dot product alias")
        writer.write_line(f"assign d_dot_out[depth] = d_add_l{last_level}_out[depth][0];")
        writer.write_line(f"assign d_dot_valid[depth] = d_add_l{last_level}_valid[depth][0];")
        writer.write_blank()
        
        # Delay lines
        self._generate_delay_lines(writer)
        
        # Comparison
        self._generate_comparison(writer)
        
        writer.dedent()
        writer.write_line("end")
        writer.end_generate()
        writer.write_blank()
    
    def _generate_multipliers(self, writer: RTLWriter):
        """Generate multiplier instances"""
        writer.write_separator("Multipliers (parallel)")
        writer.write_blank()
        
        writer.begin_generate()
        writer.write_line("genvar param_idx;")
        writer.write_blank()
        writer.write_line("for (param_idx = 0; param_idx < N_PARAMS; param_idx = param_idx + 1) begin : mult_gen")
        writer.indent()
        writer.write_blank()
        
        # Get coefficient and parameter
        writer.write_comment("Get halfplane coefficient and parameter")
        writer.write_wire("hp_coeff", width="[DATA_WIDTH-1:0]")
        writer.write_wire("param_val", width="[DATA_WIDTH-1:0]")
        writer.write_blank()
        
        writer.write_line("assign hp_coeff = halfplanes[d_hp_idx[depth] * (N_PARAMS+1) + param_idx];")
        writer.write_line("assign param_val = param_fifo[param_idx][d_fifo_ptr[depth]];")
        writer.write_blank()
        
        # Valid signal
        writer.write_wire("fp_compute_valid")
        writer.write_line("assign fp_compute_valid = d_valid[depth] && !d_is_leaf[depth];")
        writer.write_blank()
        
        # Multiplier instance
        writer.write_line("fp_mult #(")
        writer.indent()
        writer.write_line(".DATA_WIDTH(DATA_WIDTH),")
        writer.write_line(".METADATA_WIDTH(METADATA_WIDTH)")
        writer.dedent()
        writer.write_line(") mult (")
        writer.indent()
        writer.write_line(".aclk(clk),")
        writer.write_line(".aresetn(rst_n),")
        writer.write_line(".s_axis_a_tdata(param_val),")
        writer.write_line(".s_axis_a_tvalid(fp_compute_valid),")
        writer.write_line(".s_axis_b_tdata(hp_coeff),")
        writer.write_line(".s_axis_b_tvalid(fp_compute_valid),")
        writer.write_line(".m_axis_result_tdata(d_mult_out[depth][param_idx]),")
        writer.write_line(".m_axis_result_tvalid(d_mult_valid[depth][param_idx]),")
        writer.write_line(".m_axis_result_tready(1'b1)")
        writer.dedent()
        writer.write_line(");")
        
        writer.dedent()
        writer.write_line("end")
        writer.end_generate()
        writer.write_blank()
    
    def _generate_adder_tree(self, writer: RTLWriter):
        """Generate adder tree with delay-aligned passthroughs"""
        writer.write_separator("Adder Tree with Delay-Aligned Passthroughs")
        writer.write_blank()
        
        for level_idx, level in enumerate(self.tree_structure):
            num_adders = sum(1 for a in level if not a['is_passthrough'])
            num_passthrough = sum(1 for a in level if a['is_passthrough'])
            
            writer.write_comment(f"Level {level_idx + 1}: {num_adders} adder(s), {num_passthrough} passthrough(s)")
            
            for adder in level:
                idx = adder['index']
                output_signal = f"d_add_l{level_idx+1}_out[depth][{idx}]"
                output_valid = f"d_add_l{level_idx+1}_valid[depth][{idx}]"
                
                if adder['is_passthrough']:
                    # Passthrough with delay alignment
                    writer.write_comment(f"Passthrough {idx} with delay alignment")
                    
                    # Data delay
                    writer.write_line("simple_delay_line #(")
                    writer.indent()
                    writer.write_line(".DATA_WIDTH(DATA_WIDTH),")
                    writer.write_line(".DEPTH(ACTUAL_ADD_LATENCY)")
                    writer.dedent()
                    writer.write_line(f") delay_l{level_idx+1}_passthrough_{idx}_data (")
                    writer.indent()
                    writer.write_line(".clk(clk),")
                    writer.write_line(".rst_n(rst_n),")
                    writer.write_line(f".data_in({adder['left_signal']}),")
                    writer.write_line(f".data_out({output_signal})")
                    writer.dedent()
                    writer.write_line(");")
                    writer.write_blank()
                    
                    # Valid delay
                    writer.write_line("valid_delay_line #(")
                    writer.indent()
                    writer.write_line(".DEPTH(ACTUAL_ADD_LATENCY)")
                    writer.dedent()
                    writer.write_line(f") delay_l{level_idx+1}_passthrough_{idx}_valid (")
                    writer.indent()
                    writer.write_line(".clk(clk),")
                    writer.write_line(".rst_n(rst_n),")
                    writer.write_line(f".valid_in({adder['left_valid']}),")
                    writer.write_line(f".valid_out({output_valid})")
                    writer.dedent()
                    writer.write_line(");")
                    writer.write_blank()
                    
                else:
                    # Real adder
                    writer.write_comment(f"Adder {idx}")
                    
                    # Pair valid
                    writer.write_wire(f"l{level_idx+1}_pair{idx}_valid")
                    writer.write_line(f"assign l{level_idx+1}_pair{idx}_valid = "
                                    f"{adder['left_valid']} && {adder['right_valid']};")
                    writer.write_blank()
                    
                    writer.write_line("fp_add #(")
                    writer.indent()
                    writer.write_line(".DATA_WIDTH(DATA_WIDTH),")
                    writer.write_line(".METADATA_WIDTH(METADATA_WIDTH)")
                    writer.dedent()
                    writer.write_line(f") add_l{level_idx+1}_{idx} (")
                    writer.indent()
                    writer.write_line(".aclk(clk),")
                    writer.write_line(".aresetn(rst_n),")
                    writer.write_line(f".s_axis_a_tdata({adder['left_signal']}),")
                    writer.write_line(f".s_axis_a_tvalid(l{level_idx+1}_pair{idx}_valid),")
                    writer.write_line(f".s_axis_b_tdata({adder['right_signal']}),")
                    writer.write_line(f".s_axis_b_tvalid(l{level_idx+1}_pair{idx}_valid),")
                    writer.write_line(f".m_axis_result_tdata({output_signal}),")
                    writer.write_line(f".m_axis_result_tvalid({output_valid}),")
                    writer.write_line(".m_axis_result_tready(1'b1)")
                    writer.dedent()
                    writer.write_line(");")
                    writer.write_blank()
        
        writer.write_blank()
    
    def _generate_delay_lines(self, writer: RTLWriter):
        """Generate delay lines for metadata"""
        writer.write_separator("Delay Lines for Metadata")
        writer.write_blank()
        
        delay_params = [
            ("d_node_id", 8, "node_id"),
            ("d_hp_idx", 8, "hp_idx"),
            ("d_fifo_ptr", 8, "fifo_ptr"),
        ]
        
        for sig_name, width, inst_name in delay_params:
            writer.write_line("simple_delay_line #(")
            writer.indent()
            writer.write_line(f".DATA_WIDTH({width}),")
            writer.write_line(f".DEPTH(COMPARISON_LATENCY)")
            writer.dedent()
            writer.write_line(f") delay_{inst_name} (")
            writer.indent()
            writer.write_line(".clk(clk),")
            writer.write_line(".rst_n(rst_n),")
            writer.write_line(f".data_in({sig_name}[depth]),")
            writer.write_line(f".data_out({sig_name}_delayed[depth])")
            writer.dedent()
            writer.write_line(");")
            writer.write_blank()
        
        writer.write_line("valid_delay_line #(")
        writer.indent()
        writer.write_line(f".DEPTH(COMPARISON_LATENCY)")
        writer.dedent()
        writer.write_line(") delay_is_leaf (")
        writer.indent()
        writer.write_line(".clk(clk),")
        writer.write_line(".rst_n(rst_n),")
        writer.write_line(".valid_in(d_is_leaf[depth]),")
        writer.write_line(".valid_out(d_is_leaf_delayed[depth])")
        writer.dedent()
        writer.write_line(");")
        writer.write_blank()
        
        writer.write_line("valid_delay_line #(")
        writer.indent()
        writer.write_line(f".DEPTH(COMPARISON_LATENCY)")
        writer.dedent()
        writer.write_line(") delay_valid (")
        writer.indent()
        writer.write_line(".clk(clk),")
        writer.write_line(".rst_n(rst_n),")
        writer.write_line(".valid_in(d_valid[depth]),")
        writer.write_line(".valid_out(d_valid_delayed[depth])")
        writer.dedent()
        writer.write_line(");")
        writer.write_blank()
        
        # Threshold delay (to comparison input)
        writer.write_comment("Threshold delay (to comparison input)")
        writer.write_line("simple_delay_line #(")
        writer.indent()
        writer.write_line(".DATA_WIDTH(DATA_WIDTH),")
        writer.write_line(f".DEPTH(DOT_PRODUCT_LATENCY)")
        writer.dedent()
        writer.write_line(") delay_threshold (")
        writer.indent()
        writer.write_line(".clk(clk),")
        writer.write_line(".rst_n(rst_n),")
        writer.write_line(".data_in(halfplanes[d_hp_idx[depth] * (N_PARAMS+1) + N_PARAMS]),")
        writer.write_line(".data_out(d_threshold_delayed[depth])")
        writer.dedent()
        writer.write_line(");")
        writer.write_blank()
    
    def _generate_comparison(self, writer: RTLWriter):
        """Generate comparison"""
        writer.write_separator("Comparison")
        writer.write_blank()
        
        writer.write_line("fp_compare #(")
        writer.indent()
        writer.write_line(".DATA_WIDTH(DATA_WIDTH),")
        writer.write_line(".METADATA_WIDTH(METADATA_WIDTH)")
        writer.dedent()
        writer.write_line(") compare (")
        writer.indent()
        writer.write_line(".aclk(clk),")
        writer.write_line(".aresetn(rst_n),")
        writer.write_line(".s_axis_a_tdata(d_dot_out[depth]),")
        writer.write_line(".s_axis_a_tvalid(d_dot_valid[depth]),")
        writer.write_line(".s_axis_b_tdata(d_threshold_delayed[depth]),")
        writer.write_line(".s_axis_b_tvalid(d_dot_valid[depth]),")
        writer.write_line(".m_axis_result_tdata(d_cmp_out[depth]),")
        writer.write_line(".m_axis_result_tdata_full(),")
        writer.write_line(".m_axis_result_tvalid(d_cmp_valid[depth]),")
        writer.write_line(".m_axis_result_tready(1'b1),")
        writer.write_line(".s_axis_metadata_tdata({METADATA_WIDTH{1'b0}}),")
        writer.write_line(".s_axis_metadata_tvalid(1'b0),")
        writer.write_line(".m_axis_metadata_tdata(),")
        writer.write_line(".m_axis_metadata_tvalid(),")
        writer.write_line(".m_axis_metadata_tready(1'b1)")
        writer.dedent()
        writer.write_line(");")
        writer.write_blank()
    
    def _generate_next_depth_logic(self, writer: RTLWriter):
        """Generate next depth logic (SEPARATE generate block)"""
        writer.write_separator("Next Depth Logic (Tree Traversal)")
        writer.write_blank()
        
        writer.begin_generate()
        writer.write_line("genvar next_depth;")
        writer.write_blank()
        writer.write_line("for (next_depth = 0; next_depth < MAX_DEPTH - 1; next_depth = next_depth + 1) begin : next_depth_gen")
        writer.indent()
        writer.write_blank()
        
        # Calculate next node
        writer.write_wire("next_base", width="[7:0]")
        writer.write_wire("next_node", width="[7:0]")
        writer.write_blank()
        writer.write_line("assign next_base = d_node_id_delayed[next_depth] + jump_list[d_node_id_delayed[next_depth]];")
        writer.write_line("assign next_node = next_base + {7'b0, d_cmp_out[next_depth]};")
        writer.write_blank()
        
        # Traversal logic
        writer.begin_always("posedge clk or negedge rst_n")
        writer.begin_if("!rst_n")
        writer.write_line("d_node_id[next_depth+1] <= 8'd0;")
        writer.write_line("d_hp_idx[next_depth+1] <= 8'd0;")
        writer.write_line("d_fifo_ptr[next_depth+1] <= 8'd0;")
        writer.write_line("d_valid[next_depth+1] <= 1'b0;")
        writer.write_line("d_is_leaf[next_depth+1] <= 1'b0;")
        writer.begin_else_if("d_cmp_valid[next_depth]")
        
        # Leaf check
        writer.begin_if("d_is_leaf_delayed[next_depth]")
        writer.write_comment("Reached leaf, propagate")
        writer.write_line("d_node_id[next_depth+1] <= d_node_id_delayed[next_depth];")
        writer.write_line("d_hp_idx[next_depth+1] <= d_hp_idx_delayed[next_depth];")
        writer.write_line("d_fifo_ptr[next_depth+1] <= d_fifo_ptr_delayed[next_depth];")
        writer.write_line("d_valid[next_depth+1] <= 1'b1;")
        writer.write_line("d_is_leaf[next_depth+1] <= 1'b1;")
        
        # Non-leaf: traverse
        writer.begin_else()
        writer.write_comment("Traverse to next node")
        writer.write_line("d_node_id[next_depth+1] <= next_node;")
        writer.write_line("d_hp_idx[next_depth+1] <= hp_list[next_node];")
        writer.write_line("d_fifo_ptr[next_depth+1] <= d_fifo_ptr_delayed[next_depth];")
        writer.write_line("d_valid[next_depth+1] <= 1'b1;")
        writer.write_line("d_is_leaf[next_depth+1] <= (jump_list[next_node] == 8'd0);")
        writer.end_if()
        
        writer.begin_else()
        writer.write_line("d_valid[next_depth+1] <= 1'b0;")
        writer.end_if()
        writer.end_always()
        
        writer.dedent()
        writer.write_line("end")
        writer.end_generate()
        writer.write_blank()
    
    def generate_final_leaf_stage(self, writer: RTLWriter):
        """Generate final leaf detection stage"""
        writer.write_separator("Final Leaf Stage")
        writer.write_blank()
        
        writer.write_comment("Detect final leaf from any depth")
        writer.write_reg("final_leaf_valid")
        writer.write_reg("final_hp_idx", width="[7:0]")
        writer.write_reg("final_fifo_ptr", width="[7:0]")
        writer.write_blank()
        
        # Generate priority encoder
        writer.write_comment("Loop variable for priority encoder")
        writer.write_line("integer depth_idx;")
        writer.write_blank()
        
        writer.begin_always("posedge clk or negedge rst_n")
        writer.begin_if("!rst_n")
        writer.write_line("final_leaf_valid <= 1'b0;")
        writer.write_line("final_hp_idx <= 8'd0;")
        writer.write_line("final_fifo_ptr <= 8'd0;")
        writer.begin_else()
        
        # Priority encoder using loop
        writer.write_comment("Priority: later depth wins (defensive)")
        writer.write_line("final_leaf_valid <= 1'b0;")
        writer.write_line("for (depth_idx = 0; depth_idx < MAX_DEPTH; depth_idx = depth_idx + 1) begin")
        writer.indent()
        writer.write_line("if (d_valid_delayed[depth_idx] && d_is_leaf_delayed[depth_idx]) begin")
        writer.indent()
        writer.write_line("final_leaf_valid <= 1'b1;")
        writer.write_line("final_hp_idx <= d_hp_idx_delayed[depth_idx];")
        writer.write_line("final_fifo_ptr <= d_fifo_ptr_delayed[depth_idx];")
        writer.dedent()
        writer.write_line("end")
        writer.dedent()
        writer.write_line("end")
        
        writer.end_if()
        writer.end_always()
        writer.write_blank()
    
    def generate_solution_stage(self, writer: RTLWriter):
        """Generate solution computation stage"""
        writer.write_separator("Solution Computation (MAC)")
        writer.write_blank()
        
        # S1 stage
        writer.write_comment("S1 stage: prepare for solution computation")
        writer.write_reg("s1_params", width="[DATA_WIDTH-1:0]", array="0:N_PARAMS-1")
        writer.write_reg("s1_hp_idx", width="[7:0]")
        writer.write_reg("s1_valid")
        writer.write_blank()
        
        writer.write_comment("Loop variable for S1 stage")
        writer.write_line("integer s1_idx;")
        writer.write_blank()
        
        writer.begin_always("posedge clk or negedge rst_n")
        writer.begin_if("!rst_n")
        writer.write_line("s1_valid <= 1'b0;")
        writer.write_line("s1_hp_idx <= 8'd0;")
        writer.write_line("for (s1_idx = 0; s1_idx < N_PARAMS; s1_idx = s1_idx + 1) begin")
        writer.indent()
        writer.write_line("s1_params[s1_idx] <= {DATA_WIDTH{1'b0}};")
        writer.dedent()
        writer.write_line("end")
        writer.begin_else_if("final_leaf_valid")
        writer.write_line("for (s1_idx = 0; s1_idx < N_PARAMS; s1_idx = s1_idx + 1) begin")
        writer.indent()
        writer.write_line("s1_params[s1_idx] <= param_fifo[s1_idx][final_fifo_ptr];")
        writer.dedent()
        writer.write_line("end")
        writer.write_line("s1_hp_idx <= final_hp_idx;")
        writer.write_line("s1_valid <= 1'b1;")
        writer.begin_else()
        writer.write_line("s1_valid <= 1'b0;")
        writer.end_if()
        writer.end_always()
        writer.write_blank()
        
        # Solution results
        writer.write_comment("Solution results")
        writer.write_line(f"wire [DATA_WIDTH-1:0] sol_result [0:N_SOLUTIONS-1];")
        writer.write_line(f"wire sol_valid [0:N_SOLUTIONS-1];")
        writer.write_blank()
        
        # Generate solution computations
        self._generate_solution_computations(writer)
    
    def _generate_solution_computations(self, writer: RTLWriter):
        """Generate solution computation instances"""
        writer.write_comment("Generate solution computations")
        writer.begin_generate()
        writer.write_line("genvar sol_idx;")
        writer.write_line("for (sol_idx = 0; sol_idx < N_SOLUTIONS; sol_idx = sol_idx + 1) begin : sol_gen")
        writer.indent()
        writer.write_blank()
        
        # Feedback coefficients
        writer.write_comment("Get feedback coefficients for this solution")
        writer.write_line("wire [DATA_WIDTH-1:0] fb_scales [0:N_PARAMS-1];")
        writer.write_line("wire [DATA_WIDTH-1:0] fb_offset;")
        writer.write_blank()
        
        writer.begin_generate()
        writer.write_line("genvar fb_idx;")
        writer.write_line("for (fb_idx = 0; fb_idx < N_PARAMS; fb_idx = fb_idx + 1) begin : fb_scale_gen")
        writer.indent()
        writer.write_line("assign fb_scales[fb_idx] = feedbacks[s1_hp_idx * (N_PARAMS+1) * N_SOLUTIONS + sol_idx * (N_PARAMS+1) + fb_idx];")
        writer.dedent()
        writer.write_line("end")
        writer.end_generate()
        writer.write_line("assign fb_offset = feedbacks[s1_hp_idx * (N_PARAMS+1) * N_SOLUTIONS + sol_idx * (N_PARAMS+1) + N_PARAMS];")
        writer.write_blank()
        
        # Pack for MAC
        writer.write_comment("Pack parameters and feedback scales for MAC")
        writer.write_line("wire [N_PARAMS*DATA_WIDTH-1:0] mac_a_packed;")
        writer.write_line("wire [N_PARAMS*DATA_WIDTH-1:0] mac_b_packed;")
        writer.write_blank()
        
        # Generate packing using generate
        writer.begin_generate()
        writer.write_line("genvar pack_idx;")
        writer.write_line("for (pack_idx = 0; pack_idx < N_PARAMS; pack_idx = pack_idx + 1) begin : pack_gen")
        writer.indent()
        writer.write_line("assign mac_a_packed[(N_PARAMS-1-pack_idx)*DATA_WIDTH +: DATA_WIDTH] = s1_params[pack_idx];")
        writer.write_line("assign mac_b_packed[(N_PARAMS-1-pack_idx)*DATA_WIDTH +: DATA_WIDTH] = fb_scales[pack_idx];")
        writer.dedent()
        writer.write_line("end")
        writer.end_generate()
        writer.write_blank()
        
        # MAC instance
        writer.write_comment("MAC: compute dot product of params and feedback scales")
        writer.write_line("wire [DATA_WIDTH-1:0] mac_result;")
        writer.write_line("wire mac_valid;")
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
        
        # Delay offset
        writer.write_comment("Delay offset to align with MAC output")
        writer.write_line("wire [DATA_WIDTH-1:0] offset_delayed;")
        writer.write_line("wire offset_valid_delayed;")
        writer.write_blank()
        
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
        
        # Add offset
        writer.write_comment("Add offset to MAC result")
        writer.write_line("wire add_offset_valid;")
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
        
        writer.dedent()
        writer.write_line("end")
        writer.end_generate()
        writer.write_blank()
    
    def generate_output_stage(self, writer: RTLWriter):
        """Generate output stage"""
        writer.write_separator("Output Stage")
        writer.write_blank()
        
        # Output registers
        writer.write_comment("Output registers")
        writer.write_reg("output_sol", width="[DATA_WIDTH-1:0]", array="0:N_SOLUTIONS-1")
        writer.write_reg("output_valid_reg")
        writer.write_blank()
        
        # Debug counter
        writer.write_comment("Debug counter (optional)")
        writer.write_reg("output_count", width="[31:0]")
        writer.write_blank()
        
        writer.write_comment("Loop variable for output stage")
        writer.write_line("integer out_idx;")
        writer.write_blank()
        
        writer.begin_always("posedge clk or negedge rst_n")
        writer.begin_if("!rst_n")
        writer.write_line("output_valid_reg <= 1'b0;")
        writer.write_line("output_count <= 32'd0;")
        writer.write_line("for (out_idx = 0; out_idx < N_SOLUTIONS; out_idx = out_idx + 1) begin")
        writer.indent()
        writer.write_line("output_sol[out_idx] <= {DATA_WIDTH{1'b0}};")
        writer.dedent()
        writer.write_line("end")
        writer.begin_else()
        writer.write_line("for (out_idx = 0; out_idx < N_SOLUTIONS; out_idx = out_idx + 1) begin")
        writer.indent()
        writer.write_line("output_sol[out_idx] <= sol_result[out_idx];")
        writer.dedent()
        writer.write_line("end")
        writer.write_line("output_valid_reg <= sol_valid[0];")
        writer.write_blank()
        writer.write_line("if (sol_valid[0]) begin")
        writer.indent()
        writer.write_line("output_count <= output_count + 1;")
        writer.write_comment("$display debug (commented out)")
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
        
        # Output assignments
        writer.write_separator("Output Assignments")
        writer.write_blank()
        
        # Use generate for output assignments
        writer.begin_generate()
        writer.write_line("genvar out_sol_idx;")
        writer.write_line("for (out_sol_idx = 0; out_sol_idx < N_SOLUTIONS; out_sol_idx = out_sol_idx + 1) begin : out_assign_gen")
        writer.indent()
        writer.write_line("case (out_sol_idx)")
        for i in range(self.config.n_solutions):
            writer.write_line(f"{i}: assign sol_out_{i} = output_sol[out_sol_idx];")
        writer.write_line("default: ;")
        writer.write_line("endcase")
        writer.dedent()
        writer.write_line("end")
        writer.end_generate()
        
        writer.write_line("assign valid_out = output_valid_reg;")
        writer.write_line("assign ready_out = 1'b1;")
        writer.write_blank()