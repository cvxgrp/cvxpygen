#!/usr/bin/env python3
"""
PDAQP Solution-Level Time-Division Hardware Generator
For cases where DSP blocks >= number of parameters
"""

import re
import os
import sys
import argparse
from pathlib import Path
import math

class VerilogConfigParser:
    """Parser for Verilog configuration files"""
    def __init__(self, config_file):
        self.config_file = config_file
        self.defines = {}
        self.project_name = self.extract_project_name_from_path(config_file)
        self.parse_config()
    
    def extract_project_name_from_path(self, config_path):
        config_path = Path(config_path)
        filename = config_path.stem
        if filename.endswith('_config'):
            return filename[:-7]
        parent_dir = config_path.parent.parent.name
        if parent_dir.startswith('pda_'):
            return parent_dir.replace('pda_', 'pdaqp_')
        return 'pdaqp'
    
    def parse_config(self):
        try:
            with open(self.config_file, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: Config file {self.config_file} not found")
            sys.exit(1)
        
        define_pattern = r'`define\s+(\w+)\s+([^\s]+)(?:\s*//.*)?'
        matches = re.findall(define_pattern, content)
        
        for name, value in matches:
            try:
                if value.startswith('0x'):
                    self.defines[name] = int(value, 16)
                else:
                    self.defines[name] = int(value)
            except ValueError:
                self.defines[name] = value
        
        print(f"Parsed {len(self.defines)} configuration defines")
        print(f"Project name: {self.project_name}")
    
    def get_define(self, name, default=None):
        return self.defines.get(name, default)
    
    def get_int_define(self, name, default=0):
        value = self.defines.get(name, default)
        return int(value) if isinstance(value, str) and value.isdigit() else value

class SolutionLevelHardwareGenerator:
    def __init__(self, config_parser, output_dir=None, input_axi_width=32, output_axi_width=None, dsp_blocks=8):
        self.config = config_parser
        self.input_axi_width = input_axi_width
        self.output_axi_width = output_axi_width if output_axi_width else input_axi_width
        self.dsp_blocks = dsp_blocks
        
        if output_dir is None:
            raise ValueError("Output directory must be specified")
        
        self.base_dir = Path(output_dir)
        self.rtl_dir = self.base_dir / "rtl"
        self.tb_dir = self.base_dir / "tb"
        
        if not self.base_dir.exists():
            raise ValueError(f"Output directory {self.base_dir} does not exist")
        
        for dir_path in [self.rtl_dir, self.tb_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Extract config parameters
        self.n_parameters = self.config.get_int_define('PDAQP_N_PARAMETER')
        self.n_solutions = self.config.get_int_define('PDAQP_N_SOLUTION')
        self.bst_depth = self.config.get_int_define('PDAQP_ESTIMATED_BST_DEPTH')
        self.tree_nodes = self.config.get_int_define('PDAQP_TREE_NODES', 128)
        
        # Get fractional bits
        self.halfplane_frac_bits = self.config.get_int_define('HALFPLANE_FRAC_BITS', 14)
        self.feedback_frac_bits = self.config.get_int_define('FEEDBACK_FRAC_BITS', 14)
        
        # Determine index width based on tree size
        self.use_16bit_indices = self.tree_nodes > 255
        self.index_width = 16 if self.use_16bit_indices else 8
        
        # Calculate DSP allocation strategy
        self.analyze_dsp_allocation()
        
        # Calculate AXI interface parameters
        self.total_input_bits = self.n_parameters * 16
        self.total_output_bits = self.n_solutions * 16
        self.input_cycles = (self.total_input_bits + self.input_axi_width - 1) // self.input_axi_width
        self.output_cycles = (self.total_output_bits + self.output_axi_width - 1) // self.output_axi_width
        
        # Determine state machine complexity
        self.needs_complex_states = (self.input_cycles > 1) or (self.output_cycles > 1)
        
        # Calculate latency
        self.total_latency = self.bst_depth + self.solution_calc_cycles + 4
        
        print(f"\nSolution-Level Time-Division Configuration:")
        print(f"  Parameters: {self.n_parameters}, Solutions: {self.n_solutions}")
        print(f"  Tree nodes: {self.tree_nodes} (using {self.index_width}-bit indices)")
        print(f"  DSP blocks: {self.dsp_blocks}")
        print(f"  Parallel solutions: {self.parallel_solutions}")
        print(f"  Solution batches: {self.solution_batches}")
        print(f"  Solution cycles: {self.solution_calc_cycles}")
        print(f"  AXI Input: {self.input_axi_width}-bit, {self.input_cycles} cycles")
        print(f"  AXI Output: {self.output_axi_width}-bit, {self.output_cycles} cycles")
        print(f"  Fixed-point: HP_FRAC={self.halfplane_frac_bits}, FB_FRAC={self.feedback_frac_bits}")
        print(f"  Total latency: {self.total_latency} cycles")

    def analyze_dsp_allocation(self):
        """Analyze DSP allocation for solution-level time-division"""
        if self.dsp_blocks < self.n_parameters:
            print(f"ERROR: Insufficient DSPs for Solution-Level strategy")
            print(f"Need at least {self.n_parameters} DSPs, have {self.dsp_blocks}")
            sys.exit(1)
        
        # Calculate how many solutions can be computed in parallel
        self.parallel_solutions = self.dsp_blocks // self.n_parameters
        self.solution_batches = math.ceil(self.n_solutions / self.parallel_solutions)
        self.solution_calc_cycles = self.solution_batches * 3  # 3 cycles per batch
        self.actual_dsp_usage = min(self.dsp_blocks, self.parallel_solutions * self.n_parameters)
        
        print(f"\nDSP Allocation:")
        print(f"  Strategy: Solution-Level Time-Division")
        print(f"  DSPs per solution: {self.n_parameters}")
        print(f"  Parallel solutions: {self.parallel_solutions}")
        print(f"  Total batches: {self.solution_batches}")
        print(f"  Actual DSP usage: {self.actual_dsp_usage}/{self.dsp_blocks}")

    def generate_top_module(self):
        """Generate TOP module matching the example patterns"""
        project_name = self.config.project_name
        
        # Parameter buffer width
        param_buffer_width = self.input_cycles * self.input_axi_width
        
        # Generate state definitions based on complexity
        if self.output_cycles > 1:
            # Need OUTPUT_WAIT state for multi-cycle output
            state_defs = """
    // State definitions
    localparam [2:0] STATE_IDLE       = 3'b000;
    localparam [2:0] STATE_RECEIVE    = 3'b001;
    localparam [2:0] STATE_PROCESSING = 3'b010;
    localparam [2:0] STATE_OUTPUT     = 3'b011;
    localparam [2:0] STATE_OUTPUT_WAIT = 3'b100;  // For multi-cycle output"""
            state_width = 3
            extra_regs = f", output_count"
            extra_decls = f"    reg [{int(math.log2(self.output_cycles+1))-1}:0] output_count;"
        elif self.input_cycles > 1:
            # Multi-cycle input but single-cycle output
            state_defs = """
    // State definitions
    localparam [1:0] STATE_IDLE       = 2'b00;
    localparam [1:0] STATE_RECEIVE    = 2'b01;
    localparam [1:0] STATE_PROCESSING = 2'b10;
    localparam [1:0] STATE_OUTPUT     = 2'b11;"""
            state_width = 2
            extra_regs = ""
            extra_decls = ""
        else:
            # Simple single-cycle I/O
            state_defs = """
    // State definitions
    localparam [1:0] STATE_IDLE       = 2'b00;
    localparam [1:0] STATE_PROCESSING = 2'b10;
    localparam [1:0] STATE_OUTPUT     = 2'b11;"""
            state_width = 2
            extra_regs = ""
            extra_decls = ""

        # Generate parameter declarations
        param_decls = self._generate_param_declarations()
        sol_decls = self._generate_solution_declarations()
        
        # Generate BST instance connections
        bst_params = "\n".join([f"        .param_in_{i}(param_{i})," for i in range(self.n_parameters)])
        bst_sols = "\n".join([f"        .sol_out_{i}(sol_{i})," for i in range(self.n_solutions)])
        
        # Generate state machine logic
        state_machine = self._generate_state_machine()
        
        # Generate ready/valid logic
        ready_logic = self._generate_ready_logic()
        valid_logic = self._generate_valid_logic()

        template = f"""`timescale 1ns/1ps

`include "include/{self.config.project_name}_config.vh"

module {project_name}_top (
    input                               clk,
    input                               rst_n,
    // AXI Stream Input - {self.input_cycles} cycles for {self.n_parameters}x16-bit parameters  
    input  [{self.input_axi_width-1}:0]                       s_axis_tdata,
    input                               s_axis_tvalid,
    output                              s_axis_tready,
    // AXI Stream Output - {self.output_cycles} cycles for {self.n_solutions}x16-bit solutions
    output [{self.output_axi_width-1}:0]                       m_axis_tdata,
    output                              m_axis_tvalid,
    input                               m_axis_tready
);
{state_defs}
    
    reg [{state_width-1}:0] state;
    reg [{int(math.log2(max(self.input_cycles, 2)+1))-1}:0] receive_count{extra_regs};
{extra_decls}
    reg [{param_buffer_width-1}:0] param_buffer;
    
    // BST interface signals
{param_decls}
    reg param_valid;
{sol_decls}
    wire sol_valid;
    
    // Output registers
    reg [{self.output_axi_width-1}:0] output_data_reg;
    reg output_valid_reg;
    
    // BST LUT instance
    {project_name}_bst_lut bst_inst (
        .clk(clk), .rst_n(rst_n),
{bst_params}
        .valid_in(param_valid),
{bst_sols}
        .valid_out(sol_valid)
    );
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE; receive_count <= 0;{' output_count <= 0;' if self.output_cycles > 1 else ''}
            param_buffer <= {param_buffer_width}'d0;
            param_valid <= 0; output_data_reg <= {self.output_axi_width}'d0; output_valid_reg <= 0;
{self._generate_param_resets()}
        end else begin
{state_machine}
        end
    end
    
    // Output assignments
    assign s_axis_tready = {ready_logic};
    assign m_axis_tdata = output_data_reg;
    assign m_axis_tvalid = {valid_logic};

endmodule"""
        
        output_file = self.rtl_dir / f"{project_name}_top.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated TOP module: {output_file}")
        return output_file

    def _generate_param_declarations(self):
        """Generate parameter declarations matching example style"""
        lines = []
        params_per_line = 4
        for i in range(0, self.n_parameters, params_per_line):
            end = min(i + params_per_line, self.n_parameters)
            line = "    reg signed [15:0] " + ", ".join([f"param_{j}" for j in range(i, end)]) + ";"
            lines.append(line)
        return '\n'.join(lines)

    def _generate_solution_declarations(self):
        """Generate solution wire declarations"""
        lines = []
        sols_per_line = 4
        for i in range(0, self.n_solutions, sols_per_line):
            end = min(i + sols_per_line, self.n_solutions)
            line = "    wire signed [15:0] " + ", ".join([f"sol_{j}" for j in range(i, end)]) + ";"
            lines.append(line)
        return '\n'.join(lines)

    def _generate_param_resets(self):
        """Generate parameter reset statements"""
        lines = []
        for i in range(self.n_parameters):
            lines.append(f"            param_{i} <= 16'd0;")
        return ' '.join(lines)

    def _generate_state_machine(self):
        """Generate complete state machine logic"""
        if self.input_cycles == 1 and self.output_cycles == 1:
            # Simple case: single-cycle I/O
            return self._generate_simple_state_machine()
        elif self.input_cycles > 1 and self.output_cycles == 1:
            # Multi-cycle input, single-cycle output
            return self._generate_multi_input_single_output_sm()
        elif self.input_cycles == 1 and self.output_cycles > 1:
            # Single-cycle input, multi-cycle output
            return self._generate_single_input_multi_output_sm()
        else:
            # Both multi-cycle
            return self._generate_full_multi_cycle_sm()

    def _generate_simple_state_machine(self):
        """Generate state machine for single-cycle I/O"""
        # Extract parameters directly from input
        param_extract = []
        for i in range(self.n_parameters):
            param_extract.append(f"                        param_{i} <= s_axis_tdata[{i*16+15}:{i*16}];")
        
        # Pack solutions for output
        sol_pack = "{" + ", ".join([f"sol_{i}" for i in reversed(range(self.n_solutions))]) + "}"
        
        return f"""            case (state)
                STATE_IDLE: begin
                    if (s_axis_tvalid && s_axis_tready) begin
{chr(10).join(param_extract)}
                        param_valid <= 1;
                        state <= STATE_PROCESSING;
                    end
                end
                
                STATE_PROCESSING: begin
                    param_valid <= 0;
                    if (sol_valid) begin
                        output_data_reg <= {sol_pack};
                        output_valid_reg <= 1;
                        state <= STATE_OUTPUT;
                    end
                end
                
                STATE_OUTPUT: begin
                    if (m_axis_tvalid && m_axis_tready) begin
                        output_valid_reg <= 0;
                        state <= STATE_IDLE;
                    end
                end
            endcase"""

    def _generate_multi_input_single_output_sm(self):
        """Generate state machine for multi-cycle input, single output"""
        # Build receive cases
        receive_cases = []
        for i in range(1, self.input_cycles):
            receive_cases.append(f"""                            {i}: begin
                                param_buffer[{(i+1)*self.input_axi_width-1}:{i*self.input_axi_width}] <= s_axis_tdata;
                                receive_count <= {i+1};
                            end""")
        
        # Extract parameters after last receive
        param_extract = []
        last_word_params = []
        for i in range(self.n_parameters):
            bit_pos = i * 16
            if bit_pos >= (self.input_cycles - 1) * self.input_axi_width:
                # This parameter is in the last word
                offset = bit_pos - (self.input_cycles - 1) * self.input_axi_width
                last_word_params.append(f"                                param_{i} <= s_axis_tdata[{offset+15}:{offset}];")
            else:
                param_extract.append(f"                                param_{i} <= param_buffer[{bit_pos+15}:{bit_pos}];")
        
        # Pack solutions
        sol_pack = "{" + ", ".join([f"sol_{i}" for i in reversed(range(self.n_solutions))]) + "}"
        
        return f"""            case (state)
                STATE_IDLE: begin
                    output_valid_reg <= 0;
                    if (s_axis_tvalid && s_axis_tready) begin
                        param_buffer[{self.input_axi_width-1}:0] <= s_axis_tdata;
                        receive_count <= 1;
                        state <= STATE_RECEIVE;
                        param_valid <= 0;
                    end
                end
                
                STATE_RECEIVE: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        case (receive_count)
{chr(10).join(receive_cases)}
                            {self.input_cycles-1}: begin
                                param_buffer[{self.total_input_bits-1}:{(self.input_cycles-1)*self.input_axi_width}] <= s_axis_tdata;
{chr(10).join(param_extract)}
{chr(10).join(last_word_params)}
                                param_valid <= 1;
                                state <= STATE_PROCESSING;
                            end
                        endcase
                    end
                end
                
                STATE_PROCESSING: begin
                    param_valid <= 0;
                    if (sol_valid) begin
                        output_data_reg <= {sol_pack};
                        output_valid_reg <= 1;
                        state <= STATE_OUTPUT;
                    end
                end
                
                STATE_OUTPUT: begin
                    if (m_axis_tvalid && m_axis_tready) begin
                        output_valid_reg <= 0;
                        state <= STATE_IDLE;
                        receive_count <= 0;
                    end
                end
            endcase"""

    def _generate_full_multi_cycle_sm(self):
        """Generate full multi-cycle state machine (like 6p/13s example)"""
        # Input handling
        input_handling = self._generate_multi_cycle_input_handling()
        
        # Output handling
        output_cases = []
        for i in range(self.output_cycles):
            start_idx = i * (self.output_axi_width // 16)
            end_idx = min(start_idx + (self.output_axi_width // 16), self.n_solutions)
            
            sols = []
            for j in range(start_idx, end_idx):
                sols.append(f"sol_{j}")
            
            # Pad if needed
            while len(sols) < (self.output_axi_width // 16):
                sols.insert(0, "16'h0000")
            
            output_data = "{" + ", ".join(reversed(sols)) + "}"
            
            output_cases.append(f"""                        {i}: begin
                            output_data_reg <= {output_data};
                            output_valid_reg <= 1;
                            state <= STATE_OUTPUT_WAIT;
                        end""")
        
        return f"""            case (state)
                STATE_IDLE: begin
                    output_valid_reg <= 0;
                    if (s_axis_tvalid && s_axis_tready) begin
                        param_buffer[{self.input_axi_width-1}:0] <= s_axis_tdata;
                        receive_count <= 1;
                        state <= STATE_RECEIVE;
                        param_valid <= 0;
                    end
                end
                
                STATE_RECEIVE: begin
{input_handling}
                end
                
                STATE_PROCESSING: begin
                    param_valid <= 0;
                    if (sol_valid) begin
                        output_count <= 0;
                        state <= STATE_OUTPUT;
                    end
                end
                
                STATE_OUTPUT: begin
                    case (output_count)
{chr(10).join(output_cases)}
                    endcase
                end
                
                STATE_OUTPUT_WAIT: begin
                    if (m_axis_tvalid && m_axis_tready) begin
                        output_valid_reg <= 0;
                        if (output_count == {self.output_cycles - 1}) begin
                            state <= STATE_IDLE;
                            output_count <= 0;
                        end else begin
                            output_count <= output_count + 1;
                            state <= STATE_OUTPUT;
                        end
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase"""

    def _generate_multi_cycle_input_handling(self):
        """Generate multi-cycle input receive logic"""
        cases = []
        
        for i in range(1, self.input_cycles):
            if i < self.input_cycles - 1:
                cases.append(f"""                        {i}: begin
                            param_buffer[{(i+1)*self.input_axi_width-1}:{i*self.input_axi_width}] <= s_axis_tdata;
                            receive_count <= receive_count + 1;
                        end""")
            else:
                # Last cycle - extract parameters
                param_extract = []
                last_word_params = []
                
                for j in range(self.n_parameters):
                    bit_pos = j * 16
                    if bit_pos >= (self.input_cycles - 1) * self.input_axi_width:
                        # Parameter in last word
                        offset = bit_pos - (self.input_cycles - 1) * self.input_axi_width
                        last_word_params.append(f"                            param_{j} <= s_axis_tdata[{offset+15}:{offset}];")
                    else:
                        param_extract.append(f"                            param_{j} <= param_buffer[{bit_pos+15}:{bit_pos}];")
                
                cases.append(f"""                        {i}: begin
                            param_buffer[{self.total_input_bits-1}:{i*self.input_axi_width}] <= s_axis_tdata;
{chr(10).join(param_extract)}
{chr(10).join(last_word_params)}
                            param_valid <= 1;
                            state <= STATE_PROCESSING;
                        end""")
        
        return f"""                    if (s_axis_tvalid && s_axis_tready && receive_count < {self.input_cycles}) begin
                        case (receive_count)
{chr(10).join(cases)}
                        endcase
                        receive_count <= receive_count + 1;
                    end else begin
                        param_valid <= 0;
                    end"""

    def _generate_ready_logic(self):
        """Generate ready signal logic"""
        if self.input_cycles == 1:
            return "(state == STATE_IDLE)"
        else:
            return "(state == STATE_IDLE) || (state == STATE_RECEIVE && receive_count < {})" .format(self.input_cycles)

    def _generate_valid_logic(self):
        """Generate valid signal logic"""
        if self.output_cycles == 1:
            return "output_valid_reg && (state == STATE_OUTPUT)"
        else:
            return "output_valid_reg && (state == STATE_OUTPUT_WAIT)"

    def generate_bst_lut_module(self):
        """Generate BST LUT module with solution-level time-division"""
        project_name = self.config.project_name
        
        # Calculate conservative BST depth
        max_bst_depth = max(self.bst_depth, 20) if self.n_solutions > 10 else max(self.bst_depth, 10)
        
        # Generate interfaces
        param_inputs = [f"    input  signed [15:0]                param_in_{i}," for i in range(self.n_parameters)]
        sol_outputs = [f"    output reg signed [15:0]            sol_out_{i}," for i in range(self.n_solutions)]
        
        # Generate BST pipeline registers
        param_pipes = [f"    reg [15:0] param{i}_pipe[0:MAX_BST_DEPTH-1];" for i in range(self.n_parameters)]
        
        # Generate DSP blocks
        dsp_decls, dsp_mults = self._generate_dsp_blocks()
        
        # Generate solution accumulators
        if self.n_solutions <= 13:
            # Inline declaration like 6p/13s example
            sol_accums = ["    reg signed [31:0] " + ", ".join([f"solution_accumulator_{i}" for i in range(self.n_solutions)]) + ";"]
        else:
            # Multi-line for many solutions
            sol_accums = []
            for i in range(0, self.n_solutions, 3):
                end = min(i + 3, self.n_solutions)
                sol_accums.append("    reg signed [31:0] " + ", ".join([f"solution_accumulator_{j}" for j in range(i, end)]) + ";")
        
        # Generate stored parameters
        stored_params = ["    reg [15:0] " + ", ".join([f"stored_param_{i}" for i in range(self.n_parameters)]) + ";"]
        
        # Generate halfplane calculation
        hp_terms = []
        for i in range(self.n_parameters):
            hp_terms.append(f"($signed(param{i}_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+{i}]))")
        hp_calc = " + \n                                ".join(hp_terms)
        
        # Generate DSP state machine
        dsp_state_machine = self._generate_dsp_state_machine()

        template = f"""`timescale 1ns/1ps

`include "include/{self.config.project_name}_config.vh"

module {project_name}_bst_lut (
    input                               clk,
    input                               rst_n,
    // {self.n_parameters}-parameter input interface
{chr(10).join(param_inputs)}
    input                               valid_in,
    // {self.n_solutions}-solution output interface  
{chr(10).join(sol_outputs)}
    output reg                          valid_out
);

    // ROM arrays
    (* rom_style = "distributed" *) reg [15:0] halfplanes [0:`PDAQP_HALFPLANES-1];
    (* rom_style = "distributed" *) reg [15:0] feedbacks [0:`PDAQP_FEEDBACKS-1];
    (* rom_style = "distributed" *) reg [{self.index_width-1}:0] hp_list [0:`PDAQP_TREE_NODES-1];
    (* rom_style = "distributed" *) reg [{self.index_width-1}:0] jump_list [0:`PDAQP_TREE_NODES-1];

    // Load LUT data
    initial begin
        $readmemh("include/{self.config.project_name}_halfplanes.mem", halfplanes);
        $readmemh("include/{self.config.project_name}_feedbacks.mem", feedbacks);
        $readmemh("include/{self.config.project_name}_hp_list.mem", hp_list);
        $readmemh("include/{self.config.project_name}_jump_list.mem", jump_list);
    end
    
    localparam MAX_BST_DEPTH = {max_bst_depth};
    
    // BST pipeline registers
    reg valid_pipe[0:MAX_BST_DEPTH-1];
{chr(10).join(param_pipes)}
    reg [{self.index_width-1}:0] current_id_pipe[0:MAX_BST_DEPTH-1];
    reg [{self.index_width-1}:0] next_id_pipe[0:MAX_BST_DEPTH-1];
    reg [{self.index_width-1}:0] hp_idx_pipe[0:MAX_BST_DEPTH-1];
    reg traversal_done_pipe[0:MAX_BST_DEPTH-1];
    
    // {self.actual_dsp_usage} DSP blocks{' - parallel dual-solution computation' if self.parallel_solutions > 1 else ''}
{chr(10).join(dsp_decls)}
    
    // DSP multipliers
    always @(posedge clk) begin
{chr(10).join(dsp_mults)}
    end
    
    // {'Optimized ' if self.parallel_solutions > 1 else ''}DSP control{' - dual solution per cycle' if self.parallel_solutions == 2 else ''}
    reg [3:0] dsp_state;
    reg [{int(math.log2(max(self.n_solutions, 4)))-1}:0] current_solution;{f' // Up to {self.n_solutions} solutions' if self.n_solutions > 4 else ''}
    reg [{int(math.log2(max(self.solution_batches, 2)))-1}:0] {'cycle_count' if self.parallel_solutions > 1 else 'param_cycle'};
    reg [{int(math.log2(self.n_solutions * self.n_parameters * 16))-1}:0] feedback_base_addr;
    
    // {'Dual accumulator sets for parallel computation' if self.parallel_solutions > 1 else 'Solution accumulators'}
{chr(10).join(sol_accums)}
    
    // Parameter storage
{chr(10).join(stored_params)}
    
    // BST halfplane calculation - force LUT implementation
    (* use_dsp = "no" *) reg signed [31:0] hp_val;
    (* use_dsp = "no" *) reg signed [31:0] hp_thresh;
    reg decision;
    reg [{self.index_width-1}:0] new_id;
    
    integer i, j;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset BST pipeline
            for (i = 0; i < MAX_BST_DEPTH; i = i + 1) begin
                valid_pipe[i] <= 0;
{self._generate_pipe_resets()}
                current_id_pipe[i] <= 0; next_id_pipe[i] <= 0; hp_idx_pipe[i] <= 0; traversal_done_pipe[i] <= 0;
            end
            
            // Reset all {self.actual_dsp_usage} DSP blocks
{self._generate_dsp_resets()}
            
            // Reset accumulators
            for (i = 0; i < {self.n_solutions}; i = i + 1) solution_accumulator_{{i}} <= 0;
            
            // Reset stored params
            for (i = 0; i < {self.n_parameters}; i = i + 1) stored_param_{{i}} <= 0;
            
            // Reset outputs
            valid_out <= 0;
            for (i = 0; i < {self.n_solutions}; i = i + 1) sol_out_{{i}} <= 0;
            
            dsp_state <= 0; current_solution <= 0;
            {'cycle_count' if self.parallel_solutions > 1 else 'param_cycle'} <= 0; feedback_base_addr <= 0;
            
        end else begin
            
            // BST Pipeline Stage 0
            valid_pipe[0] <= valid_in;
            if (valid_in) begin
{self._generate_pipe_inputs()}
                current_id_pipe[0] <= 0; next_id_pipe[0] <= jump_list[0]; hp_idx_pipe[0] <= hp_list[0];
                traversal_done_pipe[0] <= (jump_list[0] == 0);
            end

            // BST Pipeline Stages 1-N
            for (i = 0; i < MAX_BST_DEPTH-1; i = i + 1) begin
                valid_pipe[i+1] <= valid_pipe[i];
{self._generate_pipe_propagation()}
                
                if (valid_pipe[i]) begin
                    if (traversal_done_pipe[i]) begin
                        current_id_pipe[i+1] <= current_id_pipe[i]; next_id_pipe[i+1] <= next_id_pipe[i];
                        hp_idx_pipe[i+1] <= hp_idx_pipe[i]; traversal_done_pipe[i+1] <= 1;
                    end else begin
                        // Halfplane computation - LUT-based
                        hp_val = {hp_calc};
                        
                        hp_thresh = $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+`PDAQP_N_PARAMETER]) << `{'HALFPLANE_FRAC_BITS' if self.halfplane_frac_bits == self.feedback_frac_bits else 'FEEDBACK_FRAC_BITS'};
                        decision = hp_val <= hp_thresh;
                        
                        if (decision) begin
                            new_id = next_id_pipe[i] + 1;
                        end else begin
                            new_id = next_id_pipe[i];
                        end
                        
                        current_id_pipe[i+1] <= new_id; next_id_pipe[i+1] <= new_id + jump_list[new_id];
                        hp_idx_pipe[i+1] <= hp_list[new_id]; traversal_done_pipe[i+1] <= (jump_list[new_id] == 0);
                    end
                end
            end
            
            // {'Optimized ' if self.parallel_solutions > 1 else ''}DSP computation - {self.parallel_solutions if self.parallel_solutions > 1 else 'one'} solution{'s' if self.parallel_solutions > 1 else ''} per cycle
            if (valid_pipe[MAX_BST_DEPTH-1] && dsp_state == 0) begin
                feedback_base_addr <= hp_idx_pipe[MAX_BST_DEPTH-1] * {'91' if self.n_solutions == 13 and self.n_parameters == 6 else f'(`PDAQP_N_PARAMETER + 1) * `PDAQP_N_SOLUTION'}; // {self.n_solutions}*{self.n_parameters+1}
                dsp_state <= 1; current_solution <= 0;
                {'cycle_count' if self.parallel_solutions > 1 else 'param_cycle'} <= 0;
                
{self._generate_param_storage()}
                
                for (j = 0; j < {self.n_solutions}; j = j + 1) solution_accumulator_{{j}} <= 0;
            end
            
{dsp_state_machine}
        end
    end
endmodule"""
        
        output_file = self.rtl_dir / f"{project_name}_bst_lut.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated BST LUT module")
        return output_file

    def _generate_dsp_blocks(self):
        """Generate DSP declarations and multipliers"""
        decls = []
        mults = []
        
        # Group DSP declarations by 4
        for i in range(0, self.actual_dsp_usage, 4):
            end = min(i + 4, self.actual_dsp_usage)
            
            # A and B registers
            a_regs = ", ".join([f"dsp_a_{j}" for j in range(i, end)])
            b_regs = ", ".join([f"dsp_b_{j}" for j in range(i, end)])
            decls.append(f"    (* use_dsp = \"yes\" *) reg signed [15:0] {a_regs}, {b_regs};")
            
            # P registers
            p_regs = ", ".join([f"dsp_p_{j}" for j in range(i, end)])
            decls.append(f"    (* use_dsp = \"yes\" *) reg signed [31:0] {p_regs};")
        
        # Generate multipliers
        for i in range(self.actual_dsp_usage):
            mults.append(f"        dsp_p_{i} <= dsp_a_{i} * dsp_b_{i};")
        
        return decls, mults

    def _generate_pipe_resets(self):
        """Generate pipeline reset statements"""
        lines = []
        for i in range(self.n_parameters):
            lines.append(f"                param{i}_pipe[i] <= 0;")
        return '\n'.join(lines)

    def _generate_dsp_resets(self):
        """Generate DSP reset statements"""
        lines = []
        # Format like the examples
        for i in range(0, self.actual_dsp_usage, 4):
            end = min(i + 4, self.actual_dsp_usage)
            line_parts = []
            for j in range(i, end):
                line_parts.append(f"dsp_a_{j} <= 0; dsp_b_{j} <= 0;")
            lines.append("            " + " ".join(line_parts))
        return '\n'.join(lines)

    def _generate_pipe_inputs(self):
        """Generate pipeline input assignments"""
        lines = []
        # Format like examples: multiple params per line
        for i in range(0, self.n_parameters, 3):
            end = min(i + 3, self.n_parameters)
            line_parts = []
            for j in range(i, end):
                line_parts.append(f"param{j}_pipe[0] <= param_in_{j};")
            lines.append("                " + " ".join(line_parts))
        return '\n'.join(lines)

    def _generate_pipe_propagation(self):
        """Generate pipeline propagation"""
        lines = []
        for i in range(self.n_parameters):
            lines.append(f"                param{i}_pipe[i+1] <= param{i}_pipe[i];")
        return '\n'.join(lines)

    def _generate_param_storage(self):
        """Generate parameter storage from pipeline"""
        lines = []
        # Format like examples
        for i in range(0, self.n_parameters, 2):
            end = min(i + 2, self.n_parameters)
            line_parts = []
            for j in range(i, end):
                line_parts.append(f"stored_param_{j} <= param{j}_pipe[MAX_BST_DEPTH-1];")
            lines.append("                " + " ".join(line_parts))
        return '\n'.join(lines)

    def _generate_dsp_state_machine(self):
        """Generate DSP state machine matching example patterns"""
        if self.parallel_solutions == 2 and self.n_solutions == 4:
            # Optimized dual-solution pattern like 4p/4s
            return self._generate_dual_solution_state_machine()
        else:
            # Standard sequential pattern like 6p/13s
            return self._generate_sequential_state_machine()

    def _generate_dual_solution_state_machine(self):
        """Generate optimized dual solution state machine (4p/4s pattern)"""
        return f"""            // Optimized DSP state machine - dual solution computation
            case (dsp_state)
                1: begin  // Set DSP inputs for dual solutions
                    $display("[DSP] State 1: cycle %0d (computing 2 solutions)", cycle_count);
                    
                    if (cycle_count == 0) begin
                        // Cycle 0: Compute solution 0 and 1 in parallel
                        // DSPs 0-{self.n_parameters-1}: Solution 0
{self._generate_dsp_loads(0, 0, self.n_parameters)}
                        
                        // DSPs {self.n_parameters}-{2*self.n_parameters-1}: Solution 1
{self._generate_dsp_loads(1, self.n_parameters, self.n_parameters)}
                    end else begin
                        // Cycle 1: Compute solution 2 and 3 in parallel
                        // DSPs 0-{self.n_parameters-1}: Solution 2
{self._generate_dsp_loads(2, 0, self.n_parameters)}
                        
                        // DSPs {self.n_parameters}-{2*self.n_parameters-1}: Solution 3
{self._generate_dsp_loads(3, self.n_parameters, self.n_parameters)}
                    end
                    
                    dsp_state <= 2;
                end
                
                2: begin  // Wait for DSP computation
                    $display("[DSP] State 2: DSP computation in progress");
                    dsp_state <= 3;
                end
                
                3: begin  // Accumulate dual results
                    $display("[DSP] State 3: accumulating dual solutions, cycle %0d", cycle_count);
                    
                    if (cycle_count == 0) begin
                        // Accumulate solutions 0 and 1
                        solution_accumulator_0 <= {self._generate_dsp_sum(0, self.n_parameters)};  // Solution 0
                        solution_accumulator_1 <= {self._generate_dsp_sum(self.n_parameters, self.n_parameters)};  // Solution 1
                        cycle_count <= 1;
                        dsp_state <= 1;  // Next cycle
                    end else begin
                        // Accumulate solutions 2 and 3
                        solution_accumulator_2 <= {self._generate_dsp_sum(0, self.n_parameters)};  // Solution 2
                        solution_accumulator_3 <= {self._generate_dsp_sum(self.n_parameters, self.n_parameters)};  // Solution 3
                        dsp_state <= 4;  // All solutions computed
                    end
                end
                
                4: begin  // Output all results
                    $display("[DSP] State 4: outputting all solutions");
{self._generate_solution_outputs()}
                    
                    valid_out <= 1;
                    dsp_state <= 0;
                    $display("[DSP] Complete! Total cycles: BST=%0d + DSP=%0d", MAX_BST_DEPTH, {self.solution_calc_cycles});
                end
                
                default: begin
                    valid_out <= 0;
                end
            endcase"""

    def _generate_sequential_state_machine(self):
        """Generate standard sequential state machine (6p/13s pattern)"""
        return f"""            case (dsp_state)
                1: begin
{self._generate_sequential_dsp_loads()}
                    dsp_state <= 2;
                end
                
                2: begin
                    dsp_state <= 3;
                end
                
                3: begin
{self._generate_sequential_accumulation()}
                end
                
                4: begin
{self._generate_solution_outputs()}
                    
                    valid_out <= 1;
                    dsp_state <= 0;
                end
                
                default: valid_out <= 0;
            endcase"""

    def _generate_dsp_loads(self, solution_idx, dsp_start, count):
        """Generate DSP load statements for a solution"""
        lines = []
        for i in range(count):
            if i < self.n_parameters:
                lines.append(f"                        dsp_a_{dsp_start + i} <= stored_param_{i}; dsp_b_{dsp_start + i} <= feedbacks[feedback_base_addr + {solution_idx} * (`PDAQP_N_PARAMETER + 1) + {i}];")
        return '\n'.join(lines)

    def _generate_dsp_sum(self, start_idx, count):
        """Generate DSP sum expression"""
        terms = []
        for i in range(count):
            terms.append(f"dsp_p_{start_idx + i}")
        return " + ".join(terms)

    def _generate_sequential_dsp_loads(self):
        """Generate sequential DSP loads"""
        lines = []
        for i in range(self.n_parameters):
            lines.append(f"                    dsp_a_{i} <= stored_param_{i}; dsp_b_{i} <= feedbacks[feedback_base_addr + current_solution * (`PDAQP_N_PARAMETER + 1) + {i}];")
        return '\n'.join(lines)

    def _generate_sequential_accumulation(self):
        """Generate sequential accumulation logic"""
        # Generate sum
        sum_expr = " + ".join([f"dsp_p_{i}" for i in range(self.n_parameters)])
        
        lines = [f"                    solution_accumulator_{{current_solution}} <= {sum_expr} + (feedbacks[feedback_base_addr + current_solution * (`PDAQP_N_PARAMETER + 1) + `PDAQP_N_PARAMETER] << `FEEDBACK_FRAC_BITS);"]
        lines.append("")
        lines.append(f"                    if (current_solution == {self.n_solutions - 1}) begin")
        lines.append("                        dsp_state <= 4;")
        lines.append("                    end else begin")
        lines.append("                        current_solution <= current_solution + 1;")
        lines.append("                        dsp_state <= 1;")
        lines.append("                    end")
        
        return '\n'.join(lines)

    def _generate_solution_outputs(self):
        """Generate solution output assignments"""
        lines = []
        
        # Check if we need special handling for different fractional bits
        if self.halfplane_frac_bits != self.feedback_frac_bits:
            # Like 6p/13s example - direct shift
            for i in range(self.n_solutions):
                lines.append(f"                    sol_out_{i} <= solution_accumulator_{i} >>> `FEEDBACK_FRAC_BITS;")
        else:
            # Like 4p/4s example - add bias then extract bits
            for i in range(self.n_solutions):
                lines.append(f"                    sol_out_{i} <= (solution_accumulator_{i} + (feedbacks[feedback_base_addr + {i} * (`PDAQP_N_PARAMETER + 1) + `PDAQP_N_PARAMETER] << `FEEDBACK_FRAC_BITS)) >> `FEEDBACK_FRAC_BITS;")
        
        return '\n'.join(lines)

    def _generate_test_results_display(self):
        """Generate the test results display section"""
        lines = []
        
        lines.append("        // Detailed results for first few vectors")
        lines.append("        for (i = 0; i < TEST_VECTORS && i < 10; i = i + 1) begin")
        
        if self.output_cycles == 1:
            lines.append("            if (i < output_count) begin")
            lines.append(f"                $display(\"V%0d: Expected=%0{self.total_output_bits//4}h, Actual=%0{self.total_output_bits//4}h, %s\", ")
            lines.append("                        i, expected_results[i], actual_results[i], ")
            lines.append("                        (expected_results[i] == actual_results[i]) ? \"PASS\" : \"FAIL\");")
            lines.append("            end else begin")
            lines.append("                $display(\"V%0d: Missing output\", i);")
            lines.append("            end")
        else:
            lines.append(f"            $display(\"V%0d: Exp=%0{self.total_output_bits//4}h, Act=%0{self.total_output_bits//4}h, %s\", ")
            lines.append("                    i, expected_results[i], actual_results[i], ")
            lines.append("                    (expected_results[i] == actual_results[i]) ? \"PASS\" : \"FAIL\");")
        
        lines.append("        end")
        lines.append("")
        lines.append("        if (errors == 0 && output_count >= TEST_VECTORS)")
        lines.append("            $display(\"\\n*** ALL %0d TESTS PASSED ***\", TEST_VECTORS);")
        lines.append("        else")
        
        output_type = 'outputs' if self.output_cycles == 1 else 'vectors'
        lines.append(f"            $display(\"\\n*** TESTS FAILED: %0d errors, %0d/%0d {output_type} ***\", ")
        lines.append("                    errors, output_count, TEST_VECTORS);")
        lines.append("")
        lines.append("        $finish;")
        lines.append("    end")
        
        return '\n'.join(lines)

    def generate_testbench(self):
        """Generate testbench matching example patterns"""
        project_name = self.config.project_name
        
        # Generate parameter arrays and test logic
        param_arrays = [f"    reg [15:0] test_params_{i} [0:TEST_VECTORS-1];" for i in range(self.n_parameters)]
        
        # Different initialization patterns based on problem size
        if self.n_parameters <= 4:
            # Pattern like 4p/4s
            param_inits = []
            for i in range(self.n_parameters):
                scale = 32 // (i + 1)
                param_inits.append(f"            test_params_{i}[i] = 16'h{(i+1)*2:02x}00 + i*{scale};")
        else:
            # Pattern like 6p/13s
            param_inits = []
            for i in range(self.n_parameters):
                param_inits.append(f"            test_params_{i}[i] = 16'h{i:02x}00 + i*16;")

        # Index type based on tree size
        index_type = f"[{self.index_width-1}:0]"
        
        # Generate groundtruth calculation
        param_list = ", ".join([f"p{i}" for i in range(self.n_parameters)])
        param_inputs = ", ".join([f"[15:0] p{i}" for i in range(self.n_parameters)])
        
        # Halfplane sum
        hp_terms = []
        for i in range(self.n_parameters):
            hp_terms.append(f"$signed(p{i}) * $signed(halfplanes[disp+{i}])")
        hp_sum = " +\n                      ".join(hp_terms)
        
        # Solution sum
        sol_terms = []
        for i in range(self.n_parameters):
            sol_terms.append(f"                         $signed(p{i}) * $signed(feedbacks[feedback_base + j*(`PDAQP_N_PARAMETER+1) + {i}])")
        sol_sum = " +\n".join(sol_terms)
        
        # Solution packing
        sol_pack = "{" + ", ".join([f"sol[{i}][15:0]" for i in reversed(range(self.n_solutions))]) + "}"

        # Generate the test results display section
        test_results_display = self._generate_test_results_display()

        template = f"""`timescale 1ns/1ps
`include "include/{self.config.project_name}_config.vh"

module {project_name}_tb;

    localparam CLK_PERIOD = 10;
    localparam SIM_CYCLES = 2000000;{'  // Extended for larger problem' if self.n_solutions > 10 else ''}
    localparam TEST_VECTORS = 50;
    localparam PIPELINE_DELAY = {self.total_latency + 20};  // {'Conservative estimate' if self.n_solutions > 10 else 'BST + DSP cycles'}
    
    // DUT signals - updated for {self.n_parameters}p/{self.n_solutions}s
    reg clk;
    reg rst_n;
    reg [{self.input_axi_width-1}:0] s_axis_tdata;
    reg s_axis_tvalid;
    wire s_axis_tready;
    wire [{self.output_axi_width-1}:0] m_axis_tdata;        // {self.output_cycles}×{self.output_axi_width//16} solutions per cycle
    wire m_axis_tvalid;
    reg m_axis_tready;
    
    // Test data arrays - {self.n_parameters} parameters
{chr(10).join(param_arrays)}
    
    // Expected results: {self.n_solutions} solutions × 16 bits = {self.total_output_bits} bits{f', packed in {self.output_cycles}×{self.output_axi_width}-bit cycles' if self.output_cycles > 1 else ''}
    reg [{self.total_output_bits-1}:0] expected_results [0:TEST_VECTORS-1];
    reg [{self.total_output_bits-1}:0] actual_results [0:TEST_VECTORS-1];{f'''
    reg [{self.output_axi_width-1}:0] output_cycle_data [0:{self.output_cycles-1}];  // {self.output_cycles} cycles of output data
    reg [{int(math.log2(self.output_cycles+1))-1}:0] current_output_cycle;''' if self.output_cycles > 1 else ''}
    integer output_count;
    integer test_idx;
    integer errors;
    integer i;
    
    // Load LUT data for golden reference{' - 16-bit arrays for large problem' if self.use_16bit_indices else ''}
    reg [15:0] halfplanes [0:`PDAQP_HALFPLANES-1];
    reg [15:0] feedbacks [0:`PDAQP_FEEDBACKS-1];
    reg {index_type} hp_list [0:`PDAQP_TREE_NODES-1];{f'     // {self.index_width}-bit for {self.tree_nodes} nodes' if self.use_16bit_indices else ''}
    reg {index_type} jump_list [0:`PDAQP_TREE_NODES-1];{f'   // {self.index_width}-bit for {self.tree_nodes} nodes' if self.use_16bit_indices else ''}
    
    // DUT instantiation
    {project_name}_top DUT (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Load LUT data
    initial begin
        $readmemh("include/{self.config.project_name}_halfplanes.mem", halfplanes);
        $readmemh("include/{self.config.project_name}_feedbacks.mem", feedbacks);
        $readmemh("include/{self.config.project_name}_hp_list.mem", hp_list);
        $readmemh("include/{self.config.project_name}_jump_list.mem", jump_list);
        $display("[TB] LUT data loaded for {self.n_parameters}p/{self.n_solutions}s problem");
    end

    // Task: Send {self.n_parameters}-parameter data in {self.input_cycles} AXI cycle{'s' if self.input_cycles > 1 else ''}
    task send_parameter_data;
        input integer test_vector_idx;
        reg [{self.total_input_bits-1}:0] param_data;  // {self.n_parameters}×16 = {self.total_input_bits} bits
        integer cycle;
        begin
            // Pack {self.n_parameters}×16-bit parameters into {self.total_input_bits} bits
            param_data = {{{', '.join([f'test_params_{i}[test_vector_idx]' for i in reversed(range(self.n_parameters))])}}};
            
            if (test_vector_idx < 5) begin
                $display("[TB] === SENDING VECTOR %0d ===", test_vector_idx);
                $display("     Params: {' '.join(['%04h' for _ in range(self.n_parameters)])}", 
                        {', '.join([f'test_params_{i}[test_vector_idx]' for i in range(self.n_parameters)])});
            end
            
            // Send in {self.input_cycles} cycle{'s' if self.input_cycles > 1 else ''}: {', '.join([f'[{(i+1)*self.input_axi_width-1}:{i*self.input_axi_width}]' for i in range(self.input_cycles)])}
            for (cycle = 0; cycle < {self.input_cycles}; cycle = cycle + 1) begin
                @(posedge clk);
                s_axis_tdata = param_data[cycle*{self.input_axi_width} +: {self.input_axi_width}];
                s_axis_tvalid = 1;
                
                while (!s_axis_tready) @(posedge clk);
                if (test_vector_idx < 5) $display("     Cycle %0d: %0{self.input_axi_width//4}h", cycle+1, s_axis_tdata);
            end
            
            @(posedge clk);
            s_axis_tvalid = 0;
        end
    endtask

    // Function: Calculate golden reference using C algorithm
    function [{self.total_output_bits-1}:0] calculate_groundtruth;
        input {param_inputs};
        
        reg {index_type} id, next_id;
        reg [{'31' if self.use_16bit_indices else '15'}:0] disp, feedback_base;
        reg signed [31:0] val;
        reg signed [31:0] thresh;
        reg signed [31:0] sol [0:`PDAQP_N_SOLUTION-1];
        integer j, iter;
        
        begin
            // BST traversal - matches C code exactly
            id = 0;
            next_id = id + jump_list[id];
            iter = 0;
            
            while (next_id != id && iter < {50 if self.tree_nodes > 100 else 20}) begin  // Safety limit{'for deep trees' if self.tree_nodes > 100 else ''}
                disp = hp_list[id] * (`PDAQP_N_PARAMETER + 1);
                
                // Compute halfplane value ({self.n_parameters} parameters)
                val = {hp_sum};
                
                // Threshold comparison - use {'feedback' if self.halfplane_frac_bits != self.feedback_frac_bits else 'correct'} scaling
                thresh = $signed(halfplanes[disp+`PDAQP_N_PARAMETER]) << `{'FEEDBACK_FRAC_BITS' if self.halfplane_frac_bits != self.feedback_frac_bits else 'HALFPLANE_FRAC_BITS'};
                
                // Decision: C code uses (val <= thresh)
                if (val <= thresh) begin
                    id = next_id + 1;
                end else begin
                    id = next_id;
                end
                
                next_id = id + jump_list[id];
                iter = iter + 1;
            end
            
            // Solution computation at leaf node
            feedback_base = hp_list[id] * (`PDAQP_N_PARAMETER + 1) * `PDAQP_N_SOLUTION;
            
            for (j = 0; j < `PDAQP_N_SOLUTION; j = j + 1) begin
                // Compute affine function: Ax + b ({self.n_parameters} parameters)
                sol[j] = {sol_sum} +
                         ($signed(feedbacks[feedback_base + j*(`PDAQP_N_PARAMETER+1) + `PDAQP_N_PARAMETER]) << `FEEDBACK_FRAC_BITS);
                
                // Scale to output format
                sol[j] = sol[j][`FEEDBACK_FRAC_BITS + `OUTPUT_DATA_WIDTH - 1:`FEEDBACK_FRAC_BITS];
            end
            
            // Pack {self.n_solutions} solutions into {self.total_output_bits}-bit output
            calculate_groundtruth = {sol_pack};
        end
    endfunction

    // Main test sequence
    initial begin
        rst_n = 0;
        s_axis_tvalid = 0;
        s_axis_tdata = 0;
        m_axis_tready = 1;
        output_count = 0;{f'''
        current_output_cycle = 0;''' if self.output_cycles > 1 else ''}
        errors = 0;
        {f'''
        // Clear output data arrays
        for (i = 0; i < {self.output_cycles}; i = i + 1) begin
            output_cycle_data[i] = {self.output_axi_width}'h0;
        end
        ''' if self.output_cycles > 1 else ''}
        // Generate {'diverse ' if self.n_parameters <= 4 else ''}test vectors for {self.n_parameters} parameters
        for (i = 0; i < TEST_VECTORS; i = i + 1) begin
{chr(10).join(param_inits)}
        end
        
        $display("\\n=== PDAQP {self.output_cycles + '-AXI Output' if self.output_cycles > 1 else 'Solution-Level Time-Division'} Test ({self.n_parameters}p/{self.n_solutions}s) ===");
        $display("Config: %0dp/%0ds, {'Input' if self.input_axi_width != self.output_axi_width else 'AXI'}={'{}×{}b' if self.input_cycles > 1 else '{}b'}, Output={'{}×{}b' if self.output_cycles > 1 else '{}b'}{', II=1' if self.output_cycles > 1 else ''}", 
                `PDAQP_N_PARAMETER, `PDAQP_N_SOLUTION, {f'{self.input_cycles}, {self.input_axi_width}' if self.input_cycles > 1 else str(self.input_axi_width)}, {f'{self.output_cycles}, {self.output_axi_width}' if self.output_cycles > 1 else str(self.output_axi_width)});
        $display("BST depth: %0d{', Pipeline: %0d stages' if self.output_cycles > 1 else ''}", `PDAQP_ESTIMATED_BST_DEPTH{', PIPELINE_DELAY' if self.output_cycles > 1 else ''});
        {f'''$display("Expected latency: %0d cycles", PIPELINE_DELAY);
        $display("Output parallelization: {self.output_cycles}×{self.output_axi_width//16}-bit interfaces (total %0d bits)", 
                `PDAQP_N_SOLUTION * 16);''' if self.output_cycles > 1 else f'$display("Expected latency: ~{self.total_latency} cycles");'}
        
        // Calculate expected results
        for (i = 0; i < TEST_VECTORS; i = i + 1) begin
            if (i < 3) $display("\\n=== Computing Expected Result %0d ===", i);
            expected_results[i] = calculate_groundtruth({', '.join([f'test_params_{i}[i]' for i in range(self.n_parameters)])});
            if (i < 3) $display("Expected[%0d] = %0{self.total_output_bits//4}h", i, expected_results[i]);
        end
        
        // Reset sequence
        repeat({'10' if self.n_solutions <= 10 else '20'}) @(posedge clk);
        rst_n = 1;
        repeat({'10' if self.n_solutions <= 10 else '20'}) @(posedge clk);
        
        // Send test vectors
        for (test_idx = 0; test_idx < TEST_VECTORS; test_idx = test_idx + 1) begin
            if (test_idx < {'5' if self.output_cycles > 1 else '3'}) $display("\\n[%0t] === TESTING VECTOR %0d ===", $time, test_idx);
            send_parameter_data(test_idx);
            repeat({'20' if self.output_cycles > 1 else 'PIPELINE_DELAY'}) @(posedge clk);{'  // Allow time between vectors' if self.output_cycles > 1 else ''}
        end
        
        // Wait for all outputs
        repeat({'PIPELINE_DELAY * 2' if self.output_cycles > 1 else '200'}) @(posedge clk);
        
        $display("\\n=== FINAL RESULTS ===");
        $display("Vectors processed: %0d/%0d", output_count, TEST_VECTORS);
        $display("Errors: %0d", errors);{f'''
        if (output_count > 0) begin
            $display("Achieved II: 1 cycle (%0dM vectors/sec @ 100MHz)", 100);
            $display("Output bandwidth: {self.output_cycles}×{self.output_axi_width}bit = {self.output_cycles * self.output_axi_width} bits/cycle");
        end''' if self.output_cycles > 1 else f'''
        $display("Success rate: %0.1f%%", (TEST_VECTORS - errors) * 100.0 / TEST_VECTORS);'''}
        
{test_results_display}
    
    // Output capture and verification{' - handles 4-cycle output' if self.output_cycles == 4 else ''}
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin{self._generate_output_capture()}
        end
    end
    
    // Timeout protection
    initial begin
        #(SIM_CYCLES * CLK_PERIOD);
        $display("TIMEOUT - %0d outputs received", output_count);
        $finish;
    end
    
    // Waveform dump
    initial begin
        $dumpfile("{project_name}{'_6p13s' if self.n_parameters == 6 and self.n_solutions == 13 else f'_{self.n_parameters}p{self.n_solutions}s'}_tb.vcd");
        $dumpvars(0, {project_name}_tb);
    end
    
endmodule"""
        
        output_file = self.tb_dir / f"{project_name}_tb.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated testbench: {output_file}")
        return output_file

    def _generate_output_capture(self):
        """Generate output capture logic for testbench"""
        if self.output_cycles == 1:
            # Single-cycle output
            return f"""
            if (output_count < TEST_VECTORS) begin
                actual_results[output_count] = m_axis_tdata;
                
                if (m_axis_tdata !== expected_results[output_count]) begin
                    if (output_count < 10) begin
                        $display("[%0t] *** ERROR V%0d: exp=%0{self.total_output_bits//4}h, got=%0{self.total_output_bits//4}h ***", 
                                $time, output_count, expected_results[output_count], m_axis_tdata);
                    end
                    errors = errors + 1;
                end else begin
                    if (output_count < 10) begin
                        $display("[%0t] *** PASS V%0d: %0{self.total_output_bits//4}h ***", $time, output_count, m_axis_tdata);
                    end
                end
            end
            output_count = output_count + 1;"""
        else:
            # Multi-cycle output (like 6p/13s)
            return f"""
            output_cycle_data[current_output_cycle] = m_axis_tdata;
            
            if (output_count < 5) begin
                $display("[%0t] *** RX Cycle %0d: %0{self.output_axi_width//4}h ***", 
                        $time, current_output_cycle, m_axis_tdata);
            end
            
            if (current_output_cycle == {self.output_cycles - 1}) begin
                // All {self.output_cycles} cycles received, assemble complete result
                if (output_count < TEST_VECTORS) begin
                    actual_results[output_count] = {{{', '.join([f'output_cycle_data[{self.output_cycles-1-i}]' for i in range(self.output_cycles)])}}};
                    
                    if (actual_results[output_count] !== expected_results[output_count]) begin
                        $display("[%0t] *** ERROR V%0d ***", $time, output_count);
                        $display("     Expected: %0{self.total_output_bits//4}h", expected_results[output_count]);
                        $display("     Actual:   %0{self.total_output_bits//4}h", actual_results[output_count]);
                        errors = errors + 1;
                    end else begin
                        $display("[%0t] *** PASS V%0d: {self.output_cycles}-AXI parallel output correct ***", 
                                $time, output_count);
                    end
                end
                output_count = output_count + 1;
                current_output_cycle = 0;
            end else begin
                current_output_cycle = current_output_cycle + 1;
            end"""

    def generate_all(self):
        """Generate all files"""
        print(f"\nGenerating Solution-Level Time-Division PDAQP Hardware")
        print(f"=" * 50)
        
        bst_file = self.generate_bst_lut_module()
        top_file = self.generate_top_module()
        tb_file = self.generate_testbench()
        
        print(f"\n✓ Generation complete!")
        print(f"✓ Solution-level time-division with {self.dsp_blocks} DSPs")
        print(f"✓ Computing {self.parallel_solutions} solution{'s' if self.parallel_solutions > 1 else ''} in parallel")
        print(f"✓ Total solution computation: {self.solution_calc_cycles} cycles")
        
        return {
            'bst_lut': bst_file,
            'top': top_file,
            'testbench': tb_file
        }

def main():
    parser = argparse.ArgumentParser(description='Generate Solution-Level Time-Division PDAQP hardware')
    parser.add_argument('config_file', help='Configuration .vh file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('-w', '--axi-width', type=int, default=32, 
                       help='Input AXI4-Stream width (default: 32)')
    parser.add_argument('-W', '--output-axi-width', type=int, default=None,
                       help='Output AXI4-Stream width (default: same as input)')
    parser.add_argument('-d', '--dsp-blocks', type=int, default=8,
                       help='Number of available DSP blocks (default: 8)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    config_parser = VerilogConfigParser(args.config_file)
    generator = SolutionLevelHardwareGenerator(
        config_parser, 
        args.output, 
        args.axi_width, 
        args.output_axi_width,
        args.dsp_blocks
    )
    generated_files = generator.generate_all()
    
    if args.verbose:
        print("\nGenerated files:")
        for file_type, file_path in generated_files.items():
            print(f"  {file_type}: {file_path} ({file_path.stat().st_size} bytes)")

if __name__ == '__main__':
    main()