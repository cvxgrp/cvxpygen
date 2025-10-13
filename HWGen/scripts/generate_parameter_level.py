#!/usr/bin/env python3
"""
PDAQP Hardware Generator - Parameter-Level Time-Division Architecture
For cases where DSP blocks < number of parameters
"""

import re
import os
import sys
import argparse
from pathlib import Path
import math

class VerilogConfigParser:
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
        self.validate_required_defines()
    
    def validate_required_defines(self):
        required_defines = {
            'OUTPUT_FRAC_BITS': 14,
            'OUTPUT_INT_BITS': 2,
            'OUTPUT_DATA_WIDTH': 16,
            'HALFPLANE_FRAC_BITS': 14,
            'HALFPLANE_INT_BITS': 2,
            'FEEDBACK_FRAC_BITS': 14,
            'FEEDBACK_INT_BITS': 2
        }
        
        for name, default in required_defines.items():
            if name not in self.defines:
                self.defines[name] = default
    
    def get_define(self, name, default=None):
        return self.defines.get(name, default)
    
    def get_int_define(self, name, default=0):
        value = self.defines.get(name, default)
        return int(value) if isinstance(value, str) and value.isdigit() else value

class ParameterLevelHardwareGenerator:
    def __init__(self, config_parser, output_dir=None, axi_width=32, dsp_blocks=8):
        self.config = config_parser
        self.axi_width = axi_width
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
        
        self.n_parameters = self.config.get_int_define('PDAQP_N_PARAMETER', 2)
        self.n_solutions = self.config.get_int_define('PDAQP_N_SOLUTION', 2)
        self.bst_depth = self.config.get_int_define('PDAQP_ESTIMATED_BST_DEPTH', 7)
        
        # Calculate I/O widths and cycles
        self.total_input_bits = self.n_parameters * 16
        self.total_output_bits = self.n_solutions * 16
        self.input_cycles = (self.total_input_bits + self.axi_width - 1) // self.axi_width
        self.output_cycles = (self.total_output_bits + self.axi_width - 1) // self.axi_width
        
        # Verify parameter-level time-division is appropriate
        self.verify_strategy()
        
        # Calculate parameter batches
        self.param_batches = math.ceil(self.n_parameters / self.dsp_blocks)
        self.solution_cycles = 3 * self.n_solutions * self.param_batches
        self.total_latency = self.bst_depth + self.solution_cycles + 1
        
        print(f"\nParameter-Level Time-Division Configuration:")
        print(f"  Parameters: {self.n_parameters}, Solutions: {self.n_solutions}")
        print(f"  DSP blocks: {self.dsp_blocks}")
        print(f"  Parameter batches: {self.param_batches}")
        print(f"  Solution cycles: {self.solution_cycles}")
        print(f"  I/O: {self.input_cycles} input cycles, {self.output_cycles} output cycles")
        print(f"  Total latency: {self.total_latency} cycles")

    def verify_strategy(self):
        """Verify parameter-level strategy is appropriate"""
        if self.dsp_blocks >= self.n_parameters:
            print(f"WARNING: {self.dsp_blocks} DSPs >= {self.n_parameters} parameters")
            print(f"Consider using solution-level time-division instead")
        elif self.dsp_blocks == 0:
            print(f"ERROR: No DSP blocks available")
            sys.exit(1)

    def generate_top_module(self):
        """Generate TOP module matching the exact pattern"""
        project_name = self.config.project_name
        
        # Generate parameter buffer cases
        param_buffer_cases = []
        for i in range(self.input_cycles):
            if i == 0:
                param_buffer_cases.append(
                    f"                        param_buffer[{min((i+1)*self.axi_width-1, self.total_input_bits-1)}:0] <= s_axis_tdata;\n"
                    f"                        receive_count <= 1;\n"
                    f"                        state <= STATE_PROCESSING;\n"
                    f"                        param_valid <= 0;\n"
                    f"                        $display(\"[TOP] RX[0]: Writing %08h to buffer[{min((i+1)*self.axi_width-1, self.total_input_bits-1)}:0]\", s_axis_tdata);"
                )
            else:
                case_lines = []
                case_lines.append(f"                            {i}: begin")
                if i < self.input_cycles - 1:
                    case_lines.append(f"                                param_buffer[{min((i+1)*self.axi_width-1, self.total_input_bits-1)}:{i*self.axi_width}] <= s_axis_tdata;")
                    case_lines.append(f"                                $display(\"[TOP] RX[{i}]: Writing %08h to buffer[{min((i+1)*self.axi_width-1, self.total_input_bits-1)}:{i*self.axi_width}]\", s_axis_tdata);")
                else:
                    # Last cycle - extract parameters
                    case_lines.append(f"                                param_buffer[{self.total_input_bits-1}:{i*self.axi_width}] <= s_axis_tdata;")
                    case_lines.append(f"                                $display(\"[TOP] RX[{i}]: Writing %08h to buffer[{self.total_input_bits-1}:{i*self.axi_width}]\", s_axis_tdata);")
                    case_lines.append(f"                                $display(\"[TOP] All {self.input_cycles} cycles received, starting BST\");")
                    case_lines.append("")
                    
                    # Extract parameters from buffer
                    for j in range(self.n_parameters):
                        if j < self.n_parameters - 1:
                            case_lines.append(f"                                param_{j} <= param_buffer[{j*16+15}:{j*16}];")
                        else:
                            # Last parameter might come from current data
                            if (self.n_parameters - 1) * 16 >= (self.input_cycles - 1) * self.axi_width:
                                offset = (self.n_parameters - 1) * 16 - (self.input_cycles - 1) * self.axi_width
                                case_lines.append(f"                                param_{j} <= s_axis_tdata[{offset+15}:{offset}];")
                            else:
                                case_lines.append(f"                                param_{j} <= param_buffer[{j*16+15}:{j*16}];")
                    
                    case_lines.append("                                param_valid <= 1;")
                    case_lines.append("")
                    
                    # Debug output
                    if self.input_cycles > 1:
                        buffer_data = "{" + f"s_axis_tdata, param_buffer[{(self.input_cycles-1)*self.axi_width-1}:0]" + "}"
                    else:
                        buffer_data = "s_axis_tdata"
                    case_lines.append(f"                                $display(\"[TOP] COMPLETE BUFFER: %0{self.total_input_bits//4}h\", {buffer_data});")
                    
                    param_list = ' '.join([f'param_{i}=%04h' for i in range(self.n_parameters)])
                    param_vals = []
                    for j in range(self.n_parameters - 1):
                        param_vals.append(f"param_buffer[{j*16+15}:{j*16}]")
                    if (self.n_parameters - 1) * 16 >= (self.input_cycles - 1) * self.axi_width:
                        offset = (self.n_parameters - 1) * 16 - (self.input_cycles - 1) * self.axi_width
                        param_vals.append(f"s_axis_tdata[{offset+15}:{offset}]")
                    else:
                        param_vals.append(f"param_buffer[{(self.n_parameters-1)*16+15}:{(self.n_parameters-1)*16}]")
                    
                    case_lines.append(f"                                $display(\"[TOP] EXTRACTED PARAMS: {param_list}\", ")
                    case_lines.append(f"                                    {', '.join(param_vals)});")
                
                case_lines.append("                            end")
                param_buffer_cases.extend(case_lines)

        # Generate parameter and solution declarations
        param_decls = []
        for i in range(self.n_parameters):
            if i == self.n_parameters - 1:
                param_decls.append(f"    reg signed [15:0] param_{i};")
            else:
                param_decls.append(f"    reg signed [15:0] param_{i}, " + ', '.join([f"param_{j}" for j in range(i+1, min(i+5, self.n_parameters))]) + ";")
                i = min(i+4, self.n_parameters-1)
        
        sol_decls = []
        for i in range(self.n_solutions):
            if i == self.n_solutions - 1:
                sol_decls.append(f"    wire signed [15:0] sol_{i};")
            else:
                sol_decls.append(f"    wire signed [15:0] sol_{i}, " + ', '.join([f"sol_{j}" for j in range(i+1, min(i+5, self.n_solutions))]) + ";")
                i = min(i+4, self.n_solutions-1)

        # Generate output assignment
        output_assignment = "{" + ', '.join([f"sol_{i}" for i in reversed(range(self.n_solutions))]) + "}"

        template = f"""`timescale 1ns/1ps

`include "include/{self.config.project_name}_config.vh"

module {project_name}_top (
    input                               clk,
    input                               rst_n,
    input  [{self.axi_width-1}:0]       s_axis_tdata,
    input                               s_axis_tvalid,
    output                              s_axis_tready,
    output [{self.total_output_bits-1}:0] m_axis_tdata,
    output                              m_axis_tvalid,
    input                               m_axis_tready
);

    // State definitions
    localparam [1:0] STATE_IDLE = 2'b00;
    localparam [1:0] STATE_PROCESSING = 2'b10;
    localparam [1:0] STATE_OUTPUT = 2'b11;
    
    reg [1:0] state;
    reg [{int(math.log2(max(self.input_cycles, 3))+1)-1}:0] receive_count;
    reg [{self.total_input_bits-1}:0] param_buffer;
    
    // BST LUT interface
{''.join(param_decls)}
    reg param_valid;
{''.join(sol_decls)}
    wire sol_valid;
    
    // Output registers
    reg [{self.total_output_bits-1}:0] output_data_reg;
    reg output_valid_reg;
    
    // BST LUT instance
    {project_name}_bst_lut bst_inst (
        .clk(clk),
        .rst_n(rst_n),
{chr(10).join([f'        .param_in_{i}(param_{i}),' for i in range(self.n_parameters)])}
        .valid_in(param_valid),
{chr(10).join([f'        .sol_out_{i}(sol_{i}),' for i in range(self.n_solutions)])}
        .valid_out(sol_valid)
    );
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            receive_count <= 0;
            param_buffer <= {self.total_input_bits}'d0;
            param_valid <= 0;
            output_data_reg <= {self.total_output_bits}'d0;
            output_valid_reg <= 0;
{chr(10).join([f'            param_{i} <= 16\'d0;' for i in range(self.n_parameters)])}
        end else begin
            case (state)
                STATE_IDLE: begin
                    if (s_axis_tvalid && s_axis_tready) begin
{param_buffer_cases[0]}
                    end
                end
                
                STATE_PROCESSING: begin
                    if (s_axis_tvalid && s_axis_tready && receive_count < {self.input_cycles}) begin
                        case (receive_count)
{chr(10).join(param_buffer_cases[1:])}
                        endcase
                        receive_count <= receive_count + 1;
                    end else begin
                        param_valid <= 0;
                    end
                    
                    if (sol_valid) begin
                        output_data_reg <= {output_assignment};
                        output_valid_reg <= 1;
                        state <= STATE_OUTPUT;
                        $display("[TOP] BST completed, output: %0{self.total_output_bits//4}h", {output_assignment});
                    end
                end
                
                STATE_OUTPUT: begin
                    param_valid <= 0;
                    if (m_axis_tvalid && m_axis_tready) begin
                        output_valid_reg <= 0;
                        state <= STATE_IDLE;
                        receive_count <= 0;
                        $display("[TOP] Output transmitted, returning to IDLE");
                    end
                end
                
                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end
    
    // Output assignments
    assign s_axis_tready = (state == STATE_IDLE) || (state == STATE_PROCESSING && receive_count < {self.input_cycles});
    assign m_axis_tdata = output_data_reg;
    assign m_axis_tvalid = output_valid_reg && (state == STATE_OUTPUT);

endmodule"""
        
        output_file = self.rtl_dir / f"{project_name}_top.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated TOP module: {output_file}")
        return output_file

    def generate_bst_lut_module(self):
        """Generate BST LUT module with parameter-level time-division"""
        project_name = self.config.project_name
        
        # Generate parameter inputs/outputs
        param_inputs = [f"    input  signed [15:0]                param_in_{i}," for i in range(self.n_parameters)]
        sol_outputs = [f"    output reg signed [15:0]            sol_out_{i}," for i in range(self.n_solutions)]
        
        # Generate pipeline registers
        param_pipes = [f"    reg [15:0] param{i}_pipe[0:MAX_BST_DEPTH-1];" for i in range(self.n_parameters)]
        param_reset = [f"                param{i}_pipe[i] <= 0;" for i in range(self.n_parameters)]
        input_assigns = [f"                param{i}_pipe[0] <= param_in_{i};" for i in range(self.n_parameters)]
        param_props = [f"                param{i}_pipe[i+1] <= param{i}_pipe[i];" for i in range(self.n_parameters)]
        
        # Generate stored parameter registers
        stored_params = []
        stored_param_reset = []
        stored_param_assigns = []
        for i in range(self.n_parameters):
            if i == self.n_parameters - 1:
                stored_params.append(f"    reg [15:0] stored_param_{i};")
            else:
                params_in_line = min(5, self.n_parameters - i)
                stored_params.append(f"    reg [15:0] " + ', '.join([f"stored_param_{j}" for j in range(i, i + params_in_line)]) + ";")
                i = i + params_in_line - 1
            
            stored_param_reset.append(f"            stored_param_{i} <= 0;")
            stored_param_assigns.append(f"                stored_param_{i} <= param{i}_pipe[MAX_BST_DEPTH-1];")
        
        # Generate DSP declarations and resets
        dsp_declarations = []
        dsp_reset = []
        dsp_mult_lines = []
        
        # Declare DSPs in groups
        for i in range(0, self.dsp_blocks, 4):
            group_size = min(4, self.dsp_blocks - i)
            dsp_declarations.append(f"    (* use_dsp = \"yes\" *) reg signed [15:0] " + ', '.join([f"dsp_a_{j}" for j in range(i, i + group_size)]) + ";")
            dsp_declarations.append(f"    (* use_dsp = \"yes\" *) reg signed [15:0] " + ', '.join([f"dsp_b_{j}" for j in range(i, i + group_size)]) + ";")
            dsp_declarations.append(f"    (* use_dsp = \"yes\" *) reg signed [31:0] " + ', '.join([f"dsp_p_{j}" for j in range(i, i + group_size)]) + ";")
            
            for j in range(i, i + group_size):
                dsp_reset.append(f"            dsp_a_{j} <= 0; dsp_b_{j} <= 0;")
                dsp_mult_lines.append(f"        dsp_p_{j} <= dsp_a_{j} * dsp_b_{j};")
        
        # Generate solution accumulators
        sol_accum_decl = []
        sol_accum_reset = []
        
        for i in range(0, self.n_solutions, 3):
            group_size = min(3, self.n_solutions - i)
            sol_accum_decl.append(f"    reg signed [47:0] " + ', '.join([f"solution_accumulator_{j}" for j in range(i, i + group_size)]) + ";")
            for j in range(i, i + group_size):
                sol_accum_reset.append(f"            solution_accumulator_{j} <= 0;")
        
        output_resets = [f"            sol_out_{i} <= 0;" for i in range(self.n_solutions)]
        
        # Generate halfplane evaluation
        hp_val_terms = []
        for i in range(self.n_parameters):
            hp_val_terms.append(f"($signed(param{i}_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+{i}]))")
        hp_val_expr = " + \n                                ".join(hp_val_terms)
        
        # Generate DSP loading logic
        dsp_load_logic = self.generate_dsp_load_logic()
        
        # Generate accumulation logic
        accum_logic = self.generate_accumulation_logic()
        
        # Generate final output computation
        output_assigns = []
        for i in range(self.n_solutions):
            output_assigns.append(
                f"                    sol_out_{i} <= (solution_accumulator_{i} + (feedbacks[feedback_base_addr + {i} * (`PDAQP_N_PARAMETER + 1) + `PDAQP_N_PARAMETER] << `FEEDBACK_FRAC_BITS)) >> `FEEDBACK_FRAC_BITS;"
            )

        template = f"""`timescale 1ns/1ps

`include "include/{self.config.project_name}_config.vh"

module {project_name}_bst_lut (
    input                               clk,
    input                               rst_n,
{chr(10).join(param_inputs)}
    input                               valid_in,
{chr(10).join(sol_outputs)}
    output reg                          valid_out
);

    (* rom_style = "distributed" *) reg [15:0] halfplanes [0:`PDAQP_HALFPLANES-1];
    (* rom_style = "distributed" *) reg [15:0] feedbacks [0:`PDAQP_FEEDBACKS-1];
    (* rom_style = "distributed" *) reg [7:0] hp_list [0:`PDAQP_TREE_NODES-1];
    (* rom_style = "distributed" *) reg [7:0] jump_list [0:`PDAQP_TREE_NODES-1];

    initial begin
        $readmemh("include/{self.config.project_name}_halfplanes.mem", halfplanes);
        $readmemh("include/{self.config.project_name}_feedbacks.mem", feedbacks);
        $readmemh("include/{self.config.project_name}_hp_list.mem", hp_list);
        $readmemh("include/{self.config.project_name}_jump_list.mem", jump_list);
    end
    
    localparam MAX_BST_DEPTH = `PDAQP_ESTIMATED_BST_DEPTH;
    
    reg valid_pipe[0:MAX_BST_DEPTH-1];
{chr(10).join(param_pipes)}
    
    reg [7:0] current_id_pipe[0:MAX_BST_DEPTH-1];
    reg [7:0] next_id_pipe[0:MAX_BST_DEPTH-1];
    reg [7:0] hp_idx_pipe[0:MAX_BST_DEPTH-1];
    reg traversal_done_pipe[0:MAX_BST_DEPTH-1];
    
{chr(10).join(dsp_declarations)}
    
    always @(posedge clk) begin
{chr(10).join(dsp_mult_lines)}
    end
    
    reg [3:0] dsp_state;
    reg [2:0] current_solution;
    reg [{int(math.log2(self.param_batches+1))-1}:0] param_cycle;
    reg [15:0] feedback_base_addr;
{chr(10).join(sol_accum_decl)}
    

{chr(10).join(stored_params)}
    
    (* use_dsp = "no" *) reg signed [31:0] hp_val;
    (* use_dsp = "no" *) reg signed [31:0] hp_thresh;
    reg decision;
    reg [7:0] new_id;
    
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset BST pipeline
            for (i = 0; i < MAX_BST_DEPTH; i = i + 1) begin
                valid_pipe[i] <= 0;
{chr(10).join(param_reset)}
                current_id_pipe[i] <= 0;
                next_id_pipe[i] <= 0;
                hp_idx_pipe[i] <= 0;
                traversal_done_pipe[i] <= 0;
            end
            
            // Reset DSP
{chr(10).join(dsp_reset)}
            
            // Reset accumulators
{chr(10).join(sol_accum_reset)}
            
            // Reset stored params
{chr(10).join(stored_param_reset)}
            
            // Reset outputs
            valid_out <= 0;
{chr(10).join(output_resets)}
            
            dsp_state <= 0;
            current_solution <= 0;
            param_cycle <= 0;
            feedback_base_addr <= 0;
            
        end else begin
            valid_pipe[0] <= valid_in;
            if (valid_in) begin
{chr(10).join(input_assigns)}
                current_id_pipe[0] <= 0;
                next_id_pipe[0] <= jump_list[0];
                hp_idx_pipe[0] <= hp_list[0];
                traversal_done_pipe[0] <= (jump_list[0] == 0);
                $display("[BST] INPUT: Starting BST");
            end

            for (i = 0; i < MAX_BST_DEPTH-1; i = i + 1) begin
                valid_pipe[i+1] <= valid_pipe[i];
{chr(10).join(param_props)}
                
                if (valid_pipe[i]) begin
                    if (traversal_done_pipe[i]) begin
                        current_id_pipe[i+1] <= current_id_pipe[i];
                        next_id_pipe[i+1] <= next_id_pipe[i];
                        hp_idx_pipe[i+1] <= hp_idx_pipe[i];
                        traversal_done_pipe[i+1] <= 1;
                    end else begin
                        hp_val = {hp_val_expr};
                        
                        hp_thresh = $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+`PDAQP_N_PARAMETER]) << `FEEDBACK_FRAC_BITS;
                        decision = hp_val <= hp_thresh;
                        
                        if (decision) begin
                            new_id = next_id_pipe[i] + 1;
                        end else begin
                            new_id = next_id_pipe[i];
                        end
                        
                        current_id_pipe[i+1] <= new_id;
                        next_id_pipe[i+1] <= new_id + jump_list[new_id];
                        hp_idx_pipe[i+1] <= hp_list[new_id];
                        traversal_done_pipe[i+1] <= (jump_list[new_id] == 0);
                    end
                end
            end
            
            if (valid_pipe[MAX_BST_DEPTH-1] && dsp_state == 0) begin
                feedback_base_addr <= hp_idx_pipe[MAX_BST_DEPTH-1] * (`PDAQP_N_PARAMETER + 1) * `PDAQP_N_SOLUTION;
                dsp_state <= 1;
                current_solution <= 0;
                param_cycle <= 0;
                
{chr(10).join(stored_param_assigns)}
                
{chr(10).join([f'                solution_accumulator_{i} <= 0;' for i in range(self.n_solutions)])}
                
            end
            
            case (dsp_state)
                1: begin

{dsp_load_logic}
                    dsp_state <= 2;
                end
                
                2: begin
                    dsp_state <= 3;
                end
                
                3: begin
{accum_logic}
                end
                
                4: begin
{chr(10).join(output_assigns)}
                    
                    valid_out <= 1;
                    dsp_state <= 0;
                end
                
                default: begin
                    valid_out <= 0;
                end
            endcase
        end
    end
endmodule"""
        
        output_file = self.rtl_dir / f"{project_name}_bst_lut.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated BST LUT module")
        return output_file

    def generate_dsp_load_logic(self):
        """Generate DSP loading logic for parameter-level time-division"""
        logic = []
        
        # Generate loading for each parameter cycle
        for cycle in range(self.param_batches):
            if cycle == 0:
                logic.append(f"                    if (param_cycle == {cycle}) begin")
            else:
                logic.append(f"                    end else begin")
            
            # Load parameters for this cycle
            for i in range(self.dsp_blocks):
                param_idx = cycle * self.dsp_blocks + i
                if param_idx < self.n_parameters:
                    logic.append(f"                        dsp_a_{i} <= stored_param_{param_idx}; dsp_b_{i} <= feedbacks[feedback_base_addr + current_solution * (`PDAQP_N_PARAMETER + 1) + {param_idx}];")
                else:
                    logic.append(f"                        dsp_a_{i} <= 0; dsp_b_{i} <= 0;")
        
        logic.append("                    end")
        return '\n'.join(logic)

    def generate_accumulation_logic(self):
        """Generate accumulation logic for parameter-level time-division"""
        logic = []
        
        # Accumulate DSP results
        sum_expr = " + ".join([f"dsp_p_{i}" for i in range(self.dsp_blocks)])
        
        logic.append("                    case (current_solution)")
        for sol in range(self.n_solutions):
            logic.append(f"                        {sol}: solution_accumulator_{sol} <= solution_accumulator_{sol} + {sum_expr};")
        logic.append("                    endcase")
        logic.append("                    ")
        
        # State transition
        logic.append(f"                    if (param_cycle == {self.param_batches - 1}) begin")
        logic.append("                        param_cycle <= 0;")
        logic.append("                        if (current_solution == `PDAQP_N_SOLUTION - 1) begin")
        logic.append("                            dsp_state <= 4;")
        logic.append("                        end else begin")
        logic.append("                            current_solution <= current_solution + 1;")
        logic.append("                            dsp_state <= 1;")
        logic.append("                        end")
        logic.append("                    end else begin")
        logic.append("                        param_cycle <= 1;")
        logic.append("                        dsp_state <= 1;")
        logic.append("                    end")
        
        return '\n'.join(logic)

    def generate_testbench(self):
        """Generate testbench matching the exact pattern"""
        project_name = self.config.project_name
        
        # Generate test parameter declarations
        param_test_data = [f"    reg [15:0] test_params_{i} [0:TEST_VECTORS-1];" for i in range(self.n_parameters)]
        
        # Generate parameter initialization
        param_init = []
        for i in range(self.n_parameters):
            param_init.append(f"            test_params_{i}[i] = 16'h{i:02x}00 + i*16;")
        
        # Generate send task parameter packing
        param_pack = []
        for i in range(self.n_parameters):
            param_pack.append(f"            param_data[{i*16+15}:{i*16}] = test_params_{i}[test_vector_idx];")
        
        # Generate groundtruth parameter sum
        param_sum = " +\n                      ".join([f"$signed(p{i}) * $signed(halfplanes[disp+{i}])" for i in range(self.n_parameters)])
        
        # Generate groundtruth solution calculation
        sol_calc = []
        for i in range(self.n_parameters):
            sol_calc.append(f"                         $signed(p{i}) * $signed(feedbacks[feedback_base + j*(`PDAQP_N_PARAMETER+1) + {i}])")
        sol_sum = " +\n".join(sol_calc)
        
        # Generate solution packing
        sol_pack = "{" + ", ".join([f"sol[{i}][15:0]" for i in reversed(range(self.n_solutions))]) + "}"

        template = f"""`timescale 1ns/1ps
`include "include/{self.config.project_name}_config.vh"

module {project_name}_tb;

    localparam CLK_PERIOD = 10;
    localparam SIM_CYCLES = 2000000;
    localparam TEST_VECTORS = 100;
    localparam PIPELINE_DELAY = 20;
    
    reg clk;
    reg rst_n;
    reg [{self.axi_width-1}:0] s_axis_tdata;
    reg s_axis_tvalid;
    wire s_axis_tready;
    wire [{self.total_output_bits-1}:0] m_axis_tdata;
    wire m_axis_tvalid;
    reg m_axis_tready;
    
{chr(10).join(param_test_data)}
    reg [{self.total_output_bits-1}:0] expected_results [0:TEST_VECTORS-1];
    reg [{self.total_output_bits-1}:0] actual_results [0:TEST_VECTORS-1];
    integer output_count;
    integer test_idx;
    integer errors;
    integer i;
    
    reg [15:0] halfplanes [0:`PDAQP_HALFPLANES-1];
    reg [15:0] feedbacks [0:`PDAQP_FEEDBACKS-1];
    reg [7:0] hp_list [0:`PDAQP_TREE_NODES-1];
    reg [7:0] jump_list [0:`PDAQP_TREE_NODES-1];
    
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
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    initial begin
        $readmemh("include/{self.config.project_name}_halfplanes.mem", halfplanes);
        $readmemh("include/{self.config.project_name}_feedbacks.mem", feedbacks);
        $readmemh("include/{self.config.project_name}_hp_list.mem", hp_list);
        $readmemh("include/{self.config.project_name}_jump_list.mem", jump_list);
        
        $display("[TB] Memory loaded successfully");
    end

    task send_parameter_data;
        input integer test_vector_idx;
        reg [{self.total_input_bits-1}:0] param_data;
        integer cycle;
        begin
            param_data = 0;
{chr(10).join(param_pack)}
            
            if (test_vector_idx < 5) begin
                $display("[TB] === SENDING VECTOR %0d ===", test_vector_idx);
                $write("     Params:");
                for (i = 0; i < `PDAQP_N_PARAMETER; i = i + 1) begin
                    $write(" %h", param_data[i*16 +: 16]);
                end
                $display("");
            end
            
            for (cycle = 0; cycle < {self.input_cycles}; cycle = cycle + 1) begin
                @(posedge clk);
                s_axis_tdata = param_data[cycle*{self.axi_width} +: {self.axi_width}];
                s_axis_tvalid = 1;
                
                while (!s_axis_tready) @(posedge clk);
                if (test_vector_idx < 5) $display("     Cycle %0d: %h", cycle+1, s_axis_tdata);
            end
            
            @(posedge clk);
            s_axis_tvalid = 0;
        end
    endtask

    function [{self.total_output_bits-1}:0] calculate_groundtruth;
        input [15:0] {', '.join([f'p{i}' for i in range(self.n_parameters)])};
        
        reg [7:0] id, next_id;
        reg [15:0] disp, feedback_base;
        reg signed [31:0] val;
        reg signed [31:0] thresh;
        reg signed [31:0] sol [0:`PDAQP_N_SOLUTION-1];
        integer j, iter;
        
        begin
            id = 0;
            next_id = id + jump_list[id];
            iter = 0;
            
            while (next_id != id && iter < 50) begin
                disp = hp_list[id] * (`PDAQP_N_PARAMETER + 1);
                val = {param_sum};
                thresh = $signed(halfplanes[disp+`PDAQP_N_PARAMETER]) << `FEEDBACK_FRAC_BITS;
                
                if (val <= thresh) begin
                    id = next_id + 1;
                end else begin
                    id = next_id;
                end
                next_id = id + jump_list[id];
                iter = iter + 1;
            end
            
            feedback_base = hp_list[id] * (`PDAQP_N_PARAMETER + 1) * `PDAQP_N_SOLUTION;
            
            for (j = 0; j < `PDAQP_N_SOLUTION; j = j + 1) begin
                sol[j] = {sol_sum} +
                         ($signed(feedbacks[feedback_base + j*(`PDAQP_N_PARAMETER+1) + `PDAQP_N_PARAMETER]) << `FEEDBACK_FRAC_BITS);
                
                sol[j] = sol[j][`FEEDBACK_FRAC_BITS + `OUTPUT_DATA_WIDTH - 1:`FEEDBACK_FRAC_BITS];
            end
            
            calculate_groundtruth = {sol_pack};
        end
    endfunction

    initial begin
        rst_n = 0;
        s_axis_tvalid = 0;
        s_axis_tdata = 0;
        m_axis_tready = 1;
        output_count = 0;
        errors = 0;
        
        // Generate test vectors
        for (i = 0; i < TEST_VECTORS; i = i + 1) begin
{chr(10).join(param_init)}
        end
        
        $display("\\n=== PDAQP Universal Test ({self.n_parameters}p/{self.n_solutions}s) ===");
        $display("Config: %0dp/%0ds, AXI=%0db, %0dc/vec", 
                `PDAQP_N_PARAMETER, `PDAQP_N_SOLUTION, {self.axi_width}, {self.input_cycles});
        $display("BST depth from config: %0d (pipeline: %0d stages)", `PDAQP_ESTIMATED_BST_DEPTH, `PDAQP_ESTIMATED_BST_DEPTH + 4);
        
        // Calculate expected results
        for (i = 0; i < TEST_VECTORS; i = i + 1) begin
            if (i < 3) $display("\\n=== Computing Expected Result %0d ===", i);
            expected_results[i] = calculate_groundtruth({', '.join([f'test_params_{i}[i]' for i in range(self.n_parameters)])});
            if (i < 3) $display("Expected[%0d] = %h", i, expected_results[i]);
        end
        
        repeat(20) @(posedge clk);
        rst_n = 1;
        repeat(20) @(posedge clk);
        
        // Send test vectors
        for (test_idx = 0; test_idx < TEST_VECTORS; test_idx = test_idx + 1) begin
            if (test_idx < 5) $display("\\n[%0t] === TESTING VECTOR %0d ===", $time, test_idx);
            send_parameter_data(test_idx);
            repeat(PIPELINE_DELAY) @(posedge clk);
        end
        
        repeat(500) @(posedge clk);
        
        $display("\\n=== FINAL RESULTS ===");
        $display("Vectors processed: %0d/%0d", output_count, TEST_VECTORS);
        $display("Errors: %0d", errors);
        $display("Actual BST depth used: %0d", `PDAQP_ESTIMATED_BST_DEPTH);
        
        for (i = 0; i < TEST_VECTORS && i < output_count; i = i + 1) begin
            $display("V%0d: Expected=%h, Actual=%h, %s", 
                    i, expected_results[i], actual_results[i], 
                    (expected_results[i] == actual_results[i]) ? "PASS" : "FAIL");
        end
        
        if (errors == 0 && output_count >= TEST_VECTORS)
            $display("\\n*** ALL %0d TESTS PASSED ***", TEST_VECTORS);
        else
            $display("\\n*** TESTS FAILED: %0d errors, %0d/%0d vectors ***", errors, output_count, TEST_VECTORS);
        
        $finish;
    end
    
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            if (output_count < TEST_VECTORS) begin
                actual_results[output_count] = m_axis_tdata;
                
                if (m_axis_tdata !== expected_results[output_count]) begin
                    if (output_count < 10) begin
                        $display("[%0t] *** ERROR V%0d: exp=%h, got=%h ***", 
                                $time, output_count, expected_results[output_count], m_axis_tdata);
                    end
                    errors = errors + 1;
                end else begin
                    if (output_count < 10) begin
                        $display("[%0t] *** PASS V%0d: %h ***", $time, output_count, m_axis_tdata);
                    end
                end
            end
            output_count = output_count + 1;
        end
    end
    
    initial begin
        #(SIM_CYCLES * CLK_PERIOD);
        $display("TIMEOUT - %0d outputs received", output_count);
        $finish;
    end
    
    initial begin
        $dumpfile("{project_name}_tb.vcd");
        $dumpvars(0, {project_name}_tb);
    end
    
endmodule"""
        
        output_file = self.tb_dir / f"{project_name}_tb.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated testbench: {output_file}")
        return output_file

    def generate_all(self):
        """Generate all files"""
        print(f"\nGenerating Parameter-Level Time-Division PDAQP Hardware")
        print(f"=" * 50)
        
        bst_file = self.generate_bst_lut_module()
        top_file = self.generate_top_module()
        tb_file = self.generate_testbench()
        
        print(f"\n✓ Generation complete!")
        print(f"✓ Parameter-level time-division with {self.dsp_blocks} DSPs")
        print(f"✓ Processing {self.n_parameters} parameters in {self.param_batches} batches")
        print(f"✓ Total solution computation: {self.solution_cycles} cycles")
        
        return {
            'bst_lut': bst_file,
            'top': top_file,
            'testbench': tb_file
        }

def main():
    parser = argparse.ArgumentParser(description='Generate Parameter-Level Time-Division PDAQP hardware')
    parser.add_argument('config_file', help='Configuration .vh file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('-w', '--axi-width', type=int, default=32, 
                       choices=[8, 16, 32, 64, 128, 256, 512, 1024],
                       help='AXI4-Stream width (default: 32)')
    parser.add_argument('-d', '--dsp-blocks', type=int, default=8,
                       help='Number of available DSP blocks (default: 8)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    config_parser = VerilogConfigParser(args.config_file)
    generator = ParameterLevelHardwareGenerator(config_parser, args.output, args.axi_width, args.dsp_blocks)
    generated_files = generator.generate_all()
    
    if args.verbose:
        print("\nGenerated files:")
        for file_type, file_path in generated_files.items():
            print(f"  {file_type}: {file_path} ({file_path.stat().st_size} bytes)")

if __name__ == '__main__':
    main()