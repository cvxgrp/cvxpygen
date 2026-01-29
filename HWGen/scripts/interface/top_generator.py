"""
PDAQP Top Module Generator with Configurable Input Buffering
Generates top-level wrapper with AXI4-Stream interface and optional input buffer
BST interface remains simple and flat (param_in_0, param_in_1, ...)
"""

import os
from datetime import datetime


class TopModuleGenerator:
    """Generator for PDAQP top module with pipelined AXI4-Stream interface"""
    
    def __init__(self, config, strategy, enable_input_buffer=True):
        """
        Initialize top module generator
        
        Args:
            config: Configuration object from VerilogConfigParser
            strategy: PackingStrategy object
            enable_input_buffer: Enable input buffering (default: True)
        """
        self.config = config
        self.strategy = strategy
        self.enable_input_buffer = enable_input_buffer
        
    def generate(self, output_file, config_include_path, timing_include_path):
        """
        Generate complete top module file
        
        Args:
            output_file: Path to output Verilog file
            config_include_path: Relative path to pdaqp_config.vh
            timing_include_path: Relative path to pdaqp_timing.vh
        """
        with open(output_file, 'w') as f:
            self._write_header(f, config_include_path, timing_include_path)
            self._write_module_declaration(f)
            self._write_internal_signals(f)
            self._write_pipeline_control(f)
            self._write_input_buffering(f)
            self._write_bst_instantiation(f)
            self._write_output_packing(f)
            self._write_module_end(f)
    
    def _write_header(self, f, config_path, timing_path):
        """Write file header"""
        f.write("`timescale 1ns/1ps\n\n")
        f.write("//=======================================================================\n")
        f.write("// PDAQP Top Module - AXI4-Stream Interface\n")
        f.write("//=======================================================================\n")
        f.write(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"// Config: {self.config.n_parameters} params -> {self.config.n_solutions} solutions ")
        f.write(f"({self.config.data_format})\n")
        f.write(f"// Packing: Tier {self.strategy.tier_used}\n")
        
        input_info = self.strategy.input_strategy
        f.write(f"//   Input:  {input_info['mode']} ")
        if input_info['mode'] == 'tdm':
            f.write(f"(batch_size={input_info['batch_size']}, ")
            f.write(f"batches={input_info['num_batches']}, {input_info['port_width']}-bit)\n")
        else:
            f.write(f"({input_info['num_ports']} port(s), {input_info['port_width']}-bit)\n")
        
        output_info = self.strategy.output_strategy
        f.write(f"//   Output: {output_info['mode']} ")
        if output_info['mode'] == 'tdm':
            f.write(f"(batch_size={output_info['batch_size']}, ")
            f.write(f"batches={output_info['num_batches']}, {output_info['port_width']}-bit)\n")
        else:
            f.write(f"({output_info['num_ports']} port(s), {output_info['port_width']}-bit)\n")
        
        f.write(f"// Input Buffer: {'ENABLED' if self.enable_input_buffer else 'DISABLED'}\n")
        f.write("//=======================================================================\n\n")
        
        f.write(f'`include "{config_path}"\n')
        f.write(f'`include "{timing_path}"\n\n')
    
    def _write_module_declaration(self, f):
        """Write module declaration"""
        f.write("module pdaqp_top (\n")
        f.write("    input wire clk,\n")
        f.write("    input wire rst_n,\n")
        f.write("    \n")
        
        self._write_input_ports(f)
        f.write("    \n")
        self._write_output_ports(f)
        
        f.write(");\n\n")
    
    def _write_input_ports(self, f):
        """Write input ports"""
        input_mode = self.strategy.input_strategy['mode']
        num_ports = self.strategy.input_strategy['num_ports']
        port_width = self.strategy.input_strategy['port_width']
        
        f.write(f"    // AXI4-Stream input ({input_mode})\n")
        
        if input_mode == 'tdm':
            batch_id_width = self.strategy.input_strategy['batch_id_width']
            f.write(f"    input wire [{port_width-1}:0] s_axis_tdata,\n")
            # Fix: Use correct bit width for TID
            if batch_id_width > 0:
                f.write(f"    input wire [{batch_id_width-1}:0] s_axis_tid,\n")
            else:
                # Special case: only 1 batch, TID can be omitted or single bit
                f.write(f"    input wire s_axis_tid,\n")
            f.write(f"    input wire s_axis_tvalid,\n")
            f.write(f"    output wire s_axis_tready,\n")
        
        elif num_ports == 1:
            f.write(f"    input wire [{port_width-1}:0] s_axis_tdata,\n")
            f.write(f"    input wire s_axis_tvalid,\n")
            f.write(f"    output wire s_axis_tready,\n")
        else:
            for i in range(num_ports):
                port_name = self.strategy.get_input_port_name(i)
                f.write(f"    input wire [{port_width-1}:0] {port_name}_tdata,\n")
                f.write(f"    input wire {port_name}_tvalid,\n")
            f.write(f"    output wire s_axis_tready,\n")
    
    def _write_output_ports(self, f):
        """Write output ports"""
        output_mode = self.strategy.output_strategy['mode']
        num_ports = self.strategy.output_strategy['num_ports']
        port_width = self.strategy.output_strategy['port_width']
        
        f.write(f"    // AXI4-Stream output ({output_mode})\n")
        
        if output_mode == 'tdm':
            batch_id_width = self.strategy.output_strategy['batch_id_width']
            f.write(f"    output wire [{port_width-1}:0] m_axis_tdata,\n")
            # Fix: Use correct bit width for TID
            if batch_id_width > 0:
                f.write(f"    output wire [{batch_id_width-1}:0] m_axis_tid,\n")
            else:
                f.write(f"    output wire m_axis_tid,\n")
            f.write(f"    output wire m_axis_tvalid,\n")
            f.write(f"    input wire m_axis_tready\n")
        
        elif num_ports == 1:
            f.write(f"    output wire [{port_width-1}:0] m_axis_tdata,\n")
            f.write(f"    output wire m_axis_tvalid,\n")
            f.write(f"    input wire m_axis_tready\n")
        else:
            for i in range(num_ports):
                port_name = self.strategy.get_output_port_name(i)
                f.write(f"    output wire [{port_width-1}:0] {port_name}_tdata,\n")
                f.write(f"    output wire {port_name}_tvalid,\n")
            f.write(f"    input wire m_axis_tready\n")
    
    def _write_internal_signals(self, f):
        """Write internal signals for BST connection"""
        f.write("    //===================================================================\n")
        f.write("    // BST Interface Signals\n")
        f.write("    //===================================================================\n")
        f.write("    \n")
        
        # BST inputs
        for i in range(self.config.n_parameters):
            f.write(f"    wire [`PDAQP_PARAM_{i}_WIDTH-1:0] bst_param_in_{i};\n")
        f.write("    wire bst_valid_in;\n")
        f.write("    wire bst_ready_out;\n")
        f.write("    \n")
        
        # BST outputs
        for i in range(self.config.n_solutions):
            f.write(f"    wire [`PDAQP_SOL_{i}_WIDTH-1:0] bst_solution_{i};\n")
        f.write("    wire bst_valid_out;\n")
        f.write("    \n")
    
    def _write_pipeline_control(self, f):
        """Write pipeline control logic"""
        input_mode = self.strategy.input_strategy['mode']
        output_mode = self.strategy.output_strategy['mode']
        
        f.write("    //===================================================================\n")
        f.write("    // Pipeline Flow Control\n")
        f.write("    //===================================================================\n")
        f.write("    \n")
        
        if input_mode == 'tdm' or output_mode == 'tdm':
            self._write_tdm_control(f)
        else:
            self._write_spatial_control(f)
    
    def _write_spatial_control(self, f):
        """Write control for spatial mode"""
        f.write("    localparam PIPELINE_CAPACITY = `PDAQP_BST_LATENCY;\n")
        f.write("    localparam COUNTER_WIDTH = $clog2(PIPELINE_CAPACITY + 1);\n")
        f.write("    \n")
        f.write("    reg [COUNTER_WIDTH-1:0] requests_in_flight;\n")
        f.write("    wire can_accept = (requests_in_flight < PIPELINE_CAPACITY);\n")
        f.write("    \n")
        
        # Input valid signal
        num_input_ports = self.strategy.input_strategy['num_ports']
        if num_input_ports == 1:
            f.write("    wire all_input_valid = s_axis_tvalid;\n")
        else:
            valid_signals = [f"{self.strategy.get_input_port_name(i)}_tvalid" 
                           for i in range(num_input_ports)]
            f.write(f"    wire all_input_valid = {' && '.join(valid_signals)};\n")
        
        f.write("    wire input_fire = all_input_valid && can_accept;\n")
        
        # Output fire signal
        num_output_ports = self.strategy.output_strategy['num_ports']
        if num_output_ports == 1:
            f.write("    wire output_fire = m_axis_tvalid && m_axis_tready;\n")
        else:
            f.write(f"    wire output_fire = {self.strategy.get_output_port_name(0)}_tvalid && m_axis_tready;\n")
        
        f.write("    \n")
        f.write("    always @(posedge clk or negedge rst_n) begin\n")
        f.write("        if (!rst_n) begin\n")
        f.write("            requests_in_flight <= {COUNTER_WIDTH{1'b0}};\n")
        f.write("        end else begin\n")
        f.write("            case ({input_fire, output_fire})\n")
        f.write("                2'b10: requests_in_flight <= requests_in_flight + 1'd1;\n")
        f.write("                2'b01: requests_in_flight <= requests_in_flight - 1'd1;\n")
        f.write("                default: requests_in_flight <= requests_in_flight;\n")
        f.write("            endcase\n")
        f.write("        end\n")
        f.write("    end\n")
        f.write("    \n")
        f.write("    assign s_axis_tready = can_accept;\n\n")
    
    def _write_tdm_control(self, f):
        """Write control for TDM mode"""
        input_mode = self.strategy.input_strategy['mode']
        output_mode = self.strategy.output_strategy['mode']
        
        f.write("    localparam PIPELINE_CAPACITY = `PDAQP_BST_LATENCY;\n")
        f.write("    localparam COUNTER_WIDTH = $clog2(PIPELINE_CAPACITY + 1);\n")
        
        if input_mode == 'tdm':
            input_batches = self.strategy.input_strategy['num_batches']
            f.write(f"    localparam INPUT_NUM_BATCHES = {input_batches};\n")
        
        if output_mode == 'tdm':
            output_batches = self.strategy.output_strategy['num_batches']
            f.write(f"    localparam OUTPUT_NUM_BATCHES = {output_batches};\n")
        
        f.write("    \n")
        f.write("    reg [COUNTER_WIDTH-1:0] requests_in_flight;\n")
        f.write("    wire can_accept = (requests_in_flight < PIPELINE_CAPACITY);\n")
        
        # Fix: Define output_fire first (needed by both input modes)
        f.write("    wire output_fire = m_axis_tvalid && m_axis_tready;\n")
        
        # Fix: For TDM input, accept individual batches but only count when BST starts
        if input_mode == 'tdm':
            f.write("    wire batch_fire = s_axis_tvalid && can_accept;\n")
            f.write("    wire bst_start;\n")  # Will be defined in input buffering section
            f.write("    \n")
            f.write("    always @(posedge clk or negedge rst_n) begin\n")
            f.write("        if (!rst_n) begin\n")
            f.write("            requests_in_flight <= {COUNTER_WIDTH{1'b0}};\n")
            f.write("        end else begin\n")
            f.write("            case ({bst_start, output_fire})\n")  # Use bst_start instead of batch_fire
            f.write("                2'b10: requests_in_flight <= requests_in_flight + 1'd1;\n")
            f.write("                2'b01: requests_in_flight <= requests_in_flight - 1'd1;\n")
            f.write("                default: requests_in_flight <= requests_in_flight;\n")
            f.write("            endcase\n")
            f.write("        end\n")
            f.write("    end\n")
        else:
            f.write("    wire input_fire = s_axis_tvalid && can_accept;\n")
            f.write("    \n")
            f.write("    always @(posedge clk or negedge rst_n) begin\n")
            f.write("        if (!rst_n) begin\n")
            f.write("            requests_in_flight <= {COUNTER_WIDTH{1'b0}};\n")
            f.write("        end else begin\n")
            f.write("            case ({input_fire, output_fire})\n")
            f.write("                2'b10: requests_in_flight <= requests_in_flight + 1'd1;\n")
            f.write("                2'b01: requests_in_flight <= requests_in_flight - 1'd1;\n")
            f.write("                default: requests_in_flight <= requests_in_flight;\n")
            f.write("            endcase\n")
            f.write("        end\n")
            f.write("    end\n")
        
        f.write("    \n")
        f.write("    assign s_axis_tready = can_accept;\n\n")
    
    def _write_input_buffering(self, f):
        """Write input unpacking and buffering"""
        input_mode = self.strategy.input_strategy['mode']
        
        f.write("    //===================================================================\n")
        f.write("    // Input Unpacking\n")
        f.write("    //===================================================================\n")
        f.write("    \n")
        
        if input_mode == 'tdm':
            self._write_tdm_input_logic(f)
        else:
            self._write_spatial_input_logic(f)
    
    def _write_spatial_input_logic(self, f):
        """Write input logic for spatial mode"""
        f.write(f"    localparam ENABLE_INPUT_BUFFER = {1 if self.enable_input_buffer else 0};\n")
        f.write("    \n")
        
        if self.enable_input_buffer:
            # Buffered path
            for i in range(self.config.n_parameters):
                f.write(f"    reg [`PDAQP_PARAM_{i}_WIDTH-1:0] param_buf_{i};\n")
            f.write("    \n")
            
            f.write("    always @(posedge clk or negedge rst_n) begin\n")
            f.write("        if (!rst_n) begin\n")
            for i in range(self.config.n_parameters):
                f.write(f"            param_buf_{i} <= `PDAQP_PARAM_{i}_WIDTH'b0;\n")
            f.write("        end else if (input_fire) begin\n")
            
            self._write_input_extraction(f, "param_buf", indent="            ")
            
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            
            for i in range(self.config.n_parameters):
                f.write(f"    assign bst_param_in_{i} = param_buf_{i};\n")
            f.write("    \n")
        else:
            # Direct connection
            self._write_input_extraction(f, "bst_param_in", indent="    ", use_assign=True)
            f.write("    \n")
        
        f.write("    assign bst_valid_in = input_fire;\n")
        f.write("    \n")
    
    def _write_tdm_input_logic(self, f):
        """Write TDM input logic (batch reassembly)"""
        batch_size = self.strategy.input_strategy['batch_size']
        num_batches = self.strategy.input_strategy['num_batches']
        
        f.write(f"    localparam BATCH_SIZE = {batch_size};\n")
        f.write(f"    localparam NUM_BATCHES = {num_batches};\n")
        
        if num_batches > 1:
            f.write(f"    localparam BATCH_COUNT_WIDTH = $clog2(NUM_BATCHES);\n")
        else:
            f.write(f"    localparam BATCH_COUNT_WIDTH = 1;\n")
        
        f.write("    \n")
        
        for i in range(self.config.n_parameters):
            f.write(f"    reg [`PDAQP_PARAM_{i}_WIDTH-1:0] tdm_param_buf_{i};\n")
        
        f.write("    reg [BATCH_COUNT_WIDTH-1:0] batch_count;\n")
        
        if num_batches > 1:
            f.write("    wire last_batch = (batch_count == NUM_BATCHES - 1);\n")
        else:
            f.write("    wire last_batch = 1'b1;\n")
        
        # Fix: Define bst_start signal used in control logic
        f.write("    wire start_processing = batch_fire && last_batch;\n")
        f.write("    assign bst_start = start_processing;\n")
        f.write("    \n")
        f.write("    always @(posedge clk or negedge rst_n) begin\n")
        f.write("        if (!rst_n) begin\n")
        f.write("            batch_count <= {BATCH_COUNT_WIDTH{1'b0}};\n")
        for i in range(self.config.n_parameters):
            f.write(f"            tdm_param_buf_{i} <= `PDAQP_PARAM_{i}_WIDTH'b0;\n")
        f.write("        end else if (batch_fire) begin\n")
        
        if num_batches > 1:
            f.write("            batch_count <= last_batch ? {BATCH_COUNT_WIDTH{1'b0}} : batch_count + 1'b1;\n")
        else:
            f.write("            batch_count <= {BATCH_COUNT_WIDTH{1'b0}};\n")
        
        # Fix: Use batch_id_width for case statement
        batch_id_width = self.strategy.input_strategy['batch_id_width']
        if batch_id_width > 0:
            case_width = batch_id_width
        else:
            case_width = 1
        
        f.write("            case (s_axis_tid)\n")
        
        # Batch unpacking
        param_idx = 0
        for batch_id in range(num_batches):
            f.write(f"                {case_width}'d{batch_id}: begin\n")
            params_this_batch = min(batch_size, self.config.n_parameters - param_idx)
            
            for local_idx in range(params_this_batch):
                start_bit = local_idx * self.config.input_width
                end_bit = (local_idx + 1) * self.config.input_width - 1
                f.write(f"                    tdm_param_buf_{param_idx} <= s_axis_tdata[{end_bit}:{start_bit}];\n")
                param_idx += 1
            
            f.write("                end\n")
        
        f.write("                default: begin\n")
        f.write("                end\n")
        f.write("            endcase\n")
        f.write("        end\n")
        f.write("    end\n")
        f.write("    \n")
        
        for i in range(self.config.n_parameters):
            f.write(f"    assign bst_param_in_{i} = tdm_param_buf_{i};\n")
        f.write("    assign bst_valid_in = start_processing;\n")
        f.write("    \n")
    
    def _write_input_extraction(self, f, signal_prefix, indent="    ", use_assign=False):
        """Write param extraction from input ports"""
        input_mode = self.strategy.input_strategy['mode']
        num_ports = self.strategy.input_strategy['num_ports']
        params_per_port = self.strategy.input_strategy['params_per_port']
        
        param_idx = 0
        
        if num_ports == 1:
            for i in range(self.config.n_parameters):
                start_bit = i * self.config.input_width
                end_bit = (i + 1) * self.config.input_width - 1
                
                if use_assign:
                    f.write(f"{indent}assign {signal_prefix}_{i} = s_axis_tdata[{end_bit}:{start_bit}];\n")
                else:
                    f.write(f"{indent}{signal_prefix}_{i} <= s_axis_tdata[{end_bit}:{start_bit}];\n")
        else:
            for port_idx in range(num_ports):
                port_name = self.strategy.get_input_port_name(port_idx)
                n_params_this_port = params_per_port[port_idx]
                
                for local_idx in range(n_params_this_port):
                    start_bit = local_idx * self.config.input_width
                    end_bit = (local_idx + 1) * self.config.input_width - 1
                    
                    if use_assign:
                        f.write(f"{indent}assign {signal_prefix}_{param_idx} = {port_name}_tdata[{end_bit}:{start_bit}];\n")
                    else:
                        f.write(f"{indent}{signal_prefix}_{param_idx} <= {port_name}_tdata[{end_bit}:{start_bit}];\n")
                    
                    param_idx += 1

    def _write_bst_instantiation(self, f):
        """Write BST instantiation"""
        f.write("    //===================================================================\n")
        f.write("    // BST LUT Instantiation\n")
        f.write("    //===================================================================\n")
        f.write("    \n")
        
        f.write("    pdaqp_bst_lut u_bst_lut (\n")
        f.write("        .clk(clk),\n")
        f.write("        .rst_n(rst_n),\n")
        f.write("        .valid_in(bst_valid_in),\n")
        f.write("        \n")
        
        for i in range(self.config.n_parameters):
            f.write(f"        .param_in_{i}(bst_param_in_{i}),\n")
        f.write("        \n")
        
        f.write("        .ready_out(bst_ready_out),\n")
        f.write("        \n")
        
        for i in range(self.config.n_solutions):
            f.write(f"        .solution_{i}(bst_solution_{i}),\n")
        
        f.write("        .valid_out(bst_valid_out)\n")
        f.write("    );\n")
        f.write("    \n")

    def _write_output_packing(self, f):
        """Write output packing"""
        output_mode = self.strategy.output_strategy['mode']
        
        f.write("    //===================================================================\n")
        f.write("    // Output Packing\n")
        f.write("    //===================================================================\n")
        f.write("    \n")
        
        if output_mode == 'tdm':
            self._write_tdm_output_logic(f)
        else:
            self._write_spatial_output_logic(f)
    
    def _write_spatial_output_logic(self, f):
        """Write spatial output logic"""
        num_ports = self.strategy.output_strategy['num_ports']
        sols_per_port = self.strategy.output_strategy['sols_per_port']
        
        sol_idx = 0
        
        if num_ports == 1:
            f.write("    assign m_axis_tdata = {\n")
            
            for i in range(self.config.n_solutions - 1, -1, -1):
                comma = "," if i > 0 else ""
                f.write(f"        bst_solution_{i}{comma}\n")
            
            f.write("    };\n")
            f.write("    assign m_axis_tvalid = bst_valid_out;\n")
            f.write("    \n")
        else:
            for port_idx in range(num_ports):
                port_name = self.strategy.get_output_port_name(port_idx)
                n_sols_this_port = sols_per_port[port_idx]
                
                f.write(f"    assign {port_name}_tdata = {{\n")
                
                port_sols = []
                for _ in range(n_sols_this_port):
                    port_sols.append(sol_idx)
                    sol_idx += 1
                
                for i, sol_num in enumerate(reversed(port_sols)):
                    comma = "," if i < len(port_sols) - 1 else ""
                    f.write(f"        bst_solution_{sol_num}{comma}\n")
                
                f.write("    };\n")
                f.write(f"    assign {port_name}_tvalid = bst_valid_out;\n")
                f.write("    \n")
    
    def _write_tdm_output_logic(self, f):
        """Write TDM output logic (batch generation) with buffering"""
        batch_size = self.strategy.output_strategy['batch_size']
        num_batches = self.strategy.output_strategy['num_batches']
        port_width = self.strategy.output_strategy['port_width']
        batch_id_width = self.strategy.output_strategy['batch_id_width']
        
        f.write(f"    localparam OUT_BATCH_SIZE = {batch_size};\n")
        f.write(f"    localparam OUT_NUM_BATCHES = {num_batches};\n")
        
        if batch_id_width > 0:
            f.write(f"    localparam OUT_BATCH_ID_WIDTH = {batch_id_width};\n")
            case_width = batch_id_width
        else:
            f.write(f"    localparam OUT_BATCH_ID_WIDTH = 1;\n")
            case_width = 1
        
        f.write("    \n")
        
        # Add solution buffers to store BST output
        f.write("    // Store BST output\n")
        for i in range(self.config.n_solutions):
            f.write(f"    reg [`PDAQP_SOL_{i}_WIDTH-1:0] bst_solution_buf_{i};\n")
        f.write("    \n")
        
        f.write("    reg [OUT_BATCH_ID_WIDTH-1:0] out_batch_id;\n")
        f.write("    reg out_batch_valid;\n")
        f.write("    \n")
        
        # State machine with buffering
        f.write("    // Capture BST output and manage output state\n")
        f.write("    always @(posedge clk or negedge rst_n) begin\n")
        f.write("        if (!rst_n) begin\n")
        f.write("            out_batch_id <= {OUT_BATCH_ID_WIDTH{1'b0}};\n")
        f.write("            out_batch_valid <= 1'b0;\n")
        for i in range(self.config.n_solutions):
            f.write(f"            bst_solution_buf_{i} <= `PDAQP_SOL_{i}_WIDTH'b0;\n")
        f.write("        end else begin\n")
        f.write("            // Capture new BST output when available and we're idle\n")
        f.write("            if (bst_valid_out && !out_batch_valid) begin\n")
        for i in range(self.config.n_solutions):
            f.write(f"                bst_solution_buf_{i} <= bst_solution_{i};\n")
        f.write("                out_batch_valid <= 1'b1;\n")
        f.write("                out_batch_id <= {OUT_BATCH_ID_WIDTH{1'b0}};\n")
        f.write("            end \n")
        f.write("            // Advance through output batches\n")
        f.write("            else if (output_fire) begin\n")
        
        if num_batches > 1:
            f.write("                if (out_batch_id == OUT_NUM_BATCHES - 1) begin\n")
            f.write("                    out_batch_valid <= 1'b0;\n")
            f.write("                    out_batch_id <= {OUT_BATCH_ID_WIDTH{1'b0}};\n")
            f.write("                end else begin\n")
            f.write("                    out_batch_id <= out_batch_id + 1'b1;\n")
            f.write("                end\n")
        else:
            f.write("                out_batch_valid <= 1'b0;\n")
            f.write("                out_batch_id <= {OUT_BATCH_ID_WIDTH{1'b0}};\n")
        
        f.write("            end\n")
        f.write("        end\n")
        f.write("    end\n")
        f.write("    \n")
        
        # Packing logic using buffered values
        f.write(f"    reg [{port_width-1}:0] batch_data;\n")
        f.write("    \n")
        f.write("    always @(*) begin\n")
        f.write("        case (out_batch_id)\n")
        
        sol_idx = 0
        for batch_id in range(num_batches):
            sols_this_batch = min(batch_size, self.config.n_solutions - sol_idx)
            
            bits_used = sols_this_batch * self.config.output_width
            padding_bits = port_width - bits_used
            
            # Fix: Use correct case width
            f.write(f"            {case_width}'d{batch_id}: batch_data = {{")
            
            if padding_bits > 0:
                f.write(f"{padding_bits}'d0, ")
            
            # Use buffered solutions instead of direct BST outputs
            for local_idx in range(sols_this_batch - 1, -1, -1):
                comma = ", " if local_idx > 0 else ""
                f.write(f"bst_solution_buf_{sol_idx + local_idx}{comma}")
            
            f.write("};\n")
            sol_idx += sols_this_batch
        
        f.write(f"            default: batch_data = {port_width}'d0;\n")
        f.write("        endcase\n")
        f.write("    end\n")
        f.write("    \n")
        f.write("    assign m_axis_tdata = batch_data;\n")
        f.write("    assign m_axis_tid = out_batch_id;\n")
        f.write("    assign m_axis_tvalid = out_batch_valid;\n")
        f.write("    \n")

    def _write_module_end(self, f):
        """Write module end"""
        f.write("\nendmodule\n")