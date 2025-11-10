#!/usr/bin/env python3
"""
Non-Pipeline Hardware Design Generator for PDAQP
Generates sequential state machine based hardware design files
Uses same config.vh definitions as pipeline version but generates sequential architecture
All comments are in English for international compatibility
"""

import re
import os
import sys
import argparse
from pathlib import Path

class VerilogConfigParser:
    """Parser for Verilog configuration files (same as pipeline version)"""
    
    def __init__(self, config_file):
        self.config_file = config_file
        self.defines = {}
        # Extract project name from config file path for auto-naming
        self.project_name = self.extract_project_name_from_path(config_file)
        self.parse_config()
    
    def extract_project_name_from_path(self, config_path):
        """Extract project name from config file path automatically"""
        config_path = Path(config_path)
        
        # Method 1: Extract from filename (e.g., pdaqp_siso_config.vh -> pdaqp_siso)
        filename = config_path.stem
        if filename.endswith('_config'):
            return filename[:-7]  # Remove '_config' suffix
        
        # Method 2: Extract from parent directory (e.g., pda_siso/include/config.vh -> pda_siso)
        parent_dir = config_path.parent.parent.name
        if parent_dir.startswith('pda_'):
            return parent_dir.replace('pda_', 'pdaqp_')
        
        # Method 3: Default fallback
        return 'pdaqp_generated'
    
    def parse_config(self):
        """Parse config.vh file and extract all defines"""
        try:
            with open(self.config_file, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: Config file {self.config_file} not found")
            sys.exit(1)
        
        # Extract all `define statements
        define_pattern = r'`define\s+(\w+)\s+([^\s]+)(?:\s*//.*)?'
        matches = re.findall(define_pattern, content)
        
        for name, value in matches:
            # Try to convert to integer if possible
            try:
                if value.startswith('0x'):
                    self.defines[name] = int(value, 16)
                else:
                    self.defines[name] = int(value)
            except ValueError:
                # Keep as string if not a number
                self.defines[name] = value
        
        print(f"Parsed {len(self.defines)} configuration defines from {self.config_file}")
        print(f"Auto-detected project name: {self.project_name}")
        
        # Validate required defines exist
        self.validate_required_defines()
    
    def validate_required_defines(self):
        """Validate that all required defines exist, add missing ones with defaults"""
        required_defines = {
            'OUTPUT_FRAC_BITS': 12,  # Default: 16-bit output - 4 int bits = 12 frac bits
            'OUTPUT_INT_BITS': 4,    # Default: 4 integer bits
            'OUTPUT_DATA_WIDTH': 16, # Default: 16-bit output width
            'HALFPLANE_FRAC_BITS': 14, # Default: Q2.14 format
            'HALFPLANE_INT_BITS': 2,   # Default: 2 integer bits
            'FEEDBACK_FRAC_BITS': 12,  # Default: Q4.12 format
            'FEEDBACK_INT_BITS': 4,    # Default: 4 integer bits
            'HALFPLANE_SCALE_FACTOR': 16384  # 2^14 for Q2.14 to Q4.28 scaling
        }
        
        missing_defines = []
        for define_name, default_value in required_defines.items():
            if define_name not in self.defines:
                self.defines[define_name] = default_value
                missing_defines.append(f"{define_name}={default_value}")
        
        if missing_defines:
            print(f"Warning: Added missing defines with defaults: {', '.join(missing_defines)}")
    
    def get_define(self, name, default=None):
        """Get a define value with optional default"""
        return self.defines.get(name, default)
    
    def get_int_define(self, name, default=0):
        """Get an integer define value"""
        value = self.defines.get(name, default)
        return int(value) if isinstance(value, str) and value.isdigit() else value

class NonPipelineHardwareGenerator:
    """Generator for non-pipeline (sequential) hardware design files"""
    
    def __init__(self, config_parser, output_dir=None):
        self.config = config_parser
        
        # Use specified output directory (should be existing folder)
        if output_dir is None:
            raise ValueError("Output directory must be specified")
        
        self.base_dir = Path(output_dir)
        self.rtl_dir = self.base_dir / "rtl"
        self.tb_dir = self.base_dir / "tb"
        self.include_dir = self.base_dir / "include"
        
        # Ensure directories exist (don't create new structure, use existing)
        if not self.base_dir.exists():
            raise ValueError(f"Output directory {self.base_dir} does not exist")
        
        # Create rtl and tb subdirectories if they don't exist
        for dir_path in [self.rtl_dir, self.tb_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Extract key parameters
        self.n_parameters = self.config.get_int_define('PDAQP_N_PARAMETER', 2)
        self.n_solutions = self.config.get_int_define('PDAQP_N_SOLUTION', 2)
        self.tree_nodes = self.config.get_int_define('PDAQP_TREE_NODES', 27)
        self.halfplanes_count = self.config.get_int_define('PDAQP_HALFPLANES', 36)
        self.feedbacks_count = self.config.get_int_define('PDAQP_FEEDBACKS', 54)
        self.bst_depth = self.config.get_int_define('PDAQP_ESTIMATED_BST_DEPTH', 7)
        self.data_width = self.config.get_int_define('INPUT_DATA_WIDTH', 16)
        
        print(f"Non-Pipeline Hardware parameters: {self.n_parameters} params, {self.n_solutions} solutions, BST depth={self.bst_depth}")
        print(f"Generated files will be saved to: {self.base_dir}")

    def generate_bst_lut_module(self):
        """Generate the main BST LUT module with sequential state machine"""
        project_name = self.config.project_name
        
        # Generate parameter input ports
        param_inputs = []
        for i in range(self.n_parameters):
            param_inputs.append(f"    input  signed [15:0]                param_in_{i},")
        
        # Generate solution output ports
        sol_outputs = []
        for i in range(self.n_solutions):
            sol_outputs.append(f"    output reg signed [15:0]            sol_out_{i},")
        
        # Generate parameter storage registers
        param_storage = []
        for i in range(self.n_parameters):
            param_storage.append(f"    reg signed [15:0] param_{i};")
        
        # Generate calculation registers with expanded width
        calc_registers = []
        for i in range(self.n_solutions):
            calc_registers.append(f"    reg signed [32:0] sol_val_{i};    // 33-bit for calculations")
        
        # Generate parameter capture
        param_capture = []
        for i in range(self.n_parameters):
            param_capture.append(f"            param_{i} <= param_in_{i};")
        
        # Generate halfplane calculation terms
        hp_calc_terms = []
        for i in range(self.n_parameters):
            hp_calc_terms.append(f"($signed(param_{i}) * $signed(halfplanes[hp_list[node_id]*`PDAQP_HALFPLANE_STRIDE+{i}]))")
        
        # Generate solution calculations
        sol_calculations = []
        for i in range(self.n_solutions):
            terms = []
            for j in range(self.n_parameters):
                terms.append(f"($signed(param_{j}) * $signed(feedbacks[hp_idx*`PDAQP_SOL_PER_NODE+{i*(self.n_parameters+1)+j}]))")
            # Add offset term with proper scaling
            terms.append(f"($signed(feedbacks[hp_idx*`PDAQP_SOL_PER_NODE+{i*(self.n_parameters+1)+self.n_parameters}]) << `HALFPLANE_FRAC_BITS)")
            sol_calculations.append(f"                    sol_val_{i} <= {' + '.join(terms)};")
        
        # Generate output assignments
        output_assignments = []
        for i in range(self.n_solutions):
            output_assignments.append(f"                    sol_out_{i} <= sol_val_{i}[(`OUTPUT_INT_BITS + `OUTPUT_FRAC_BITS + `HALFPLANE_FRAC_BITS - 1):(`HALFPLANE_FRAC_BITS)];")
        
        # Generate reset for outputs
        output_reset = []
        for i in range(self.n_solutions):
            output_reset.append(f"            sol_out_{i} <= 0;")

        template = f"""`timescale 1ns/1ps
// Auto-generated Non-Pipeline BST LUT module for {project_name}
// Sequential state machine based implementation
// Generated from configuration parameters

`include "include/{project_name}_config.vh"

module {project_name}_bst_lut (
    input                               clk,
    input                               rst_n,
{chr(10).join(param_inputs)}
    input                               valid_in,
{chr(10).join(sol_outputs)}
    output reg                          valid_out
);
    // ROM storage
    (* rom_style = "distributed" *) reg [15:0] halfplanes [0:`PDAQP_HALFPLANES-1];
    (* rom_style = "distributed" *) reg [15:0] feedbacks [0:`PDAQP_FEEDBACKS-1];
    (* rom_style = "distributed" *) reg [7:0] hp_list [0:`PDAQP_TREE_NODES-1];
    (* rom_style = "distributed" *) reg [7:0] jump_list [0:`PDAQP_TREE_NODES-1];

    initial begin
        $readmemh("include/{project_name}_halfplanes.mem", halfplanes);
        $readmemh("include/{project_name}_feedbacks.mem", feedbacks);
        $readmemh("include/{project_name}_hp_list.mem", hp_list);
        $readmemh("include/{project_name}_jump_list.mem", jump_list);
    end
    
    // State machine definitions
    reg [2:0] state;
    localparam IDLE = 3'd0;
    localparam BST_TRAVERSE = 3'd1;
    localparam CALC_SOLUTION = 3'd2;
    localparam OUTPUT_RESULT = 3'd3;
    
    // BST traversal registers
    reg [7:0] node_id;
    reg [7:0] next_id;
    reg [7:0] hp_idx;
    
    // Parameter storage
{chr(10).join(param_storage)}
    
    // Calculation registers - INCREASED WIDTH to prevent overflow
    reg signed [32:0] hp_val;       // 33-bit for potential overflow
{chr(10).join(calc_registers)}
    
    // Processing control
    reg busy;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            valid_out <= 0;
{chr(10).join(output_reset)}
            busy <= 0;
            node_id <= 0;
            next_id <= 0;
            hp_idx <= 0;
        end else begin
            // Default control signals
            valid_out <= 0;
            
            // Input capture
            if (valid_in && !busy) begin
{chr(10).join(param_capture)}
                busy <= 1;
                
                // Initialize BST traversal
                node_id <= 0;
                next_id <= jump_list[0];  // Use jump_list[0] for initial next_id
                state <= BST_TRAVERSE;
                
                $display("[BST-SEQUENTIAL] Processing params={' '.join([f'%h' for _ in range(self.n_parameters)])}", {', '.join([f'param_in_{i}' for i in range(self.n_parameters)])});
            end
            
            case (state)
                IDLE: begin
                    // Wait for input
                end
                
                BST_TRAVERSE: begin
                    // Leaf node detection
                    if (next_id == 0) begin
                        // We've reached a leaf node
                        hp_idx <= hp_list[node_id];
                        state <= CALC_SOLUTION;
                        
                        $display("[BST-SEQUENTIAL] Reached leaf node %d, hp_idx=%d", 
                               node_id, hp_list[node_id]);
                    end else begin
                        // Calculate halfplane value and branch - EXPANDED BIT WIDTH
                        hp_val <= {' + '.join(hp_calc_terms)};
                                 
                        if ({' + '.join(hp_calc_terms)} <= 
                            ($signed(halfplanes[hp_list[node_id]*`PDAQP_HALFPLANE_STRIDE+`PDAQP_N_PARAMETER]) * `HALFPLANE_SCALE_FACTOR)) begin
                            // Left branch
                            node_id <= node_id + next_id;
                            next_id <= jump_list[node_id + next_id];
                            
                            $display("[BST-SEQUENTIAL] LEFT branch to node %d", node_id + next_id);
                        end else begin
                            // Right branch
                            node_id <= node_id + next_id + 1;
                            next_id <= jump_list[node_id + next_id + 1];
                            
                            $display("[BST-SEQUENTIAL] RIGHT branch to node %d", node_id + next_id + 1);
                        end
                    end
                end
                
                CALC_SOLUTION: begin
                    // Calculate all solutions with EXPANDED BIT WIDTH using vh file definitions
{chr(10).join(sol_calculations)}
                    
                    $display("[BST-SEQUENTIAL] Calculating solutions for hp_idx=%d", hp_idx);
                    state <= OUTPUT_RESULT;
                end
                
                OUTPUT_RESULT: begin
                    // Output the properly scaled solutions using vh file bit definitions
{chr(10).join(output_assignments)}
                    
                    // Assert valid output
                    valid_out <= 1'b1;
                    
                    $display("[BST-SEQUENTIAL] Output: {' '.join([f'%h' for _ in range(self.n_solutions)])}", 
                           {', '.join([f'sol_val_{i}[(`OUTPUT_INT_BITS + `OUTPUT_FRAC_BITS + `HALFPLANE_FRAC_BITS - 1):(`HALFPLANE_FRAC_BITS)]' for i in range(self.n_solutions)])});
                    
                    // Reset busy flag and return to idle
                    busy <= 0;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end
endmodule
"""
        
        output_file = self.rtl_dir / f"{project_name}_bst_lut.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated Non-Pipeline BST LUT module: {output_file}")
        return output_file

    def generate_top_module(self):
        """Generate the top-level module with AXI4-Stream interface (similar to pipeline version)"""
        project_name = self.config.project_name
        
        # Calculate output data width (n_solutions * 16 bits)
        output_width = self.n_solutions * 16
        
        # Generate parameter extraction
        param_extractions = []
        for i in range(self.n_parameters):
            start_bit = i * 16
            end_bit = start_bit + 15
            param_extractions.append(f"    wire [15:0] param_{i} = s_axis_tdata[{end_bit}:{start_bit}];")
        
        # Generate solution wires
        sol_wires = []
        for i in range(self.n_solutions):
            sol_wires.append(f"    wire [15:0] sol_{i};")
        
        # Generate output concatenation
        sol_concat = ', '.join([f'sol_{i}' for i in reversed(range(self.n_solutions))])
        
        # Generate BST instance connections
        bst_connections = []
        for i in range(self.n_parameters):
            bst_connections.append(f"        .param_in_{i}(param_{i}),")
        for i in range(self.n_solutions):
            bst_connections.append(f"        .sol_out_{i}(sol_{i}),")

        template = f"""`timescale 1ns/1ps
// Auto-generated Non-Pipeline top module for {project_name}
// AXI4-Stream interface wrapper for sequential implementation

`include "include/{project_name}_config.vh"

module {project_name}_top (
    input wire clk,
    input wire rst_n,
    
    // AXI4-Stream slave interface (input)
    input wire [31:0] s_axis_tdata,
    input wire s_axis_tvalid,
    output wire s_axis_tready,
    
    // AXI4-Stream master interface (output)
    output wire [{output_width-1}:0] m_axis_tdata,
    output wire m_axis_tvalid,
    input wire m_axis_tready
);
    // Extract parameters from input data
{chr(10).join(param_extractions)}
    
    // BST outputs
{chr(10).join(sol_wires)}
    wire bst_valid_out;
    
    // Input change detection with improved handling for sequential processing
    reg [31:0] prev_tdata;
    reg new_input_detected;
    reg valid_pending;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prev_tdata <= 0;
            new_input_detected <= 0;
            valid_pending <= 0;
        end else begin
            // Default
            new_input_detected <= 0;
            
            // Process new input when s_axis_tvalid is asserted
            if (s_axis_tvalid) begin
                // Only process when data changes or we have a pending valid
                if (s_axis_tdata != prev_tdata || valid_pending) begin
                    new_input_detected <= 1;
                    prev_tdata <= s_axis_tdata;
                    valid_pending <= 0;
                    $display("[{project_name.upper()}_TOP] New vector detected: tdata=%h, params={' '.join([f'%h' for _ in range(self.n_parameters)])}", 
                           s_axis_tdata, {', '.join([f'param_{i}' for i in range(self.n_parameters)])});
                end
            end else if (!valid_pending) begin
                // Mark valid as pending when s_axis_tvalid deasserts
                valid_pending <= 1;
            end
            
            // Output logging
            if (bst_valid_out) begin
                $display("[{project_name.upper()}_TOP] Output generated: tdata=%h", 
                       {{{sol_concat}}});
            end
        end
    end
    
    // AXI handshake signals
    assign s_axis_tready = 1'b1;  // Always ready to receive
    assign m_axis_tvalid = bst_valid_out;
    assign m_axis_tdata = {{{sol_concat}}};  // {output_width}-bit output ({self.n_solutions}x 16-bit)
    
    // BST instantiation
    {project_name}_bst_lut bst_inst (
        .clk(clk),
        .rst_n(rst_n),
{chr(10).join(bst_connections)}
        .valid_in(new_input_detected),
        .valid_out(bst_valid_out)
    );
endmodule
"""
        
        output_file = self.rtl_dir / f"{project_name}_top.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated Non-Pipeline top module: {output_file}")
        return output_file

    def generate_testbench(self):
        """Generate testbench for non-pipeline architecture (no II=1 testing)"""
        project_name = self.config.project_name
        output_width = self.n_solutions * 16
        
        # Generate parameter test data
        param_test_data = []
        for i in range(self.n_parameters):
            param_test_data.append(f"    reg [15:0] test_params_{i} [0:TEST_VECTORS-1];")
        
        # Generate groundtruth parameter inputs
        gt_param_inputs = ', '.join([f'param_{i}' for i in range(self.n_parameters)])
        
        # Generate groundtruth function parameters
        gt_function_params = []
        for i in range(self.n_parameters):
            gt_function_params.append(f"        input [15:0] param_{i};")
        
        # Generate BST traversal for groundtruth (use vh file definitions for universality)
        gt_hp_calc_terms = []
        for i in range(self.n_parameters):
            gt_hp_calc_terms.append(f"$signed(param_{i}) * $signed(halfplanes[hp_idx*`PDAQP_HALFPLANE_STRIDE+{i}])")
        
        # Generate solution calculations for groundtruth (use vh file definitions for universality)
        gt_sol_calcs = []
        for i in range(self.n_solutions):
            terms = []
            for j in range(self.n_parameters):
                terms.append(f"($signed(param_{j}) * $signed(feedbacks[feedback_base+{i}*(`PDAQP_N_PARAMETER+1)+{j}]))")
            # Use vh file definition for feedback scaling - universal approach
            terms.append(f"($signed(feedbacks[feedback_base+{i}*(`PDAQP_N_PARAMETER+1)+`PDAQP_N_PARAMETER]) << `HALFPLANE_FRAC_BITS)")
            gt_sol_calcs.append(f"            sol_val_{i} = {' + '.join(terms)};")
        
        # Generate output concatenation for groundtruth (use vh file definitions for universality)
        # Extract bits based on output format requirements - use OUTPUT_INT_BITS and OUTPUT_FRAC_BITS
        gt_output_concat = ' | '.join([f'(sol_val_{i}[(`OUTPUT_INT_BITS + `OUTPUT_FRAC_BITS + `HALFPLANE_FRAC_BITS - 1):(`HALFPLANE_FRAC_BITS)] << ({i}*`OUTPUT_DATA_WIDTH))' for i in range(self.n_solutions)])

        # Calculate expected processing delay for non-pipeline (worst case BST depth + 2 cycles for calc and output)
        expected_delay = self.bst_depth + 2

        template = f"""`timescale 1ns/1ps
`include "include/{project_name}_config.vh"

module {project_name}_tb;

    // Constants definition
    localparam CLK_PERIOD = 10;  // 10ns (100MHz)
    localparam SIM_CYCLES = 10000; // Maximum simulation cycles
    localparam TEST_VECTORS = 50; // Number of test vectors (reduced for sequential)
    localparam MAX_PROCESSING_DELAY = {expected_delay}; // Expected processing delay for sequential architecture
    localparam DEBUG_INTERVAL = 100; // Output debug info every 100 cycles
    
    // DUT interface signals
    reg clk;
    reg rst_n;
    reg [31:0] s_axis_tdata;
    reg s_axis_tvalid;
    wire s_axis_tready;
    wire [{output_width-1}:0] m_axis_tdata;  // {output_width} bits - {self.n_solutions} 16-bit outputs
    wire m_axis_tvalid;
    reg m_axis_tready;
    
    // Test signals
{chr(10).join(param_test_data)}
    reg [{output_width-1}:0] expected_results [0:TEST_VECTORS-1]; // {output_width}-bit output
    reg [{output_width-1}:0] actual_results [0:TEST_VECTORS-1];   // {output_width}-bit output
    integer output_count;
    integer test_idx;
    integer errors;
    integer i;
    integer f;
    integer debug_cycles;
    
    // Sequential processing timing verification
    reg [31:0] cycle_counter;          // Global cycle counter
    reg [31:0] input_timestamps [0:TEST_VECTORS-1];   // When inputs were sent
    reg [31:0] output_timestamps [0:TEST_VECTORS-1];  // When outputs were received
    reg [31:0] processing_delays [0:TEST_VECTORS-1];  // Processing delay per vector
    reg [31:0] max_delay;              // Maximum processing delay detected
    reg [31:0] min_delay;              // Minimum processing delay detected
    reg [31:0] total_delay;            // Total processing delay
    
    // ROM storage arrays for groundtruth generation
    reg [15:0] halfplanes [0:`PDAQP_HALFPLANES-1];
    reg [15:0] feedbacks [0:`PDAQP_FEEDBACKS-1];
    reg [7:0] hp_list [0:`PDAQP_TREE_NODES-1];
    reg [7:0] jump_list [0:`PDAQP_TREE_NODES-1];
    
    // DUT instantiation - interface for {project_name} version
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
    
    // Read ROM data files
    initial begin
        $readmemh("include/{project_name}_halfplanes.mem", halfplanes);
        $readmemh("include/{project_name}_feedbacks.mem", feedbacks);
        $readmemh("include/{project_name}_hp_list.mem", hp_list);
        $readmemh("include/{project_name}_jump_list.mem", jump_list);
    end

    // Verify memory loading
    initial begin
        #10;
        $display("Memory verification:");
        $display("  Halfplanes[0:2] = %h %h %h", halfplanes[0], halfplanes[1], halfplanes[2]);
        $display("  Feedbacks[0:2] = %h %h %h", feedbacks[0], feedbacks[1], feedbacks[2]);
        $display("  HP_list[0:2] = %h %h %h", hp_list[0], hp_list[1], hp_list[2]);
        $display("  Jump_list[0:2] = %h %h %h", jump_list[0], jump_list[1], jump_list[2]);
    end

    // Groundtruth calculation function (same as pipeline version)
    function [{output_width-1}:0] calculate_groundtruth;
{chr(10).join(gt_function_params)}
        
        reg [7:0] node_id, next_id, hp_idx;
        reg signed [31:0] hp_val;
        reg signed [31:0] hp_thresh;  // Changed to 32-bit to match manual version
        reg [15:0] feedback_base;
        reg signed [31:0] sol_val_0{'' if self.n_solutions == 1 else ', sol_val_1'}{'' if self.n_solutions <= 2 else ', ' + ', '.join([f'sol_val_{i}' for i in range(2, self.n_solutions)])};
        
        begin
            // Initialize tree traversal
            node_id = 0;
            next_id = jump_list[0];  // Fixed: use jump_list[0] instead of 1
            
            $display("[TB] Groundtruth calculation for params: {' '.join([f'%h' for _ in range(self.n_parameters)])}", {gt_param_inputs});
            
            // BST traversal - use manual version's approach
            while (next_id != 0) begin
                // Get halfplane index
                hp_idx = hp_list[node_id];
                
                $display("[TB] Node: id=%d, hp_idx=%d", node_id, hp_idx);
                
                // Calculate halfplane value - use vh file stride for universality
                hp_val = {' + '.join(gt_hp_calc_terms)};
                // Use correct threshold scaling like manual version: hp_thresh * HALFPLANE_SCALE_FACTOR
                hp_thresh = $signed(halfplanes[hp_idx*`PDAQP_HALFPLANE_STRIDE + `PDAQP_N_PARAMETER]) * `HALFPLANE_SCALE_FACTOR;
                
                $display("[TB] Halfplane calc: %h <= %h", hp_val, hp_thresh);
                
                // Navigate tree - direct comparison like manual version
                if (hp_val <= hp_thresh) begin
                    // Take left branch
                    $display("[TB] LEFT branch: %h <= %h, new id=%d", 
                            hp_val, hp_thresh, node_id + next_id);
                    node_id = node_id + next_id;
                end else begin
                    // Take right branch
                    $display("[TB] RIGHT branch: %h > %h, new id=%d", 
                            hp_val, hp_thresh, node_id + next_id + 1);
                    node_id = node_id + next_id + 1;
                end
                
                // Update next_id
                next_id = jump_list[node_id];
                $display("[TB] Update: new id=%d, next_id=%d", node_id, next_id);
            end
            
            // We've reached a leaf node - get the feedback index
            hp_idx = hp_list[node_id];
            feedback_base = hp_idx * `PDAQP_SOL_PER_NODE;  // Use vh file definition for universality
            
            $display("[TB] Reached leaf node: id=%d, hp_idx=%d, fb_base=%d", 
                    node_id, hp_idx, feedback_base);
            
            // Calculate solutions - use vh file definitions for universal scaling
{chr(10).join(gt_sol_calcs)}
            
            // Return concatenated result using vh file bit definitions
            calculate_groundtruth = {gt_output_concat};
            
            $display("[TB] Groundtruth result: %h", calculate_groundtruth);
        end
    endfunction

    // Generate random test data
    initial begin
        for (i = 0; i < TEST_VECTORS; i = i + 1) begin
{chr(10).join([f"            test_params_{j}[i] = $random & 16'hFFFF;" for j in range(self.n_parameters)])}
        end
    end

    // Global cycle counter
    always @(posedge clk) begin
        if (!rst_n)
            cycle_counter <= 0;
        else
            cycle_counter <= cycle_counter + 1;
    end

    // Main test process
    initial begin
        // Initialize signals
        rst_n = 0;
        s_axis_tvalid = 0;
        s_axis_tdata = 0;
        m_axis_tready = 1;
        output_count = 0;
        errors = 0;
        debug_cycles = 0;
        max_delay = 0;
        min_delay = 32'hFFFFFFFF;
        total_delay = 0;
        
        // Calculate expected results
        $display("[%0t ns] Calculating expected results...", $time);
        for (i = 0; i < TEST_VECTORS; i = i + 1) begin
            expected_results[i] = calculate_groundtruth({', '.join([f'test_params_{j}[i]' for j in range(self.n_parameters)])});
        end
        
        $display("\\n=================================================================");
        $display("              {project_name.upper()} NON-PIPELINE TESTBENCH STARTED               ");
        $display("=================================================================");
        $display("Configuration:");
        $display("  Test vectors: %0d", TEST_VECTORS);
        $display("  Clock period: %0d ns", CLK_PERIOD);
        $display("  Simulation cycles: %0d", SIM_CYCLES);
        $display("  Solutions: %0d", `PDAQP_N_SOLUTION);
        $display("  Parameters: %0d", `PDAQP_N_PARAMETER);
        $display("  Architecture: Sequential State Machine");
        $display("  Expected max delay: %0d cycles", MAX_PROCESSING_DELAY);
        $display("  Halfplanes format: Q%0d.%0d", `HALFPLANE_INT_BITS, `HALFPLANE_FRAC_BITS);
        $display("  Feedbacks format: Q%0d.%0d", `FEEDBACK_INT_BITS, `FEEDBACK_FRAC_BITS);
        $display("=================================================================");
        
        // Apply reset
        $display("[%0t ns] Applying reset...", $time);
        repeat(10) @(posedge clk);
        rst_n = 1;
        $display("[%0t ns] Reset released", $time);
        repeat(10) @(posedge clk);
        
        // Send test vectors one by one with sufficient spacing for sequential processing
        for (test_idx = 0; test_idx < TEST_VECTORS; test_idx = test_idx + 1) begin
            // Report progress at 10% increments
            if (test_idx % (TEST_VECTORS/10) == 0 || test_idx == TEST_VECTORS-1)
                $display("[%0t ns] Applying vector %0d/%0d (%0d%%)", 
                        $time, test_idx+1, TEST_VECTORS, 
                        ((test_idx+1)*100)/TEST_VECTORS);
            
            // Wait for ready
            @(posedge clk);
            
            // Send data and record timestamp
            input_timestamps[test_idx] = cycle_counter;
            s_axis_tdata = {{{', '.join([f'test_params_{j}[test_idx]' for j in reversed(range(self.n_parameters))])}}};
            s_axis_tvalid = 1;
            $display("[%0t ns] Sending test vector %0d, data=%h, params={' '.join([f'%h' for _ in range(self.n_parameters)])}", 
                    $time, test_idx, s_axis_tdata, {', '.join([f'test_params_{j}[test_idx]' for j in range(self.n_parameters)])});
            
            @(posedge clk);
            s_axis_tvalid = 0;
            
            // Wait sufficient time for sequential processing
            repeat(MAX_PROCESSING_DELAY * 2) @(posedge clk);
        end
        
        // Wait for all outputs to complete
        $display("[%0t ns] Waiting for all outputs to complete...", $time);
        repeat(MAX_PROCESSING_DELAY * 3) @(posedge clk);
        
        // Calculate statistics
        if (output_count > 0) begin
            for (i = 0; i < output_count; i = i + 1) begin
                if (processing_delays[i] > max_delay) max_delay = processing_delays[i];
                if (processing_delays[i] < min_delay) min_delay = processing_delays[i];
                total_delay = total_delay + processing_delays[i];
            end
        end
        
        // Print test results with better formatting
        $display("\\n=================================================================");
        $display("                NON-PIPELINE TEST RESULTS SUMMARY                ");
        $display("=================================================================");
        $display("  Total vectors:       %0d", TEST_VECTORS);
        $display("  Vectors processed:   %0d", output_count);
        $display("  Errors detected:     %0d", errors);
        if (output_count > 0) begin
            $display("  Error rate:          %.2f%%", (errors*100.0)/output_count);
            $display("  Average delay:       %.1f cycles", (total_delay*1.0)/output_count);
            $display("  Min delay:           %0d cycles", min_delay);
            $display("  Max delay:           %0d cycles", max_delay);
        end else begin
            $display("  Error rate:          N/A (no outputs received)");
        end
        $display("  Expected max delay:  %0d cycles", MAX_PROCESSING_DELAY);
        $display("  Simulation time:     %0t ns", $time);
        $display("=================================================================");
        
        if (errors == 0 && output_count >= TEST_VECTORS)
            $display("                         TEST PASSED!                           ");
        else if (errors == 0 && output_count < TEST_VECTORS)
            $display("            TEST INCOMPLETE - NO ERRORS IN OUTPUTS RECEIVED     ");
        else
            $display("                         TEST FAILED!                           ");
        $display("=================================================================");
        
        // Write results to file
        f = $fopen("{project_name}_sequential_test_results.csv", "w");
        $fwrite(f, "Index,{','.join([f'Param{i}' for i in range(self.n_parameters)])},Expected_Output,Actual_Output,Processing_Delay,Status\\n");
        for (i = 0; i < output_count && i < TEST_VECTORS; i = i + 1) begin
            $fwrite(f, "%0d,{','.join(['%h' for _ in range(self.n_parameters)])},{output_width}'h%h,{output_width}'h%h,%0d,%s\\n", 
                   i, {','.join([f'test_params_{j}[i]' for j in range(self.n_parameters)])}, expected_results[i], actual_results[i], 
                   processing_delays[i], (actual_results[i] === expected_results[i]) ? "PASS" : "FAIL");
        end
        $fclose(f);
        
        #(CLK_PERIOD*10)
        $finish;
    end
    
    // Capture output responses with timing analysis
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            $display("[%0t ns] **** DETECTED OUTPUT: m_axis_tdata = %h ****", $time, m_axis_tdata);
            
            // Record output and calculate processing delay
            actual_results[output_count] = m_axis_tdata;
            output_timestamps[output_count] = cycle_counter;
            
            if (output_count < TEST_VECTORS) begin
                processing_delays[output_count] = cycle_counter - input_timestamps[output_count];
                
                // Check against expected result
                if (m_axis_tdata !== expected_results[output_count]) begin
                    $display("[%0t ns] ERROR at vector %0d:", $time, output_count);
                    $display("  Params: {' '.join([f'%h' for _ in range(self.n_parameters)])}", {', '.join([f'test_params_{j}[output_count]' for j in range(self.n_parameters)])});
                    $display("  Expected: %h", expected_results[output_count]);
                    $display("  Got:      %h", m_axis_tdata);
                    $display("  Delay:    %0d cycles", processing_delays[output_count]);
                    errors = errors + 1;
                end else begin
                    $display("[%0t ns] Vector %0d PASSED (delay: %0d cycles)", $time, output_count, processing_delays[output_count]);
                end
            end
            
            output_count = output_count + 1;
        end
    end
    
    // Waveform dump
    initial begin
        $dumpfile("{project_name}_sequential_tb.vcd");
        $dumpvars(0, {project_name}_tb);
    end
    
endmodule
"""
        
        output_file = self.tb_dir / f"{project_name}_tb.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated Non-Pipeline testbench: {output_file}")
        return output_file

    def generate_makefile(self):
        """Generate Makefile for compilation and simulation"""
        project_name = self.config.project_name
        
        template = f"""# Auto-generated Makefile for {project_name} (Non-Pipeline)

# Simulation parameters
SIM = iverilog
VIEWER = gtkwave
VERILOG_SOURCES = rtl/{project_name}_bst_lut.v rtl/{project_name}_top.v tb/{project_name}_tb.v
VCD_FILE = {project_name}_sequential_tb.vcd
EXECUTABLE = {project_name}_sequential_sim

# Default target
all: compile run

# Compile the design
compile:
	@echo "Compiling {project_name} non-pipeline design..."
	$(SIM) -o $(EXECUTABLE) $(VERILOG_SOURCES)

# Run simulation
run: compile
	@echo "Running {project_name} non-pipeline simulation..."
	./$(EXECUTABLE)

# View waveforms
view: run
	@echo "Opening waveform viewer..."
	$(VIEWER) $(VCD_FILE) &

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f $(EXECUTABLE) $(VCD_FILE) {project_name}_sequential_test_results.csv

# Help
help:
	@echo "Available targets:"
	@echo "  all     - Compile and run simulation (default)"
	@echo "  compile - Compile design only"
	@echo "  run     - Run simulation"
	@echo "  view    - Open waveform viewer"
	@echo "  clean   - Clean generated files"
	@echo "  help    - Show this help"

.PHONY: all compile run view clean help
"""
        
        output_file = self.base_dir / "Makefile"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated Non-Pipeline Makefile: {output_file}")
        return output_file

    def generate_all(self):
        """Generate all non-pipeline hardware design files"""
        print(f"\nGenerating complete non-pipeline hardware design for {self.config.project_name}...")
        
        # Generate RTL files
        bst_file = self.generate_bst_lut_module()
        top_file = self.generate_top_module()
        
        # Generate testbench
        tb_file = self.generate_testbench()
        
        # Generate build files
        make_file = self.generate_makefile()
        
        print(f"\n✅ Non-Pipeline Hardware generation complete!")
        print(f"📁 Output directory: {self.base_dir}")
        print(f"📋 Generated files:")
        print(f"   🔧 RTL: {bst_file.name}, {top_file.name}")
        print(f"   🧪 Testbench: {tb_file.name}")
        print(f"   🏗️  Build: {make_file.name}")
        print(f"\n🏛️  Architecture: Sequential State Machine")
        print(f"⏱️  Characteristics: Low Area, High Latency, No II=1 Support")
        print(f"\n🚀 To build and run:")
        print(f"   cd {self.base_dir}")
        print(f"   make run")
        
        return {
            'bst_lut': bst_file,
            'top': top_file,
            'testbench': tb_file,
            'makefile': make_file
        }

def main():
    parser = argparse.ArgumentParser(description='Generate non-pipeline hardware design for PDAQP')
    parser.add_argument('config_file', help='Path to configuration .vh file')
    parser.add_argument('-o', '--output', help='Output directory (auto-generated if not specified)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Parse configuration
    config_parser = VerilogConfigParser(args.config_file)
    
    # Generate hardware
    generator = NonPipelineHardwareGenerator(config_parser, args.output)
    generated_files = generator.generate_all()
    
    if args.verbose:
        print(f"\nDetailed file information:")
        for file_type, file_path in generated_files.items():
            print(f"  {file_type}: {file_path} ({file_path.stat().st_size} bytes)")

if __name__ == '__main__':
    main()