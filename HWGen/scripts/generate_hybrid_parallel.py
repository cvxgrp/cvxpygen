#!/usr/bin/env python3
"""
DSP-Limited Hardware Design Generator for PDAQP (Simplified - Hybrid Parallel Only)
Generates hardware design files with DSP resource constraints
Only supports Hybrid Parallel implementation when D >= n×p
Based on config.vh definitions, following manual version design patterns
"""

import re
import os
import sys
import argparse
from pathlib import Path

class VerilogConfigParser:
    """Parser for Verilog configuration files"""
    
    def __init__(self, config_file):
        self.config_file = config_file
        self.defines = {}
        # Extract project name from config file path for auto-naming
        self.project_name = self.extract_project_name_from_path(config_file)
        self.parse_config()
        
        # Validate required defines exist
        self.validate_required_defines()
    
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
        return 'pdaqp_generated_dsp'
    
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
    
    def validate_required_defines(self):
        """Validate that all required defines exist, add missing ones with defaults"""
        required_defines = {
            'OUTPUT_FRAC_BITS': 12,  # Default: 16-bit output - 4 int bits = 12 frac bits
            'OUTPUT_INT_BITS': 4,    # Default: 4 integer bits
            'OUTPUT_DATA_WIDTH': 16, # Default: 16-bit output width
            'HALFPLANE_FRAC_BITS': 14, # Default: Q2.14 format
            'HALFPLANE_INT_BITS': 2,   # Default: 2 integer bits
            'FEEDBACK_FRAC_BITS': 12,  # Default: Q4.12 format
            'FEEDBACK_INT_BITS': 4     # Default: 4 integer bits
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

class DSPLimitedHardwareGenerator:
    """Generator for DSP-constrained hardware design files - Simplified Hybrid Parallel Only"""
    
    def __init__(self, config_parser, output_dir=None):
        self.config = config_parser
        
        # DSP resource constraints
        self.max_dsp_blocks = 8      # Maximum 8 DSP blocks
        self.dsp_width = 16          # 16-bit DSP blocks
        self.current_dsp_usage = 0   # Track current DSP usage
        
        # Use existing DSP limited directory structure
        if output_dir is None:
            # Auto-detect existing DSP folder based on project name
            project_name = self.config.project_name
            if "siso" in project_name:
                output_dir = "pdaqp_siso_dsp_generated"
            else:
                output_dir = f"{project_name}_dsp_generated"
        
        self.base_dir = Path(output_dir)
        self.rtl_dir = self.base_dir / "rtl"
        self.tb_dir = self.base_dir / "tb"
        self.include_dir = self.base_dir / "include"
        
        # Create directories if they don't exist
        for dir_path in [self.base_dir, self.rtl_dir, self.tb_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Extract key parameters
        self.n_parameters = self.config.get_int_define('PDAQP_N_PARAMETER', 2)
        self.n_solutions = self.config.get_int_define('PDAQP_N_SOLUTION', 2)
        self.tree_nodes = self.config.get_int_define('PDAQP_TREE_NODES', 27)
        self.halfplanes_count = self.config.get_int_define('PDAQP_HALFPLANES', 36)
        self.feedbacks_count = self.config.get_int_define('PDAQP_FEEDBACKS', 54)
        self.bst_depth = self.config.get_int_define('PDAQP_ESTIMATED_BST_DEPTH', 7)
        self.data_width = self.config.get_int_define('INPUT_DATA_WIDTH', 16)
        
        # Calculate DSP requirements and validate
        self.validate_dsp_allocation()
        
        print(f"DSP-Constrained Hardware parameters: {self.n_parameters} params, {self.n_solutions} solutions, BST depth={self.bst_depth}")
        print(f"DSP resource allocation: {self.current_dsp_usage}/{self.max_dsp_blocks} blocks")
        print(f"Generated files will be saved to: {self.base_dir}")

    def validate_dsp_allocation(self):
        """
        Validate DSP allocation for Hybrid Parallel implementation only
        This simplified version only supports the case where D >= n×p
        """
        # Calculate DSP requirements
        hp_dsp_need = self.n_parameters  # Halfplane evaluation DSP requirement
        fb_dsp_need = self.n_parameters * self.n_solutions  # Feedback calculation requirement
        total_ideal_dsp = hp_dsp_need + fb_dsp_need
        
        print(f"DSP requirement analysis (Hybrid Parallel Only):")
        print(f"  Problem scale: n={self.n_parameters}, p={self.n_solutions}")
        print(f"  Halfplane evaluation: {hp_dsp_need} DSP blocks")
        print(f"  Feedback calculation: {fb_dsp_need} DSP blocks") 
        print(f"  Total ideal requirement: {total_ideal_dsp} DSP blocks")
        print(f"  Available DSP resources: {self.max_dsp_blocks} DSP blocks")
        
        # For simplified version, only support hybrid parallel implementation
        if fb_dsp_need > self.max_dsp_blocks:
            print(f"\n❌ ERROR: Insufficient DSP resources for Hybrid Parallel implementation")
            print(f"   Required: {fb_dsp_need} DSPs for feedback calculation")
            print(f"   Available: {self.max_dsp_blocks} DSPs")
            print(f"   This simplified version only supports Hybrid Parallel (D >= n×p)")
            print(f"   Please use the full version for time-division multiplexing support")
            sys.exit(1)
        
        # Hybrid allocation strategy: halfplane uses LUT, feedback uses DSP
        print(f"\nHybrid Resource Allocation Strategy:")
        print(f"  Halfplane evaluation: Traditional LUT-based multiplication (0 DSP)")
        print(f"  Feedback calculation: DSP-optimized implementation")
        
        self.hp_dsp_allocation = 0  # No DSP for halfplane
        self.fb_dsp_allocation = fb_dsp_need
        self.dsp_strategy = "hybrid_parallel"
        self.current_dsp_usage = fb_dsp_need
        
        print(f"  → Strategy: Hybrid Parallel (Traditional HP + DSP FB)")
        print(f"  → DSP allocation: {self.fb_dsp_allocation}/{self.max_dsp_blocks} blocks")
        print(f"  → Expected throughput: 100% (single-cycle solution computation)")
        print(f"  → Pipeline latency: BST_depth + 2 = {self.bst_depth + 2} cycles")

    def generate_dsp_limited_bst_lut(self):
        """Generate DSP-constrained BST LUT module (Hybrid Parallel only)"""
        project_name = self.config.project_name + "_dsp"
        
        # Generate parameter input ports
        param_inputs = []
        for i in range(self.n_parameters):
            param_inputs.append(f"    input  signed [15:0]                param_in_{i},")
        
        # Generate solution output ports
        sol_outputs = []
        for i in range(self.n_solutions):
            sol_outputs.append(f"    output reg signed [15:0]            sol_out_{i},")
        
        # Generate 16-bit DSP instantiations for all feedback calculations
        fb_dsp_instances = []
        for sol_idx in range(self.n_solutions):
            for param_idx in range(self.n_parameters):
                fb_dsp_instances.append(f"""
    // 16-bit DSP block for solution {sol_idx}, parameter {param_idx}
    reg signed [15:0] dsp_{sol_idx}_{param_idx}_a_reg, dsp_{sol_idx}_{param_idx}_b_reg;
    reg signed [31:0] dsp_{sol_idx}_{param_idx}_product_reg;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            dsp_{sol_idx}_{param_idx}_a_reg <= 16'd0;
            dsp_{sol_idx}_{param_idx}_b_reg <= 16'd0;
            dsp_{sol_idx}_{param_idx}_product_reg <= 32'd0;
        end else begin
            // 16-bit DSP: param * coefficient (inferred as DSP block)
            dsp_{sol_idx}_{param_idx}_a_reg <= param{param_idx}_pipe[MAX_BST_DEPTH-1];
            dsp_{sol_idx}_{param_idx}_b_reg <= feedback_coeff_{sol_idx}_{param_idx};
            dsp_{sol_idx}_{param_idx}_product_reg <= dsp_{sol_idx}_{param_idx}_a_reg * dsp_{sol_idx}_{param_idx}_b_reg;
        end
    end
    
    assign fb_mult_result_{sol_idx}_{param_idx} = dsp_{sol_idx}_{param_idx}_product_reg;""")

        # Generate pipeline registers for parameters
        param_pipes = []
        for i in range(self.n_parameters):
            param_pipes.append(f"    reg [15:0] param{i}_pipe[0:PIPE_DEPTH-1]; // Parameter {i} pipeline")
        
        # Generate solution pipeline registers
        sol_pipes = []
        for i in range(self.n_solutions):
            sol_pipes.append(f"    reg [31:0] sol{i}_pipe[0:1]; // Solution {i} accumulator pipeline")

        # Generate DSP result wire declarations for all feedback calculations
        dsp_wires = []
        for sol_idx in range(self.n_solutions):
            for param_idx in range(self.n_parameters):
                dsp_wires.append(f"    wire [31:0] fb_mult_result_{sol_idx}_{param_idx};")
                dsp_wires.append(f"    wire [15:0] feedback_coeff_{sol_idx}_{param_idx};")

        # Generate halfplane calculation logic (traditional multiplication)
        hp_calc_logic = f"""
                        // Traditional halfplane evaluation (DSPs reserved for feedback calculation)
                        // Q2.14 * Q2.14 = Q4.28, threshold needs to be scaled to Q4.28
                        hp_val = {' + '.join([f'($signed(param{i}_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+{i}]))' for i in range(self.n_parameters)])};"""

        # Generate feedback calculation logic using DSP blocks
        fb_calc_logic = []
        for sol_idx in range(self.n_solutions):
            terms = []
            for param_idx in range(self.n_parameters):
                terms.append(f"fb_mult_result_{sol_idx}_{param_idx}")
            
            fb_calc_logic.append(f"                sol{sol_idx}_pipe[0] <= {' + '.join(terms)};")
        
        # Generate continuous assignments for DSP coefficient connections
        fb_coeff_assignments = []
        for sol_idx in range(self.n_solutions):
            for param_idx in range(self.n_parameters):
                fb_coeff_assignments.append(f"    assign feedback_coeff_{sol_idx}_{param_idx} = feedbacks[final_hp_idx_pipe[0]*`PDAQP_SOL_PER_NODE+{sol_idx*(self.n_parameters+1)+param_idx}];")

        template = f"""`timescale 1ns/1ps
// Auto-generated DSP-constrained BST LUT module for {project_name}
// Simplified version: Hybrid Parallel implementation only (D >= n×p)
// DSP usage: {self.current_dsp_usage} x 16-bit DSP blocks

`include "include/{self.config.project_name}_config.vh"

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
        $readmemh("include/{self.config.project_name}_halfplanes.mem", halfplanes);
        $readmemh("include/{self.config.project_name}_feedbacks.mem", feedbacks);
        $readmemh("include/{self.config.project_name}_hp_list.mem", hp_list);
        $readmemh("include/{self.config.project_name}_jump_list.mem", jump_list);
    end
    
    // Pipeline stages and registers
    localparam MAX_BST_DEPTH = {self.bst_depth};
    localparam PIPE_DEPTH = MAX_BST_DEPTH + 2;
    
    // Pipeline registers - input parameters
    reg valid_pipe[0:PIPE_DEPTH-1];
{chr(10).join(param_pipes)}
    
    // BST traversal pipeline
    reg [7:0] node_id_pipe[0:MAX_BST_DEPTH-1];
    reg [7:0] next_id_pipe[0:MAX_BST_DEPTH-1];
    reg [7:0] hp_idx_pipe[0:MAX_BST_DEPTH-1];
    reg reached_leaf_pipe[0:MAX_BST_DEPTH-1];
    
    // Final calculation pipeline
    reg [7:0] final_hp_idx_pipe[0:1];
{chr(10).join(sol_pipes)}
    
    // DSP resource declarations and connections
{chr(10).join(dsp_wires)}
    
    // Temporary variables for halfplane calculation
    reg signed [31:0] hp_val;
    reg signed [31:0] hp_thresh;
    reg decision;
    
    integer i;

    // DSP instantiations for feedback calculation ({self.current_dsp_usage} DSP blocks)
{chr(10).join(fb_dsp_instances)}

    // Continuous assignments for DSP coefficient connections
{chr(10).join(fb_coeff_assignments)}
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all pipeline registers
            for (i = 0; i < PIPE_DEPTH; i = i + 1) begin
                valid_pipe[i] <= 0;
            end
            
            for (i = 0; i < MAX_BST_DEPTH; i = i + 1) begin
                node_id_pipe[i] <= 0;
                next_id_pipe[i] <= 0;
                hp_idx_pipe[i] <= 0;
                reached_leaf_pipe[i] <= 0;
            end
            
            for (i = 0; i < 2; i = i + 1) begin
                final_hp_idx_pipe[i] <= 0;
            end
            
            valid_out <= 0;
            
        end else begin
            //----------------
            // STAGE 0: Input
            //----------------
            valid_pipe[0] <= valid_in;
            if (valid_in) begin
{chr(10).join([f"                param{i}_pipe[0] <= param_in_{i};" for i in range(self.n_parameters)])}
                node_id_pipe[0] <= 0;
                next_id_pipe[0] <= 1;
                hp_idx_pipe[0] <= hp_list[0];
                reached_leaf_pipe[0] <= 0;
            end
            
            //----------------------------
            // STAGE 1-N: BST Traversal
            //----------------------------
            for (i = 0; i < MAX_BST_DEPTH-1; i = i + 1) begin
                valid_pipe[i+1] <= valid_pipe[i];
{chr(10).join([f"                param{j}_pipe[i+1] <= param{j}_pipe[i];" for j in range(self.n_parameters)])}
                
                if (valid_pipe[i]) begin
                    if (reached_leaf_pipe[i] || next_id_pipe[i] == 0) begin
                        // Already reached leaf - pass through
                        node_id_pipe[i+1] <= node_id_pipe[i];
                        next_id_pipe[i+1] <= next_id_pipe[i];
                        hp_idx_pipe[i+1] <= hp_idx_pipe[i];
                        reached_leaf_pipe[i+1] <= 1;
                    end else begin
                        // Traditional halfplane evaluation
{hp_calc_logic}
                        hp_thresh = $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+`PDAQP_N_PARAMETER]) * `HALFPLANE_SCALE_FACTOR;
                        decision = hp_val <= hp_thresh;
                        
                        if (decision) begin
                            // Left branch
                            node_id_pipe[i+1] <= node_id_pipe[i] + next_id_pipe[i];
                            next_id_pipe[i+1] <= jump_list[node_id_pipe[i] + next_id_pipe[i]];
                            hp_idx_pipe[i+1] <= hp_list[node_id_pipe[i] + next_id_pipe[i]];
                            reached_leaf_pipe[i+1] <= (jump_list[node_id_pipe[i] + next_id_pipe[i]] == 0);
                        end else begin
                            // Right branch
                            node_id_pipe[i+1] <= node_id_pipe[i] + next_id_pipe[i] + 1;
                            next_id_pipe[i+1] <= jump_list[node_id_pipe[i] + next_id_pipe[i] + 1];
                            hp_idx_pipe[i+1] <= hp_list[node_id_pipe[i] + next_id_pipe[i] + 1];
                            reached_leaf_pipe[i+1] <= (jump_list[node_id_pipe[i] + next_id_pipe[i] + 1] == 0);
                        end
                    end
                end
            end
            
            //---------------------------
            // STAGE N+1: DSP-based Feedback Calculation
            //---------------------------
            valid_pipe[MAX_BST_DEPTH] <= valid_pipe[MAX_BST_DEPTH-1];
{chr(10).join([f"            param{i}_pipe[MAX_BST_DEPTH] <= param{i}_pipe[MAX_BST_DEPTH-1];" for i in range(self.n_parameters)])}
            
            if (valid_pipe[MAX_BST_DEPTH-1]) begin
                final_hp_idx_pipe[0] <= hp_idx_pipe[MAX_BST_DEPTH-1];
                // DSP-based solution calculations (single-cycle)
{chr(10).join(fb_calc_logic)}
            end
            
            //---------------------------
            // STAGE N+2: Add Offsets and Output
            //---------------------------
            valid_pipe[MAX_BST_DEPTH+1] <= valid_pipe[MAX_BST_DEPTH];
            
            if (valid_pipe[MAX_BST_DEPTH]) begin
                final_hp_idx_pipe[1] <= final_hp_idx_pipe[0];
                // Add offset terms (scaled for Q6.26 format)
{chr(10).join([f"                sol{i}_pipe[1] <= sol{i}_pipe[0] + ($signed(feedbacks[final_hp_idx_pipe[0]*`PDAQP_SOL_PER_NODE+{i*(self.n_parameters+1)+self.n_parameters}]) << `HALFPLANE_FRAC_BITS);" for i in range(self.n_solutions)])}
            end
            
            //---------------------------
            // OUTPUT STAGE
            //---------------------------
            valid_out <= valid_pipe[MAX_BST_DEPTH+1];
            
            if (valid_pipe[MAX_BST_DEPTH+1]) begin
                // Extract output using vh file definitions
{chr(10).join([f"                sol_out_{i} <= sol{i}_pipe[1][(`OUTPUT_INT_BITS + `OUTPUT_FRAC_BITS + `HALFPLANE_FRAC_BITS - 1):(`HALFPLANE_FRAC_BITS)];" for i in range(self.n_solutions)])}
            end
        end
    end
endmodule
"""
        
        output_file = self.rtl_dir / f"{project_name}_bst_lut.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated DSP-constrained BST LUT module: {output_file}")
        print(f"DSP resource usage: {self.current_dsp_usage}/{self.max_dsp_blocks} blocks")
        return output_file

    def generate_top_module(self):
        """Generate the top-level module with AXI4-Stream interface"""
        # Same as before, no changes needed
        project_name = self.config.project_name + "_dsp"
        
        # Calculate output data width
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
// Auto-generated DSP-constrained top module for {project_name}
// AXI4-Stream interface wrapper - Hybrid Parallel implementation only

`include "include/{self.config.project_name}_config.vh"

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
    
    // Input valid pulse generation
    reg valid_pulse;
    reg [31:0] prev_tdata;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pulse <= 0;
            prev_tdata <= 32'hFFFFFFFF;
        end else begin
            valid_pulse <= 0;
            if (s_axis_tvalid && s_axis_tready && s_axis_tdata != prev_tdata) begin
                valid_pulse <= 1;
                prev_tdata <= s_axis_tdata;
            end
        end
    end
    
    // AXI handshake signals
    assign s_axis_tready = 1'b1;
    assign m_axis_tvalid = bst_valid_out;
    assign m_axis_tdata = {{{sol_concat}}};
    
    // DSP-constrained BST LUT instance
    {project_name}_bst_lut bst_inst (
        .clk(clk),
        .rst_n(rst_n),
{chr(10).join(bst_connections)}
        .valid_in(valid_pulse),
        .valid_out(bst_valid_out)
    );
endmodule
"""
        
        output_file = self.rtl_dir / f"{project_name}_top.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated DSP-constrained top module: {output_file}")
        return output_file

    def generate_testbench(self):
        """Generate testbench for DSP-constrained version"""
        # Same testbench as before, just update comments
        project_name = self.config.project_name + "_dsp"
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
        
        template = f"""`timescale 1ns/1ps
`include "include/{self.config.project_name}_config.vh"

module {project_name}_tb;

    // Constants definition
    localparam CLK_PERIOD = 10;
    localparam SIM_CYCLES = 5000;
    localparam TEST_VECTORS = 100;  // Full testing for hybrid parallel
    localparam PIPELINE_DELAY = {self.bst_depth + 2};
    
    // DUT interface signals
    reg clk;
    reg rst_n;
    reg [31:0] s_axis_tdata;
    reg s_axis_tvalid;
    wire s_axis_tready;
    wire [{output_width-1}:0] m_axis_tdata;
    wire m_axis_tvalid;
    reg m_axis_tready;
    
    // Test signals
{chr(10).join(param_test_data)}
    reg [{output_width-1}:0] expected_results [0:TEST_VECTORS-1];
    reg [{output_width-1}:0] actual_results [0:TEST_VECTORS-1];
    integer output_count;
    integer test_idx;
    integer errors;
    integer i;
    
    // ROM storage arrays for groundtruth
    reg [15:0] halfplanes [0:`PDAQP_HALFPLANES-1];
    reg [15:0] feedbacks [0:`PDAQP_FEEDBACKS-1];
    reg [7:0] hp_list [0:`PDAQP_TREE_NODES-1];
    reg [7:0] jump_list [0:`PDAQP_TREE_NODES-1];
    
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
    
    // Read ROM data files
    initial begin
        $readmemh("include/{self.config.project_name}_halfplanes.mem", halfplanes);
        $readmemh("include/{self.config.project_name}_feedbacks.mem", feedbacks);
        $readmemh("include/{self.config.project_name}_hp_list.mem", hp_list);
        $readmemh("include/{self.config.project_name}_jump_list.mem", jump_list);
    end

    // Groundtruth calculation function
    function [{output_width-1}:0] calculate_groundtruth;
{chr(10).join(gt_function_params)}
        
        reg [7:0] node_id, next_id, hp_idx;
        reg signed [31:0] hp_val;
        reg signed [31:0] hp_thresh;
        reg [15:0] feedback_base;
        reg signed [31:0] sol_val_0{'' if self.n_solutions == 1 else ', sol_val_1'}{'' if self.n_solutions <= 2 else ', ' + ', '.join([f'sol_val_{i}' for i in range(2, self.n_solutions)])};
        
        begin
            // Tree traversal
            node_id = 0;
            next_id = jump_list[0];
            
            while (next_id != 0) begin
                hp_idx = hp_list[node_id];
                hp_val = {' + '.join([f'($signed(param_{j}) * $signed(halfplanes[hp_idx*`PDAQP_HALFPLANE_STRIDE+{j}]))' for j in range(self.n_parameters)])};
                hp_thresh = $signed(halfplanes[hp_idx*`PDAQP_HALFPLANE_STRIDE + `PDAQP_N_PARAMETER]) * `HALFPLANE_SCALE_FACTOR;
                
                if (hp_val <= hp_thresh) begin
                    node_id = node_id + next_id;
                end else begin
                    node_id = node_id + next_id + 1;
                end
                
                next_id = jump_list[node_id];
            end
            
            hp_idx = hp_list[node_id];
            feedback_base = hp_idx * `PDAQP_SOL_PER_NODE;
            
            // Solution calculation
{chr(10).join([f'            sol_val_{i} = ' + ' + '.join([f'($signed(param_{j}) * $signed(feedbacks[feedback_base+{i}*(`PDAQP_N_PARAMETER+1)+{j}]))' for j in range(self.n_parameters)]) + f' + ($signed(feedbacks[feedback_base+{i}*(`PDAQP_N_PARAMETER+1)+`PDAQP_N_PARAMETER]) << `HALFPLANE_FRAC_BITS);' for i in range(self.n_solutions)])}
            
            calculate_groundtruth = {' | '.join([f'(sol_val_{i}[(`OUTPUT_INT_BITS + `OUTPUT_FRAC_BITS + `HALFPLANE_FRAC_BITS - 1):(`HALFPLANE_FRAC_BITS)] << ({i}*`OUTPUT_DATA_WIDTH))' for i in range(self.n_solutions)])};
        end
    endfunction

    // Generate test data
    initial begin
        for (i = 0; i < TEST_VECTORS; i = i + 1) begin
{chr(10).join([f"            test_params_{j}[i] = $random & 16'hFFFF;" for j in range(self.n_parameters)])}
        end
    end

    // Main test process
    initial begin
        // Initialize
        rst_n = 0;
        s_axis_tvalid = 0;
        s_axis_tdata = 0;
        m_axis_tready = 1;
        output_count = 0;
        errors = 0;
        
        // Calculate expected results
        for (i = 0; i < TEST_VECTORS; i = i + 1) begin
            expected_results[i] = calculate_groundtruth({', '.join([f'test_params_{j}[i]' for j in range(self.n_parameters)])});
        end
        
        $display("\\n=================================================================");
        $display("         DSP-CONSTRAINED {project_name.upper()} TESTBENCH - HYBRID PARALLEL        ");
        $display("=================================================================");
        $display("DSP Resources: {self.current_dsp_usage}/{self.max_dsp_blocks} blocks (sufficient for single-cycle)");
        $display("Strategy: Hybrid Parallel (LUT HP + DSP FB)");
        $display("Test vectors: %0d", TEST_VECTORS);
        $display("Pipeline latency: %0d cycles", PIPELINE_DELAY);
        $display("=================================================================");
        
        // Apply reset
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);
        
        // Send test vectors
        for (test_idx = 0; test_idx < TEST_VECTORS; test_idx = test_idx + 1) begin
            @(posedge clk);
            s_axis_tdata = {{{', '.join([f'test_params_{j}[test_idx]' for j in reversed(range(self.n_parameters))])}}};
            s_axis_tvalid = 1;
            @(posedge clk);
            s_axis_tvalid = 0;
            
            if (test_idx % 10 == 0)
                $display("[%0t ns] Applied test vector %0d/%0d", $time, test_idx+1, TEST_VECTORS);
            
            repeat(PIPELINE_DELAY + 2) @(posedge clk);
        end
        
        // Wait for outputs
        repeat(PIPELINE_DELAY * 2) @(posedge clk);
        
        // Print results
        $display("\\n=================================================================");
        $display("                  HYBRID PARALLEL TEST RESULTS                   ");
        $display("=================================================================");
        $display("  Total vectors:     %0d", TEST_VECTORS);
        $display("  Vectors processed: %0d", output_count);
        $display("  Errors detected:   %0d", errors);
        if (output_count > 0)
            $display("  Error rate:        %.2f%%", (errors*100.0)/output_count);
        $display("  DSP blocks used:   %0d/{self.max_dsp_blocks} ({(self.current_dsp_usage/self.max_dsp_blocks)*100:.1f}%%)", {self.current_dsp_usage});
        $display("  Throughput:        100%% (single-cycle solution computation)");
        $display("=================================================================");
        
        if (errors == 0 && output_count >= TEST_VECTORS)
            $display("               HYBRID PARALLEL TEST PASSED!                     ");
        else
            $display("               HYBRID PARALLEL TEST FAILED!                     ");
        $display("=================================================================");
        
        #(CLK_PERIOD*10)
        $finish;
    end
    
    // Capture outputs
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            actual_results[output_count] = m_axis_tdata;
            
            if (output_count < TEST_VECTORS && m_axis_tdata !== expected_results[output_count]) begin
                $display("[%0t ns] ERROR at vector %0d: Expected %h, Got %h", 
                        $time, output_count, expected_results[output_count], m_axis_tdata);
                errors = errors + 1;
            end
            
            output_count = output_count + 1;
        end
    end
    
    // Waveform dump
    initial begin
        $dumpfile("{project_name}_tb.vcd");
        $dumpvars(0, {project_name}_tb);
    end
    
endmodule
"""
        
        output_file = self.tb_dir / f"{project_name}_tb.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated DSP-constrained testbench: {output_file}")
        return output_file

    def generate_makefile(self):
        """Generate Makefile for DSP-constrained compilation and simulation"""
        project_name = self.config.project_name + "_dsp"
        
        template = f"""# Auto-generated Makefile for DSP-constrained {project_name}
# Simplified version: Hybrid Parallel implementation only
# Resource usage: {self.current_dsp_usage}/{self.max_dsp_blocks} x 16-bit DSP blocks

# Simulation parameters
SIM = iverilog
VIEWER = gtkwave
VERILOG_SOURCES = rtl/{project_name}_bst_lut.v rtl/{project_name}_top.v tb/{project_name}_tb.v
VCD_FILE = {project_name}_tb.vcd
EXECUTABLE = {project_name}_sim

# Default target
all: compile run

# Compile the design
compile:
	@echo "Compiling DSP-constrained {project_name} design..."
	@echo "Strategy: Hybrid Parallel (sufficient DSP resources)"
	@echo "DSP usage: {self.current_dsp_usage}/{self.max_dsp_blocks} blocks"
	$(SIM) -o $(EXECUTABLE) $(VERILOG_SOURCES)

# Run simulation
run: compile
	@echo "Running DSP-constrained {project_name} simulation..."
	./$(EXECUTABLE)

# View waveforms
view: run
	@echo "Opening waveform viewer..."
	$(VIEWER) $(VCD_FILE) &

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f $(EXECUTABLE) $(VCD_FILE) {project_name}_test_results.csv

# Resource usage report
resource_report:
	@echo "DSP Resource Usage Report for {project_name}"
	@echo "==========================================="
	@echo "Implementation: Hybrid Parallel (simplified version)"
	@echo "Target DSP limit: {self.max_dsp_blocks} x 16-bit blocks"
	@echo "Allocated DSP blocks: {self.current_dsp_usage}"
	@echo "Halfplane evaluation: LUT-based (0 DSP)"
	@echo "Feedback calculation: DSP-based ({self.fb_dsp_allocation} DSP)"
	@echo "DSP utilization: {(self.current_dsp_usage/self.max_dsp_blocks)*100:.1f}%"
	@echo "Pipeline latency: {self.bst_depth + 2} cycles"
	@echo "Throughput: 100% (single-cycle solution computation)"

# Help
help:
	@echo "Available targets:"
	@echo "  all             - Compile and run simulation (default)"
	@echo "  compile         - Compile design only"
	@echo "  run             - Run simulation"
	@echo "  view            - Open waveform viewer"
	@echo "  resource_report - Show DSP resource usage"
	@echo "  clean           - Clean generated files"
	@echo "  help            - Show this help"

.PHONY: all compile run view clean resource_report help
"""
        
        output_file = self.base_dir / "Makefile"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated DSP-constrained Makefile: {output_file}")
        return output_file

    def generate_all(self):
        """Generate all DSP-constrained hardware design files"""
        print(f"\nGenerating DSP-constrained hardware design for {self.config.project_name}...")
        print(f"Simplified version: Hybrid Parallel implementation only")
        print(f"DSP resource constraint: {self.max_dsp_blocks} x 16-bit blocks")
        
        # Generate RTL files
        bst_file = self.generate_dsp_limited_bst_lut()
        top_file = self.generate_top_module()
        
        # Generate testbench
        tb_file = self.generate_testbench()
        
        # Generate build files
        make_file = self.generate_makefile()
        
        print(f"\n DSP-constrained hardware generation complete!")
        print(f" Output directory: {self.base_dir}")
        print(f" Generated files:")
        print(f"    RTL: {bst_file.name}, {top_file.name}")
        print(f"    Testbench: {tb_file.name}")
        print(f"    Build: {make_file.name}")
        print(f" DSP resource usage: {self.current_dsp_usage}/{self.max_dsp_blocks} blocks ({(self.current_dsp_usage/self.max_dsp_blocks)*100:.1f}%)")
        print(f" Performance: 100% throughput (single-cycle solution computation)")
        print(f"\n To build and run:")
        print(f"   cd {self.base_dir}")
        print(f"   make run")
        print(f"   make resource_report  # View DSP usage details")
        
        return {
            'bst_lut': bst_file,
            'top': top_file,
            'testbench': tb_file,
            'makefile': make_file
        }

def main():
    parser = argparse.ArgumentParser(description='Generate DSP-constrained hardware design for PDAQP (Simplified - Hybrid Parallel Only)')
    parser.add_argument('config_file', help='Path to configuration .vh file')
    parser.add_argument('-o', '--output', help='Output directory (auto-generated if not specified)')
    parser.add_argument('-d', '--dsp-limit', type=int, default=8, help='DSP resource limit (default: 8)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Parse configuration
    config_parser = VerilogConfigParser(args.config_file)
    
    # Generate DSP-constrained hardware
    generator = DSPLimitedHardwareGenerator(config_parser, args.output)
    generator.max_dsp_blocks = args.dsp_limit  # Allow override of DSP limit
    generator.validate_dsp_allocation()  # Validate with new limit
    
    generated_files = generator.generate_all()
    
    if args.verbose:
        print(f"\nDetailed file information:")
        for file_type, file_path in generated_files.items():
            print(f"  {file_type}: {file_path} ({file_path.stat().st_size} bytes)")

if __name__ == '__main__':
    main()