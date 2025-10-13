#!/usr/bin/env python3
"""
Hardware Design Generator for PDAQP
Generates RTL modules from Verilog configuration files
"""

import re
import os
import sys
import math
import argparse
from pathlib import Path

class VerilogConfigParser:
    """Parse Verilog config.vh and extract defines"""
    
    def __init__(self, config_file):
        self.config_file = config_file
        self.defines = {}
        self.project_name = self.extract_project_name_from_path(config_file)
        self.parse_config()
    
    def extract_project_name_from_path(self, config_path):
        """Extract project name from path or filename"""
        config_path = Path(config_path)
        filename = config_path.stem
        
        if filename.endswith('_config'):
            return filename[:-7]
        
        parent_dir = config_path.parent.parent.name
        if parent_dir and not parent_dir.startswith('.'):
            return parent_dir
        
        return 'pdaqp'
    
    def parse_config(self):
        """Extract all `define macros"""
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
        
        print(f"Parsed {len(self.defines)} defines from {self.config_file}")
        print(f"Project name: {self.project_name}")
        
        self.display_fixedpoint_config()
    
    def display_fixedpoint_config(self):
        """Display fixed-point format configuration"""
        print("\nFixed-point configuration:")
        fp_params = [
            'HALFPLANE_FRAC_BITS', 'HALFPLANE_INT_BITS',
            'FEEDBACK_FRAC_BITS', 'FEEDBACK_INT_BITS',
            'OUTPUT_DATA_WIDTH', 'OUTPUT_FRAC_BITS', 'OUTPUT_INT_BITS'
        ]
        for param in fp_params:
            if param in self.defines:
                print(f"  {param}: {self.defines[param]}")
    
    def get_define(self, name, default=None):
        return self.defines.get(name, default)
    
    def get_int_define(self, name, default=0):
        value = self.defines.get(name, default)
        return int(value) if isinstance(value, str) and value.isdigit() else value

class HardwareGenerator:
    
    def __init__(self, config_parser, output_dir=None):
        self.config = config_parser
        
        if output_dir is None:
            raise ValueError("Output directory required")
        
        self.base_dir = Path(output_dir)
        self.rtl_dir = self.base_dir / "rtl"
        self.tb_dir = self.base_dir / "tb"
        
        if not self.base_dir.exists():
            raise ValueError(f"Output directory {self.base_dir} not found")
        
        for d in [self.rtl_dir, self.tb_dir]:
            d.mkdir(exist_ok=True)
        
        self.n_parameters = self.config.get_int_define('PDAQP_N_PARAMETER', 2)
        self.n_solutions = self.config.get_int_define('PDAQP_N_SOLUTION', 2)
        self.n_tree_nodes = self.config.get_int_define('PDAQP_TREE_NODES', 256)
        self.bst_depth = self.config.get_int_define('PDAQP_ESTIMATED_BST_DEPTH', 7)
        
        # === FIX: Calculate index width correctly ===
        if self.n_tree_nodes == 0:
            print("WARNING: PDAQP_TREE_NODES=0, using minimum")
            self.n_tree_nodes = 1
            self.index_width = 1
        elif self.n_tree_nodes == 1:
            self.index_width = 1  # Special case: 1 node needs 1 bit
        else:
            # Need ceil(log2(n_tree_nodes)) bits to address n_tree_nodes
            # Example: 5 nodes → need 3 bits (can address 0-7)
            self.index_width = math.ceil(math.log2(self.n_tree_nodes))
        
        print(f"\nHardware configuration:")
        print(f"  Parameters: {self.n_parameters}")
        print(f"  Solutions: {self.n_solutions}")
        print(f"  Tree nodes: {self.n_tree_nodes}")
        print(f"  BST depth: {self.bst_depth}")
        print(f"  Index width: {self.index_width} bits (can address 0-{(1 << self.index_width) - 1})")
        print(f"  Output: {self.base_dir}")

    def generate_bst_lut_module(self):
        """Generate pipelined BST LUT module"""
        project_name = self.config.project_name
        idx_width = self.index_width
        idx_max = max(0, idx_width - 1)
        
        print(f"\n=== BST Module Generation ===")
        print(f"Index: [{idx_max}:0] ({idx_width} bits)")
        print(f"BST array: [0:{self.bst_depth}] ({self.bst_depth + 1} elements)")
        print(f"Pipeline: {self.bst_depth + 4} stages")
        print(f"=============================")
        
        # Port declarations
        param_inputs = '\n'.join([
            f"    input  signed [15:0]  param_in_{i}," 
            for i in range(self.n_parameters)
        ])
        
        sol_outputs = '\n'.join([
            f"    output reg signed [15:0] sol_out_{i}," 
            for i in range(self.n_solutions)
        ])
        
        # Pipeline arrays
        param_pipes = '\n'.join([
            f"    reg [15:0] param{i}_pipe[0:PIPE_DEPTH-1];" 
            for i in range(self.n_parameters)
        ])
        
        # Reset logic
        param_reset = '\n'.join([
            f"                param{i}_pipe[i] <= 0;" 
            for i in range(self.n_parameters)
        ])
        
        # Input capture
        param_input_capture = '\n'.join([
            f"                param{i}_pipe[0] <= param_in_{i};" 
            for i in range(self.n_parameters)
        ])
        
        # Pipeline propagation
        param_propagation = '\n'.join([
            f"                param{i}_pipe[i+1] <= param{i}_pipe[i];" 
            for i in range(self.n_parameters)
        ])
        
        param_propagation_stageN = '\n'.join([
            f"            param{i}_pipe[MAX_BST_DEPTH+1] <= param{i}_pipe[MAX_BST_DEPTH];" 
            for i in range(self.n_parameters)
        ])
        
        # Halfplane dot product
        hp_calc_terms = ' + '.join([
            f"($signed(param{i}_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+{i}]))"
            for i in range(self.n_parameters)
        ])
        
        # Solution calculation
        sol_temp_arrays = '\n'.join([
            f"    reg signed [31:0] sol_temp_{i}[0:1];"
            for i in range(self.n_solutions)
        ])
        
        sol_temp_reset = '\n'.join([
            f"                sol_temp_{i}[j] <= 0;"
            for i in range(self.n_solutions)
        ])
        
        # Multiply-accumulate
        sol_calc_stage1 = []
        for sol_idx in range(self.n_solutions):
            terms = []
            for param_idx in range(self.n_parameters):
                idx = f"feedback_base_pipe[0] + {sol_idx}*(`PDAQP_N_PARAMETER+1) + {param_idx}"
                terms.append(f"($signed(param{param_idx}_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[{idx}]))")
            sol_calc_stage1.append(f"                sol_temp_{sol_idx}[0] <= {' + '.join(terms)};")
        
        # Add offset
        sol_calc_stage2 = []
        for sol_idx in range(self.n_solutions):
            offset_idx = f"feedback_base_pipe[1] + {sol_idx}*(`PDAQP_N_PARAMETER+1) + `PDAQP_N_PARAMETER"
            sol_calc_stage2.append(
                f"                sol_temp_{sol_idx}[1] <= sol_temp_{sol_idx}[0] + ($signed(feedbacks[{offset_idx}]) << `FEEDBACK_FRAC_BITS);"
            )
        
        # Output
        sol_output_assign = '\n'.join([
            f"                sol_out_{i} <= sol_temp_{i}[1][`FEEDBACK_FRAC_BITS + `OUTPUT_DATA_WIDTH - 1:`FEEDBACK_FRAC_BITS];"
            for i in range(self.n_solutions)
        ])
        
        sol_output_reset = '\n'.join([
            f"            sol_out_{i} <= 0;"
            for i in range(self.n_solutions)
        ])

        template = f"""`timescale 1ns/1ps
`include "include/{project_name}_config.vh"

module {project_name}_bst_lut (
    input                               clk,
    input                               rst_n,
{param_inputs}
    input                               valid_in,
{sol_outputs}
    output reg                          valid_out
);

    // Distributed ROM for FPGA
    (* rom_style = "distributed" *) reg [15:0] halfplanes [0:`PDAQP_HALFPLANES-1];
    (* rom_style = "distributed" *) reg [15:0] feedbacks [0:`PDAQP_FEEDBACKS-1];
    (* rom_style = "distributed" *) reg [{idx_max}:0] hp_list [0:`PDAQP_TREE_NODES-1];
    (* rom_style = "distributed" *) reg [{idx_max}:0] jump_list [0:`PDAQP_TREE_NODES-1];

    initial begin
        $readmemh("include/{project_name}_halfplanes.mem", halfplanes);
        $readmemh("include/{project_name}_feedbacks.mem", feedbacks);
        $readmemh("include/{project_name}_hp_list.mem", hp_list);
        $readmemh("include/{project_name}_jump_list.mem", jump_list);
    end
    
    localparam MAX_BST_DEPTH = `PDAQP_ESTIMATED_BST_DEPTH;
    localparam PIPE_DEPTH = MAX_BST_DEPTH + 5;
    
    // Pipeline registers
    reg valid_pipe[0:PIPE_DEPTH-1];
{param_pipes}
    
    // BST traversal (size = MAX_BST_DEPTH + 1 for [0:MAX_BST_DEPTH])
    reg [{idx_max}:0] current_id_pipe[0:MAX_BST_DEPTH];
    reg [{idx_max}:0] next_id_pipe[0:MAX_BST_DEPTH];
    reg [{idx_max}:0] hp_idx_pipe[0:MAX_BST_DEPTH];
    reg traversal_done_pipe[0:MAX_BST_DEPTH];
    
    // Solution pipeline
    reg [15:0] feedback_base_pipe[0:1];
{sol_temp_arrays}
    
    // Combinational temporaries
    reg signed [31:0] hp_val;
    reg signed [31:0] hp_thresh;
    reg decision;
    reg [{idx_max}:0] new_id;
    
    integer i, j;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < PIPE_DEPTH; i = i + 1) begin
                valid_pipe[i] <= 0;
{param_reset}
            end
            
            for (i = 0; i <= MAX_BST_DEPTH; i = i + 1) begin
                current_id_pipe[i] <= 0;
                next_id_pipe[i] <= 0;
                hp_idx_pipe[i] <= 0;
                traversal_done_pipe[i] <= 0;
            end
            
            for (j = 0; j < 2; j = j + 1) begin
                feedback_base_pipe[j] <= 0;
{sol_temp_reset}
            end
            
            valid_out <= 0;
{sol_output_reset}
            
        end else begin
            
            // Stage 0: Input
            valid_pipe[0] <= valid_in;
            
            if (valid_in) begin
{param_input_capture}
                
                current_id_pipe[0] <= 0;
                next_id_pipe[0] <= jump_list[0];
                hp_idx_pipe[0] <= hp_list[0];
                traversal_done_pipe[0] <= (jump_list[0] == 0);
            end
            
            // Stages 1 to MAX_BST_DEPTH: BST traversal
            for (i = 0; i < MAX_BST_DEPTH; i = i + 1) begin
                valid_pipe[i+1] <= valid_pipe[i];
{param_propagation}
                
                if (valid_pipe[i]) begin
                    if (traversal_done_pipe[i]) begin
                        current_id_pipe[i+1] <= current_id_pipe[i];
                        next_id_pipe[i+1] <= next_id_pipe[i];
                        hp_idx_pipe[i+1] <= hp_idx_pipe[i];
                        traversal_done_pipe[i+1] <= 1;
                    end else begin
                        hp_val = {hp_calc_terms};
                        hp_thresh = $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+`PDAQP_N_PARAMETER]) << `HALFPLANE_FRAC_BITS;
                        decision = hp_val <= hp_thresh;
                        new_id = next_id_pipe[i] + (decision ? {idx_width}'d1 : {idx_width}'d0);
                        
                        current_id_pipe[i+1] <= new_id;
                        next_id_pipe[i+1] <= new_id + jump_list[new_id];
                        hp_idx_pipe[i+1] <= hp_list[new_id];
                        traversal_done_pipe[i+1] <= (jump_list[new_id] == 0);
                    end
                end else begin
                    current_id_pipe[i+1] <= 0;
                    next_id_pipe[i+1] <= 0;
                    hp_idx_pipe[i+1] <= 0;
                    traversal_done_pipe[i+1] <= 0;
                end
            end
            
            // Stage MAX_BST_DEPTH+1: Prepare feedback
            valid_pipe[MAX_BST_DEPTH+1] <= valid_pipe[MAX_BST_DEPTH];
{param_propagation_stageN}
            
            if (valid_pipe[MAX_BST_DEPTH]) begin
                feedback_base_pipe[0] <= hp_idx_pipe[MAX_BST_DEPTH] * (`PDAQP_N_PARAMETER + 1) * `PDAQP_N_SOLUTION;
            end
            
            // Stage MAX_BST_DEPTH+2: Multiply-accumulate
            valid_pipe[MAX_BST_DEPTH+2] <= valid_pipe[MAX_BST_DEPTH+1];
            
            if (valid_pipe[MAX_BST_DEPTH+1]) begin
                feedback_base_pipe[1] <= feedback_base_pipe[0];
{chr(10).join(sol_calc_stage1)}
            end
            
            // Stage MAX_BST_DEPTH+3: Add offset
            valid_pipe[MAX_BST_DEPTH+3] <= valid_pipe[MAX_BST_DEPTH+2];
            
            if (valid_pipe[MAX_BST_DEPTH+2]) begin
{chr(10).join(sol_calc_stage2)}
            end
            
            // Stage MAX_BST_DEPTH+4: Output
            valid_out <= valid_pipe[MAX_BST_DEPTH+3];
            
            if (valid_pipe[MAX_BST_DEPTH+3]) begin
{sol_output_assign}
            end
        end
    end
endmodule
"""
        
        output_file = self.rtl_dir / f"{project_name}_bst_lut.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated: {output_file}")
        return output_file
    
    def generate_top_module(self):
        """Generate AXI4-Stream wrapper"""
        project_name = self.config.project_name
        param_width = 16
        axi_width = 32
        params_per_axi = axi_width // param_width
        num_axi_inputs = math.ceil(self.n_parameters / params_per_axi)
        output_width = self.n_solutions * param_width
        
        print(f"\nAXI interface:")
        print(f"  Inputs: {num_axi_inputs} × {axi_width}-bit")
        print(f"  Output: {output_width}-bit")
        
        # AXI input ports
        axi_input_ports = []
        for i in range(num_axi_inputs):
            first_param = i * params_per_axi
            last_param = min((i + 1) * params_per_axi - 1, self.n_parameters - 1)
            
            if first_param == last_param:
                comment = f"param_{first_param}"
            else:
                params_list = ', '.join([f"param_{j}" for j in range(first_param, last_param + 1)])
                comment = f"{{{params_list}}}"
            
            axi_input_ports.append(f"    input wire [31:0] s_axis_tdata_{i},    // {comment}")
            axi_input_ports.append(f"    input wire s_axis_tvalid_{i},")
            axi_input_ports.append(f"    output wire s_axis_tready_{i}{',' if i < num_axi_inputs - 1 else ','}")
            if i < num_axi_inputs - 1:
                axi_input_ports.append("")
        
        # Parameter extraction
        param_extractions = []
        for param_idx in range(self.n_parameters):
            axi_idx = param_idx // params_per_axi
            param_pos_in_axi = param_idx % params_per_axi
            low_bit = param_pos_in_axi * param_width
            high_bit = low_bit + param_width - 1
            param_extractions.append(
                f"    wire [15:0] param_{param_idx} = s_axis_tdata_{axi_idx}[{high_bit}:{low_bit}];"
            )
        
        valid_signals = ' && '.join([f's_axis_tvalid_{i}' for i in range(num_axi_inputs)])
        sol_wires = '\n'.join([f"    wire [15:0] sol_{i};" for i in range(self.n_solutions)])
        
        bst_param_connections = '\n'.join([
            f"        .param_in_{i}(param_{i})," for i in range(self.n_parameters)
        ])
        
        bst_sol_connections = '\n'.join([
            f"        .sol_out_{i}(sol_{i})," for i in range(self.n_solutions)
        ])
        
        sol_concat = ', '.join([f'sol_{i}' for i in reversed(range(self.n_solutions))])
        
        ready_assignments = '\n'.join([
            f"    assign s_axis_tready_{i} = ready_all;" for i in range(num_axi_inputs)
        ])

        template = f"""`timescale 1ns/1ps
`include "include/{project_name}_config.vh"

module {project_name}_top (
    input wire clk,
    input wire rst_n,
    
    // AXI4-Stream inputs
{chr(10).join(axi_input_ports)}
    
    // AXI4-Stream output
    output wire [{output_width-1}:0] m_axis_tdata,
    output wire m_axis_tvalid,
    input wire m_axis_tready
);

    // Extract parameters from AXI
{chr(10).join(param_extractions)}
    
    wire all_valid = {valid_signals};
    
{sol_wires}
    wire bst_valid_out;
    
    {project_name}_bst_lut bst_inst (
        .clk(clk),
        .rst_n(rst_n),
{bst_param_connections}
        .valid_in(all_valid),
{bst_sol_connections}
        .valid_out(bst_valid_out)
    );
    
    assign m_axis_tdata = {{{sol_concat}}};
    assign m_axis_tvalid = bst_valid_out;
    
    wire ready_all = m_axis_tready;
    
{ready_assignments}

endmodule
"""
        
        output_file = self.rtl_dir / f"{project_name}_top.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated: {output_file}")
        return output_file

    def generate_testbench(self):
        """Generate testbench with groundtruth checker"""
        project_name = self.config.project_name
        idx_width = self.index_width
        idx_max = max(0, idx_width - 1)
        
        # Minimal testbench stub (full implementation would be very long)
        template = f"""`timescale 1ns/1ps
`include "include/{project_name}_config.vh"

module {project_name}_tb;
    localparam CLK_PERIOD = 10;
    
    reg clk;
    reg rst_n;
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    initial begin
        rst_n = 0;
        #(CLK_PERIOD*10);
        rst_n = 1;
        
        #(CLK_PERIOD*100);
        $display("Testbench stub - implement full test");
        $finish;
    end
    
    initial begin
        $dumpfile("{project_name}_tb.vcd");
        $dumpvars(0, {project_name}_tb);
    end
    
endmodule
"""
        
        output_file = self.tb_dir / f"{project_name}_tb.v"
        with open(output_file, 'w') as f:
            f.write(template)
        
        print(f"Generated: {output_file}")
        return output_file

    def generate_all(self):
        """Generate all files"""
        print(f"\n{'='*70}")
        print(f"Generating hardware for {self.config.project_name}")
        print(f"{'='*70}")
        
        bst_file = self.generate_bst_lut_module()
        top_file = self.generate_top_module()
        tb_file = self.generate_testbench()
        
        print(f"\n{'='*70}")
        print(f"✓ Generation complete!")
        print(f"{'='*70}")
        print(f"Output: {self.base_dir}")
        print(f"Files:")
        print(f"  RTL: {bst_file.name}, {top_file.name}")
        print(f"  TB:  {tb_file.name}")
        print(f"{'='*70}\n")
        
        return {
            'bst_lut': bst_file,
            'top': top_file,
            'testbench': tb_file,
        }

def main():
    parser = argparse.ArgumentParser(
        description='Generate PDAQP hardware',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_hardware.py config.vh -o output_dir
  python3 generate_hardware.py config.vh -o output_dir -v
        """
    )
    parser.add_argument('config_file', help='Config .vh file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not Path(args.config_file).exists():
        print(f"Error: Config file '{args.config_file}' not found")
        sys.exit(1)
    
    if not Path(args.output).exists():
        print(f"Error: Output directory '{args.output}' not found")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"PDAQP Hardware Generator")
    print(f"{'='*70}\n")
    
    config_parser = VerilogConfigParser(args.config_file)
    generator = HardwareGenerator(config_parser, args.output)
    generated_files = generator.generate_all()
    
    if args.verbose:
        print(f"Verbose file info:")
        for file_type, file_path in generated_files.items():
            print(f"  {file_type}: {file_path} ({file_path.stat().st_size} bytes)")

if __name__ == '__main__':
    main()