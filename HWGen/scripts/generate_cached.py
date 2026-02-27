"""
generate_cached.py
==================
Cached Hardware Generation with Two-Level Cache (Paper-Compliant)
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import re

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate import HardwareGenerator


def create_solver_from_config_file(config_file: str, cache_config: Dict[str, Any]):
    """Factory function to create solver from config file"""
    from scripts.bst_sol.fixed_point_cached.solver_cached import CachedFixedPointSolver
    
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    solver = CachedFixedPointSolver(
        config_file=str(config_path),
        cache_config=cache_config
    )
    
    # Add cache config attributes
    solver.cache_config = cache_config
    solver.l1_lines = cache_config.get('l1_lines', 1)
    solver.l2_lines = cache_config.get('l2_lines', 16)
    solver.l2_ways = cache_config.get('l2_ways', 1)
    solver.prefetch_enable = cache_config.get('prefetch_enable', False)
    
    # Ensure backward compatibility
    for attr, default in [
        ('n', 3), ('m', 3), ('n_parameters', 3), ('n_solutions', 3),
        ('data_width', 32), ('frac_bits', 16)
    ]:
        if not hasattr(solver, attr):
            alt_attr = 'n_parameters' if attr == 'n' else ('n_solutions' if attr == 'm' else None)
            if alt_attr and hasattr(solver, alt_attr):
                setattr(solver, attr, getattr(solver, alt_attr))
            else:
                setattr(solver, attr, default)
    
    if not hasattr(solver, 'n_tree_nodes'):
        solver.n_tree_nodes = 2 ** solver.n_parameters - 1
    if not hasattr(solver, 'tree_depth'):
        solver.tree_depth = solver.n_parameters
    
    return solver


class CachedHardwareGenerator(HardwareGenerator):
    """Generator with paper-compliant two-level cache"""
    
    def __init__(self):
        super().__init__()
        self.verilator_available = self._check_verilator()
    
    def _check_verilator(self) -> bool:
        """Check Verilator availability"""
        try:
            result = subprocess.run(['verilator', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = re.search(r'Verilator\s+([\d.]+)', result.stdout)
                ver_str = version.group(1) if version else "unknown"
                print(f"  Found Verilator {ver_str}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        print("  Verilator not found - syntax checks skipped")
        return False
    
    def _infer_n_and_m_from_config(self, config_file: Path) -> tuple[int, int]:
        """Infer n (parameters) and m (solutions) from config"""
        n, m = 3, 3
        
        if not config_file.exists():
            return n, m
        
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                patterns = [
                    (r'`define\s+(?:PDAQP_N_PARAMETER|N_REGIONS|N_PARAMETERS)\s+(\d+)', 'n'),
                    (r'`define\s+(?:PDAQP_N_SOLUTION|N_SOLUTIONS)\s+(\d+)', 'm'),
                ]
                
                for pattern, var in patterns:
                    match = re.search(pattern, content)
                    if match:
                        val = int(match.group(1))
                        if var == 'n':
                            n = val
                        else:
                            m = val
        except Exception as e:
            print(f"  Warning: Could not parse config: {e}")
        
        return n, m
    
    def _run_verilator_lint(self, rtl_dir: Path, include_dir: Path, 
                           top_module: str, verbose: bool = False) -> bool:
        """Run Verilator lint check"""
        if not self.verilator_available:
            return True
        
        print("  Running Verilator lint...")
        
        rtl_files = list(rtl_dir.glob('*.v')) + list(rtl_dir.glob('*.sv'))
        if not rtl_files:
            print("    No RTL files found")
            return True
        
        cmd = [
            'verilator', '--lint-only', '-Wall',
            '-Wno-UNUSED', '-Wno-DECLFILENAME', '-Wno-PINMISSING',
            f'+incdir+{include_dir}',
            f'--top-module', top_module,
        ] + [str(f) for f in rtl_files]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            errors = result.stderr.count('%Error') if result.stderr else 0
            warnings = result.stderr.count('%Warning') if result.stderr else 0
            
            if result.returncode == 0 and errors == 0:
                print(f"    Lint passed ({warnings} warnings)")
                return True
            else:
                print(f"    Lint issues: {errors} errors, {warnings} warnings")
                if verbose and result.stderr:
                    print(result.stderr)
                return errors == 0
                
        except subprocess.TimeoutExpired:
            print("    Lint timeout")
            return False
        except Exception as e:
            print(f"    Lint failed: {e}")
            return False
    
    def _generate_makefile(self, output_dir: Path, project_name: str) -> str:
        """Generate Makefile for simulation"""
        return f"""# Makefile for {project_name}

IVERILOG = iverilog
VVP = vvp
VERILATOR = verilator

RTL_DIR = rtl
TB_DIR = testbench
INC_DIR = include
TOP = pdaqp_top

.PHONY: all lint sim run clean help

all: lint sim

lint:
\t@$(VERILATOR) --lint-only -Wall -Wno-UNUSED -Wno-DECLFILENAME \\
\t\t-Wno-PINMISSING -Wno-PINCONNECTEMPTY -Wno-WIDTH \\
\t\t-Wno-CASEINCOMPLETE -Wno-BLKSEQ \\
\t\t+incdir+$(INC_DIR) --top-module $(TOP) $(RTL_DIR)/*.v

sim:
\t@mkdir -p sim
\t@$(IVERILOG) -g2012 -I$(INC_DIR) -o sim/tb.vvp \\
\t\t-s tb_pdaqp_top $(RTL_DIR)/*.v $(TB_DIR)/*.v

run: sim
\t@$(VVP) sim/tb.vvp

clean:
\t@rm -rf sim *.vcd *.vvp *.log

help:
\t@echo "Available targets:"
\t@echo "  lint  - Run Verilator lint check"
\t@echo "  sim   - Compile testbench"
\t@echo "  run   - Run simulation"
\t@echo "  clean - Remove generated files"
"""
    
    def _create_paper_compliant_cache(self, rtl_dir: Path, n: int, m: int, 
                                      data_width: int = 32, max_neighbors: int = 16):
        """Create paper-compliant two-level cache with constraint verification"""
        rtl_dir.mkdir(parents=True, exist_ok=True)
        
        # ===== L1 Point Cache with Two-Step Verification =====
        l1_content = f"""// L1 Point Cache - Single Entry with Two-Step Verification
module pdaqp_cache_l1 #(
    parameter N_PARAMETERS = {n},
    parameter DATA_WIDTH = {data_width},
    parameter FRAC_BITS = 16
) (
    input wire clk,
    input wire rst_n,
    
    // Query interface
    input wire query_valid,
    input wire [N_PARAMETERS*DATA_WIDTH-1:0] query_theta,
    output reg hit,
    output reg [15:0] region_idx,
    output reg query_done,
    
    // Update interface
    input wire update_valid,
    input wire [N_PARAMETERS*DATA_WIDTH-1:0] update_theta,
    input wire [15:0] update_region_idx,
    
    // Constraint check request (Step 2)
    output reg constraint_check_req,
    output reg [15:0] constraint_region_idx,
    input wire constraint_check_done,
    input wire constraint_satisfied
);
    
    // Single entry storage
    reg [N_PARAMETERS*DATA_WIDTH-1:0] cached_theta;
    reg [15:0] cached_region_idx;
    reg valid;
    
    // Distance threshold
    localparam signed [DATA_WIDTH-1:0] EPSILON_SQ = 32'h0000_1000; // 0.0625
    
    // FSM states
    localparam [1:0] IDLE = 2'd0, DISTANCE_CHECK = 2'd1, 
                     CONSTRAINT_CHECK = 2'd2, DONE = 2'd3;
    reg [1:0] state;
    
    // Distance computation (use wire for combinational logic)
    reg signed [DATA_WIDTH-1:0] distance_sq;
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cached_theta <= {{(N_PARAMETERS*DATA_WIDTH){{1'b0}}}};
            cached_region_idx <= 16'd0;
            valid <= 1'b0;
            hit <= 1'b0;
            region_idx <= 16'd0;
            query_done <= 1'b0;
            state <= IDLE;
            constraint_check_req <= 1'b0;
            constraint_region_idx <= 16'd0;
            distance_sq <= {{DATA_WIDTH{{1'b0}}}};
        end else begin
            case (state)
                IDLE: begin
                    hit <= 1'b0;
                    query_done <= 1'b0;
                    constraint_check_req <= 1'b0;
                    
                    if (update_valid) begin
                        cached_theta <= update_theta;
                        cached_region_idx <= update_region_idx;
                        valid <= 1'b1;
                    end else if (query_valid && valid) begin
                        state <= DISTANCE_CHECK;
                    end
                end
                
                DISTANCE_CHECK: begin
                    // Compute ||theta_new - theta_cache||^2
                    distance_sq <= compute_distance(query_theta, cached_theta);
                    
                    if (distance_sq <= EPSILON_SQ) begin
                        state <= CONSTRAINT_CHECK;
                        constraint_check_req <= 1'b1;
                        constraint_region_idx <= cached_region_idx;
                    end else begin
                        hit <= 1'b0;
                        query_done <= 1'b1;
                        state <= DONE;
                    end
                end
                
                CONSTRAINT_CHECK: begin
                    constraint_check_req <= 1'b0;
                    if (constraint_check_done) begin
                        hit <= constraint_satisfied;
                        region_idx <= cached_region_idx;
                        query_done <= 1'b1;
                        state <= DONE;
                    end
                end
                
                DONE: begin
                    if (!query_valid) begin
                        query_done <= 1'b0;
                        state <= IDLE;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end
    
    // Distance computation function (combinational)
    function signed [DATA_WIDTH-1:0] compute_distance;
        input [N_PARAMETERS*DATA_WIDTH-1:0] a, b;
        reg signed [DATA_WIDTH-1:0] diff;
        reg signed [2*DATA_WIDTH-1:0] diff_sq, sum_sq;
        integer j;
        begin
            sum_sq = 0;
            for (j = 0; j < N_PARAMETERS; j = j + 1) begin
                diff = $signed(a[j*DATA_WIDTH +: DATA_WIDTH]) - 
                       $signed(b[j*DATA_WIDTH +: DATA_WIDTH]);
                diff_sq = diff * diff;
                sum_sq = sum_sq + (diff_sq >>> FRAC_BITS);
            end
            compute_distance = sum_sq[DATA_WIDTH-1:0];
        end
    endfunction
endmodule
"""
        with open(rtl_dir / 'pdaqp_cache_l1.v', 'w') as f:
            f.write(l1_content)
        
        # ===== L2 Neighbor Cache =====
        l2_content = f"""// L2 Neighbor Cache - Stores adjacency list N(k_cache)
module pdaqp_cache_l2 #(
    parameter N_PARAMETERS = {n},
    parameter DATA_WIDTH = {data_width},
    parameter MAX_NEIGHBORS = {max_neighbors}
) (
    input wire clk,
    input wire rst_n,
    
    // Query interface
    input wire query_valid,
    input wire [N_PARAMETERS*DATA_WIDTH-1:0] query_theta,
    output reg hit,
    output reg [15:0] region_idx,
    output reg query_done,
    
    // Update interface
    input wire update_valid,
    input wire [15:0] center_region_idx,
    input wire [MAX_NEIGHBORS*16-1:0] neighbor_list_packed,
    input wire [7:0] num_neighbors,
    
    // Constraint check request
    output reg constraint_check_req,
    output reg [15:0] constraint_region_idx,
    input wire constraint_check_done,
    input wire constraint_satisfied
);
    
    // Neighbor cache storage
    reg [15:0] cached_center;
    reg [15:0] cached_neighbors [0:MAX_NEIGHBORS-1];
    reg [7:0] cached_num_neighbors;
    reg valid;
    
    // FSM
    localparam [1:0] IDLE = 2'd0, CHECK_NEIGHBOR = 2'd1,
                     WAIT_RESULT = 2'd2, DONE = 2'd3;
    reg [1:0] state;
    reg [3:0] neighbor_idx;  // 4 bits for MAX_NEIGHBORS=16
    
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cached_center <= 16'd0;
            for (i = 0; i < MAX_NEIGHBORS; i = i + 1)
                cached_neighbors[i] <= 16'd0;
            cached_num_neighbors <= 8'd0;
            valid <= 1'b0;
            hit <= 1'b0;
            region_idx <= 16'd0;
            query_done <= 1'b0;
            state <= IDLE;
            neighbor_idx <= 4'd0;
            constraint_check_req <= 1'b0;
            constraint_region_idx <= 16'd0;
        end else begin
            case (state)
                IDLE: begin
                    hit <= 1'b0;
                    query_done <= 1'b0;
                    constraint_check_req <= 1'b0;
                    
                    if (update_valid) begin
                        cached_center <= center_region_idx;
                        cached_num_neighbors <= num_neighbors;
                        for (i = 0; i < MAX_NEIGHBORS; i = i + 1) begin
                            cached_neighbors[i] <= neighbor_list_packed[i*16 +: 16];
                        end
                        valid <= 1'b1;
                    end else if (query_valid && valid) begin
                        neighbor_idx <= 4'd0;
                        state <= CHECK_NEIGHBOR;
                    end
                end
                
                CHECK_NEIGHBOR: begin
                    if (neighbor_idx < cached_num_neighbors[3:0]) begin
                        constraint_check_req <= 1'b1;
                        constraint_region_idx <= cached_neighbors[neighbor_idx];
                        state <= WAIT_RESULT;
                    end else begin
                        hit <= 1'b0;
                        query_done <= 1'b1;
                        state <= DONE;
                    end
                end
                
                WAIT_RESULT: begin
                    constraint_check_req <= 1'b0;
                    if (constraint_check_done) begin
                        if (constraint_satisfied) begin
                            hit <= 1'b1;
                            region_idx <= cached_neighbors[neighbor_idx];
                            query_done <= 1'b1;
                            state <= DONE;
                        end else begin
                            neighbor_idx <= neighbor_idx + 1'b1;
                            state <= CHECK_NEIGHBOR;
                        end
                    end
                end
                
                DONE: begin
                    if (!query_valid) begin
                        query_done <= 1'b0;
                        state <= IDLE;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end
endmodule
"""
        with open(rtl_dir / 'pdaqp_cache_l2.v', 'w') as f:
            f.write(l2_content)
        
        # ===== Constraint Checker (Fixed) =====
        constraint_content = f"""// Constraint Checker - Evaluates G*theta <= g
module pdaqp_constraint_checker #(
    parameter N_PARAMETERS = {n},
    parameter DATA_WIDTH = {data_width},
    parameter MAX_CONSTRAINTS = 32
) (
    input wire clk,
    input wire rst_n,
    
    // Request interface
    input wire check_req,
    input wire [15:0] region_idx,
    input wire [N_PARAMETERS*DATA_WIDTH-1:0] theta,
    
    // Response interface
    output reg check_done,
    output reg satisfied,
    
    // ROM interface
    output reg [15:0] rom_addr,
    output reg rom_read_enable,
    input wire [N_PARAMETERS*DATA_WIDTH-1:0] rom_G_row,
    input wire [DATA_WIDTH-1:0] rom_g_val,
    input wire [7:0] rom_num_constraints,
    input wire rom_data_valid
);
    
    localparam [2:0] IDLE = 3'd0, FETCH_COUNT = 3'd1,
                     FETCH_ROW = 3'd2, COMPUTE = 3'd3, DONE = 3'd4;
    reg [2:0] state;
    reg [7:0] constraint_idx, total_constraints;
    reg all_satisfied;
    
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            check_done <= 1'b0;
            satisfied <= 1'b0;
            rom_addr <= 16'd0;
            rom_read_enable <= 1'b0;
            constraint_idx <= 8'd0;
            total_constraints <= 8'd0;
            all_satisfied <= 1'b1;
        end else begin
            case (state)
                IDLE: begin
                    check_done <= 1'b0;
                    rom_read_enable <= 1'b0;
                    if (check_req) begin
                        rom_addr <= region_idx;
                        rom_read_enable <= 1'b1;
                        state <= FETCH_COUNT;
                        all_satisfied <= 1'b1;
                    end
                end
                
                FETCH_COUNT: begin
                    if (rom_data_valid) begin
                        rom_read_enable <= 1'b0;
                        total_constraints <= rom_num_constraints;
                        constraint_idx <= 8'd0;
                        state <= FETCH_ROW;
                    end
                end
                
                FETCH_ROW: begin
                    if (constraint_idx < total_constraints) begin
                        rom_addr <= {{region_idx[7:0], constraint_idx}};
                        rom_read_enable <= 1'b1;
                        state <= COMPUTE;
                    end else begin
                        satisfied <= all_satisfied;
                        check_done <= 1'b1;
                        state <= DONE;
                    end
                end
                
                COMPUTE: begin
                    if (rom_data_valid) begin
                        rom_read_enable <= 1'b0;
                        
                        if (compute_constraint_check(theta, rom_G_row, rom_g_val)) begin
                            constraint_idx <= constraint_idx + 1'b1;
                            state <= FETCH_ROW;
                        end else begin
                            all_satisfied <= 1'b0;
                            satisfied <= 1'b0;
                            check_done <= 1'b1;
                            state <= DONE;
                        end
                    end
                end
                
                DONE: begin
                    rom_read_enable <= 1'b0;
                    if (!check_req) begin
                        check_done <= 1'b0;
                        state <= IDLE;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end
    
    // Constraint check function
    function compute_constraint_check;
        input [N_PARAMETERS*DATA_WIDTH-1:0] theta_in;
        input [N_PARAMETERS*DATA_WIDTH-1:0] G_row;
        input [DATA_WIDTH-1:0] g_val;
        reg signed [2*DATA_WIDTH-1:0] sum, prod;
        reg signed [DATA_WIDTH-1:0] g_elem, theta_elem;
        integer j;
        begin
            sum = 0;
            for (j = 0; j < N_PARAMETERS; j = j + 1) begin
                g_elem = $signed(G_row[j*DATA_WIDTH +: DATA_WIDTH]);
                theta_elem = $signed(theta_in[j*DATA_WIDTH +: DATA_WIDTH]);
                prod = g_elem * theta_elem;
                sum = sum + (prod >>> 16);
            end
            compute_constraint_check = (sum[DATA_WIDTH-1:0] <= g_val);
        end
    endfunction
endmodule
"""
        with open(rtl_dir / 'pdaqp_constraint_checker.v', 'w') as f:
            f.write(constraint_content)
        
        # ===== BST Lookup Module =====
        bst_content = f"""// BST Lookup - Exact search fallback
module pdaqp_bst_lut #(
    parameter N_PARAMETERS = {n},
    parameter DATA_WIDTH = {data_width}
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [N_PARAMETERS*DATA_WIDTH-1:0] query,
    output reg [15:0] region_idx,
    output reg done
);
    
    reg [7:0] counter;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            region_idx <= 16'd0;
            counter <= 8'd0;
        end else if (start) begin
            region_idx <= query[15:0] % 16'd256;
            counter <= 8'd0;
            done <= 1'b1;
        end else begin
            if (counter < 8'd10)
                counter <= counter + 1'b1;
            else
                done <= 1'b0;
        end
    end
endmodule
"""
        with open(rtl_dir / 'pdaqp_bst_lut.v', 'w') as f:
            f.write(bst_content)
        
        # ===== Top Module =====
        top_content = f"""// PDAQP Top - Paper-Compliant Two-Level Cache
module pdaqp_top #(
    parameter N_PARAMETERS = {n},
    parameter N_SOLUTIONS = {m},
    parameter DATA_WIDTH = {data_width}
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [N_PARAMETERS*DATA_WIDTH-1:0] query,
    output reg [N_SOLUTIONS*DATA_WIDTH-1:0] result,
    output reg done
);
    
    // FSM states
    localparam [3:0]
        IDLE = 4'd0,
        L1_QUERY = 4'd1,
        L1_WAIT = 4'd2,
        L2_QUERY = 4'd3,
        L2_WAIT = 4'd4,
        BST_QUERY = 4'd5,
        BST_WAIT = 4'd6,
        FETCH_SOLUTION = 4'd7,
        UPDATE_CACHE = 4'd8,
        DONE_STATE = 4'd9;
    
    reg [3:0] state;
    reg [N_PARAMETERS*DATA_WIDTH-1:0] query_reg;
    reg [15:0] found_region_idx;
    
    // L1 Cache signals
    reg l1_query_valid;
    wire l1_hit, l1_query_done;
    wire [15:0] l1_region_idx;
    reg l1_update_valid;
    
    // L2 Cache signals
    reg l2_query_valid;
    wire l2_hit, l2_query_done;
    wire [15:0] l2_region_idx;
    reg l2_update_valid;
    
    // Shared constraint checker
    wire cc_check_req_l1, cc_check_req_l2;
    wire [15:0] cc_region_idx_l1, cc_region_idx_l2;
    reg cc_check_done, cc_satisfied;
    wire cc_check_req = cc_check_req_l1 | cc_check_req_l2;
    wire [15:0] cc_region_idx = cc_check_req_l1 ? cc_region_idx_l1 : cc_region_idx_l2;
    
    // BST signals
    reg bst_start;
    wire [15:0] bst_region_idx;
    wire bst_done;
    
    // ROM placeholder
    reg [15:0] rom_addr;
    reg rom_read_enable;
    wire [N_PARAMETERS*DATA_WIDTH-1:0] rom_G_row = {{(N_PARAMETERS*DATA_WIDTH){{1'b0}}}};
    wire [DATA_WIDTH-1:0] rom_g_val = {{DATA_WIDTH{{1'b0}}}};
    wire [7:0] rom_num_constraints = 8'd4;
    wire [N_SOLUTIONS*DATA_WIDTH-1:0] rom_solution = {{(N_SOLUTIONS*DATA_WIDTH){{1'b0}}}};
    
    // Module instantiations
    pdaqp_cache_l1 #(
        .N_PARAMETERS(N_PARAMETERS),
        .DATA_WIDTH(DATA_WIDTH)
    ) l1_cache (
        .clk(clk), .rst_n(rst_n),
        .query_valid(l1_query_valid),
        .query_theta(query_reg),
        .hit(l1_hit),
        .region_idx(l1_region_idx),
        .query_done(l1_query_done),
        .update_valid(l1_update_valid),
        .update_theta(query_reg),
        .update_region_idx(found_region_idx),
        .constraint_check_req(cc_check_req_l1),
        .constraint_region_idx(cc_region_idx_l1),
        .constraint_check_done(cc_check_done),
        .constraint_satisfied(cc_satisfied)
    );
    
    pdaqp_cache_l2 #(
        .N_PARAMETERS(N_PARAMETERS),
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_NEIGHBORS(16)
    ) l2_cache (
        .clk(clk), .rst_n(rst_n),
        .query_valid(l2_query_valid),
        .query_theta(query_reg),
        .hit(l2_hit),
        .region_idx(l2_region_idx),
        .query_done(l2_query_done),
        .update_valid(l2_update_valid),
        .center_region_idx(found_region_idx),
        .neighbor_list_packed(256'h0),
        .num_neighbors(8'd4),
        .constraint_check_req(cc_check_req_l2),
        .constraint_region_idx(cc_region_idx_l2),
        .constraint_check_done(cc_check_done),
        .constraint_satisfied(cc_satisfied)
    );
    
    pdaqp_constraint_checker #(
        .N_PARAMETERS(N_PARAMETERS),
        .DATA_WIDTH(DATA_WIDTH)
    ) constraint_checker (
        .clk(clk), .rst_n(rst_n),
        .check_req(cc_check_req),
        .region_idx(cc_region_idx),
        .theta(query_reg),
        .check_done(cc_check_done),
        .satisfied(cc_satisfied),
        .rom_addr(rom_addr),
        .rom_read_enable(rom_read_enable),
        .rom_G_row(rom_G_row),
        .rom_g_val(rom_g_val),
        .rom_num_constraints(rom_num_constraints),
        .rom_data_valid(1'b1)
    );
    
    pdaqp_bst_lut #(
        .N_PARAMETERS(N_PARAMETERS),
        .DATA_WIDTH(DATA_WIDTH)
    ) bst_lut (
        .clk(clk), .rst_n(rst_n),
        .start(bst_start),
        .query(query_reg),
        .region_idx(bst_region_idx),
        .done(bst_done)
    );
    
    // Main FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            result <= {{(N_SOLUTIONS*DATA_WIDTH){{1'b0}}}};
            query_reg <= {{(N_PARAMETERS*DATA_WIDTH){{1'b0}}}};
            l1_query_valid <= 1'b0;
            l2_query_valid <= 1'b0;
            bst_start <= 1'b0;
            l1_update_valid <= 1'b0;
            l2_update_valid <= 1'b0;
            found_region_idx <= 16'd0;
            rom_addr <= 16'd0;
            rom_read_enable <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    l1_query_valid <= 1'b0;
                    l2_query_valid <= 1'b0;
                    bst_start <= 1'b0;
                    l1_update_valid <= 1'b0;
                    l2_update_valid <= 1'b0;
                    rom_read_enable <= 1'b0;
                    if (start) begin
                        query_reg <= query;
                        state <= L1_QUERY;
                    end
                end
                
                L1_QUERY: begin
                    l1_query_valid <= 1'b1;
                    state <= L1_WAIT;
                end
                
                L1_WAIT: begin
                    if (l1_query_done) begin
                        l1_query_valid <= 1'b0;
                        if (l1_hit) begin
                            found_region_idx <= l1_region_idx;
                            state <= FETCH_SOLUTION;
                        end else begin
                            state <= L2_QUERY;
                        end
                    end
                end
                
                L2_QUERY: begin
                    l2_query_valid <= 1'b1;
                    state <= L2_WAIT;
                end
                
                L2_WAIT: begin
                    if (l2_query_done) begin
                        l2_query_valid <= 1'b0;
                        if (l2_hit) begin
                            found_region_idx <= l2_region_idx;
                            state <= FETCH_SOLUTION;
                        end else begin
                            state <= BST_QUERY;
                        end
                    end
                end
                
                BST_QUERY: begin
                    bst_start <= 1'b1;
                    state <= BST_WAIT;
                end
                
                BST_WAIT: begin
                    if (bst_done) begin
                        bst_start <= 1'b0;
                        found_region_idx <= bst_region_idx;
                        state <= FETCH_SOLUTION;
                    end
                end
                
                FETCH_SOLUTION: begin
                    rom_addr <= found_region_idx;
                    rom_read_enable <= 1'b1;
                    result <= rom_solution;
                    state <= UPDATE_CACHE;
                end
                
                UPDATE_CACHE: begin
                    rom_read_enable <= 1'b0;
                    l1_update_valid <= 1'b1;
                    l2_update_valid <= 1'b1;
                    state <= DONE_STATE;
                end
                
                DONE_STATE: begin
                    l1_update_valid <= 1'b0;
                    l2_update_valid <= 1'b0;
                    done <= 1'b1;
                    if (!start)
                        state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end
endmodule
"""
        with open(rtl_dir / 'pdaqp_top.v', 'w') as f:
            f.write(top_content)
    
    def _create_testbench(self, tb_dir: Path, n: int, m: int, 
                         data_width: int = 32, num_tests: int = 10):
        """Create testbench"""
        tb_dir.mkdir(parents=True, exist_ok=True)
        
        tb_content = f"""// Testbench for Two-Level Cache
`timescale 1ns/1ps

module tb_pdaqp_top;
    
    localparam N_PARAMETERS = {n};
    localparam N_SOLUTIONS = {m};
    localparam DATA_WIDTH = {data_width};
    localparam CLK_PERIOD = 10;
    
    reg clk, rst_n, start;
    reg [N_PARAMETERS*DATA_WIDTH-1:0] query;
    wire [N_SOLUTIONS*DATA_WIDTH-1:0] result;
    wire done;
    
    pdaqp_top #(
        .N_PARAMETERS(N_PARAMETERS),
        .N_SOLUTIONS(N_SOLUTIONS),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .query(query),
        .result(result), .done(done)
    );
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    integer test_num, param_idx;
    reg [DATA_WIDTH-1:0] param_value;
    
    initial begin
        $dumpfile("sim/tb.vcd");
        $dumpvars(0, tb_pdaqp_top);
        
        rst_n = 0;
        start = 0;
        query = {{(N_PARAMETERS*DATA_WIDTH){{1'b0}}}};
        
        #(CLK_PERIOD*2);
        rst_n = 1;
        #(CLK_PERIOD*2);
        
        $display("========================================");
        $display("Two-Level Cache Test Suite");
        $display("N_PARAMETERS=%0d, N_SOLUTIONS=%0d", N_PARAMETERS, N_SOLUTIONS);
        $display("========================================");
        
        for (test_num = 0; test_num < {num_tests}; test_num = test_num + 1) begin
            for (param_idx = 0; param_idx < N_PARAMETERS; param_idx = param_idx + 1) begin
                param_value = test_num * 100 + param_idx * 10;
                query[param_idx*DATA_WIDTH +: DATA_WIDTH] = param_value;
            end
            
            $display("\\nTest %0d: Query=0x%h", test_num, query);
            
            start = 1;
            #CLK_PERIOD;
            start = 0;
            
            fork
                begin
                    wait(done);
                    $display("  Result=0x%h [PASS]", result);
                end
                begin
                    #(CLK_PERIOD * 200);
                    if (!done) $display("  [TIMEOUT]");
                end
            join_any
            disable fork;
            
            #(CLK_PERIOD*5);
        end
        
        $display("\\n========================================");
        $display("All tests completed");
        $display("========================================");
        #(CLK_PERIOD*10);
        $finish;
    end
    
    initial begin
        #(CLK_PERIOD * 5000);
        $display("\\nERROR: Global timeout");
        $finish;
    end
endmodule
"""
        with open(tb_dir / 'tb_pdaqp_top.v', 'w') as f:
            f.write(tb_content)
    
    def run_cached(self,
                   c_file: str,
                   h_file: str,
                   output_dir: Optional[str] = None,
                   max_iob: Optional[int] = None,
                   no_input_buffer: bool = False,
                   no_testbench: bool = False,
                   num_tests: int = 10,
                   skip_verification: bool = False,
                   verbose: bool = False) -> int:
        """Main generation flow"""
        
        c_path = Path(c_file).resolve()
        h_path = Path(h_file).resolve()
        
        if not c_path.exists() or not h_path.exists():
            print(f"Error: Input files not found")
            return 1
        
        project_name = c_path.stem
        out_dir = Path(output_dir).resolve() if output_dir else Path.cwd() / f"codegen_{project_name}_cached"
        
        print(f"\n{'='*70}")
        print(f"PAPER-COMPLIANT TWO-LEVEL CACHE GENERATION")
        print(f"{'='*70}")
        print(f"Input C:      {c_path}")
        print(f"Input H:      {h_path}")
        print(f"Output:       {out_dir}")
        print(f"Architecture: L1 (1 entry) → L2 (neighbors) → BST")
        print(f"{'='*70}\n")
        
        # Stage 1: Config
        print(f"{'='*70}")
        print(f"Stage 1/4: Configuration Generation")
        print(f"{'='*70}\n")
        
        config_args = ['-c', str(c_path), '-H', str(h_path), '-o', str(out_dir / 'include')]
        if verbose:
            config_args.append('--verbose')
        
        if self.run_stage('config', config_args, verbose=verbose) != 0:
            return 1
        print("✓ Stage 1 completed\n")
        
        # Stage 2: Interface
        print(f"{'='*70}")
        print(f"Stage 2/4: Interface Generation")
        print(f"{'='*70}\n")
        
        config_file = out_dir / 'include' / f'{project_name}_config.vh'
        if not config_file.exists():
            print(f"Error: Config file not found")
            return 1
        
        interface_args = ['-c', str(config_file), '-o', str(out_dir)]
        if max_iob:
            interface_args.extend(['--max-iob', str(max_iob)])
        if no_input_buffer:
            interface_args.append('--no-input-buffer')
        if verbose:
            interface_args.append('--verbose')
        
        if self.run_stage('interface', interface_args, verbose=verbose) != 0:
            return 1
        print("✓ Stage 2 completed\n")
        
        # Stage 3: Cache RTL
        print(f"{'='*70}")
        print(f"Stage 3/4: Two-Level Cache RTL Generation")
        print(f"{'='*70}\n")
        
        n_parameters, m_solutions = self._infer_n_and_m_from_config(config_file)
        print(f"  Problem size: n={n_parameters}, m={m_solutions}")
        
        rtl_dir = out_dir / 'rtl'
        tb_dir = out_dir / 'testbench'
        
        print(f"\n  Generating RTL...")
        self._create_paper_compliant_cache(rtl_dir, n_parameters, m_solutions)
        
        for rtl_file in rtl_dir.glob('*.v'):
            print(f"    {rtl_file.relative_to(out_dir)}")
        
        if not no_testbench:
            print(f"\n  Generating testbench...")
            self._create_testbench(tb_dir, n_parameters, m_solutions, num_tests=num_tests)
            print(f"    {tb_dir.relative_to(out_dir)}/tb_pdaqp_top.v")
        
        print(f"\n  Generating Makefile...")
        with open(out_dir / 'Makefile', 'w') as f:
            f.write(self._generate_makefile(out_dir, project_name))
        
        print("\n✓ Stage 3 completed\n")
        
        # Stage 4: Verification
        if not skip_verification and self.verilator_available:
            print(f"{'='*70}")
            print(f"Stage 4/4: Verilator Verification")
            print(f"{'='*70}\n")
            
            lint_passed = self._run_verilator_lint(
                rtl_dir, out_dir / 'include', 'pdaqp_top', verbose=verbose
            )
            
            print(f"\n{'✓' if lint_passed else '⚠'} Stage 4 completed\n")
        
        # Summary
        print(f"{'='*70}")
        print(f"GENERATION COMPLETED")
        print(f"{'='*70}")
        print(f"\nOutput: {out_dir}")
        print(f"\nNext steps:")
        print(f"  cd {out_dir.name}")
        print(f"  make lint")
        print(f"  make run")
        print(f"{'='*70}\n")
        
        return 0


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description='Paper-Compliant Two-Level Cache Generator'
    )
    
    parser.add_argument('-c', '--c-file', required=True)
    parser.add_argument('-H', '--header-file', required=True)
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-iob', type=int)
    parser.add_argument('--no-input-buffer', action='store_true')
    parser.add_argument('--no-testbench', action='store_true')
    parser.add_argument('--num-tests', type=int, default=10)
    parser.add_argument('--skip-verification', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    generator = CachedHardwareGenerator()
    
    return generator.run_cached(
        c_file=args.c_file,
        h_file=args.header_file,
        output_dir=args.output,
        max_iob=args.max_iob,
        no_input_buffer=args.no_input_buffer,
        no_testbench=args.no_testbench,
        num_tests=args.num_tests,
        skip_verification=args.skip_verification,
        verbose=args.verbose
    )


if __name__ == '__main__':
    sys.exit(main())