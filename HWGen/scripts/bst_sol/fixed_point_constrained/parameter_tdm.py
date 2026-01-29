# fixed_point_constrained/parameter_tdm.py

"""
Strategy 3: Parameter-Level Time-Division Multiplexing
Partitions p parameters into B = ceil(p/D) batches
"""

from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path
import math


@dataclass
class ParameterTDMConfig:
    """Parameter-Level TDM architecture configuration"""
    n: int
    p: int
    dsp_count: int
    batch_count: int
    params_per_batch: int
    cycles_per_solution: int
    total_cycles: int
    
    def __post_init__(self):
        assert 0 < self.dsp_count < self.p, \
            f"Invalid DSP count {self.dsp_count} for Parameter-TDM"


def configure_parameter_tdm(n: int, p: int, dsp_count: int) -> ParameterTDMConfig:
    """Configure Parameter-Level Time-Division Multiplexing"""
    assert n > 0 and p > 0, "n and p must be positive"
    assert 0 < dsp_count < p, f"Invalid DSP count (need 0 < D < {p})"
    
    B = math.ceil(p / dsp_count)
    return ParameterTDMConfig(
        n=n, p=p, dsp_count=dsp_count, batch_count=B,
        params_per_batch=dsp_count,
        cycles_per_solution=3 * B,
        total_cycles=3 * B * n
    )


class ParameterTDMScheduler:
    """Scheduler for Parameter-Level Time-Division"""
    def __init__(self, config: ParameterTDMConfig):
        self.config = config
    
    def get_parameter_range(self, batch_idx: int) -> Tuple[int, int]:
        D = self.config.dsp_count
        start = batch_idx * D
        end = min(start + D, self.config.p)
        return start, end


class ParameterTDMGenerator:
    """RTL Generator for Parameter-Level TDM"""
    
    def __init__(self, config: ParameterTDMConfig, verilog_config):
        self.config = config
        self.vcfg = verilog_config
        self.scheduler = ParameterTDMScheduler(config)
        
        # Extract data widths from verilog_config (PDQAPConfig)
        self.data_width = self.vcfg.data_width
        
        # For fixed-point: use halfplane format for F matrix, feedback format for theta
        if self.vcfg.data_format.startswith('fixed'):
            self.f_width = self.data_width  # F matrix uses standard data width
            self.theta_width = self.data_width  # theta uses standard data width
            self.theta_frac = self.vcfg.feedback_frac_bits or 8
            self.output_frac = self.vcfg.output_frac_bits or 8
        else:
            # Floating-point
            self.f_width = self.data_width
            self.theta_width = self.data_width
            self.theta_frac = 0
            self.output_frac = 0
    
    def generate_rtl(self, output_dir: Path) -> bool:
        """Generate complete RTL for Parameter-TDM solver"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            modules = {
                "bst_solver_parameter_tdm.v": self._generate_solver_module(),
                "parameter_tdm_control.v": self._generate_control_fsm(),
                "parameter_tdm_datapath.v": self._generate_datapath()
            }
            
            for filename, content in modules.items():
                with open(output_dir / filename, 'w') as f:
                    f.write(content)
            
            return True
        except Exception as e:
            print(f"Error generating RTL: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_solver_module(self) -> str:
        """Generate top-level solver module"""
        n, p, D, B = self.config.n, self.config.p, self.config.dsp_count, self.config.batch_count
        
        return f'''// Parameter-Level TDM BST Solver
// n={n}, p={p}, D={D}, B={B}

module bst_solver_parameter_tdm #(
    parameter DATA_WIDTH = {self.data_width},
    parameter THETA_WIDTH = {self.theta_width},
    parameter N_SOLUTIONS = {n},
    parameter N_PARAMETERS = {p},
    parameter DSP_COUNT = {D},
    parameter BATCH_COUNT = {B}
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    
    input wire [THETA_WIDTH-1:0] theta [0:N_PARAMETERS-1],
    
    output reg [DATA_WIDTH-1:0] x_out [0:N_SOLUTIONS-1],
    output reg valid,
    output reg done
);

    wire [31:0] solution_idx, batch_idx, param_start, param_end;
    wire load_enable, multiply_enable, accumulate_enable;
    wire add_offset_enable, clear_accumulator;
    
    parameter_tdm_datapath #(
        .DATA_WIDTH(DATA_WIDTH),
        .THETA_WIDTH(THETA_WIDTH),
        .N_SOLUTIONS(N_SOLUTIONS),
        .N_PARAMETERS(N_PARAMETERS),
        .DSP_COUNT(DSP_COUNT)
    ) datapath (
        .clk(clk), .rst_n(rst_n), .theta(theta),
        .solution_idx(solution_idx), .batch_idx(batch_idx),
        .param_start(param_start), .param_end(param_end),
        .load_enable(load_enable), .multiply_enable(multiply_enable),
        .accumulate_enable(accumulate_enable), .add_offset_enable(add_offset_enable),
        .clear_accumulator(clear_accumulator), .x_out(x_out)
    );
    
    parameter_tdm_control #(
        .N_SOLUTIONS(N_SOLUTIONS), .N_PARAMETERS(N_PARAMETERS),
        .DSP_COUNT(DSP_COUNT), .BATCH_COUNT(BATCH_COUNT)
    ) control (
        .clk(clk), .rst_n(rst_n), .start(start),
        .solution_idx(solution_idx), .batch_idx(batch_idx),
        .param_start(param_start), .param_end(param_end),
        .load_enable(load_enable), .multiply_enable(multiply_enable),
        .accumulate_enable(accumulate_enable), .add_offset_enable(add_offset_enable),
        .clear_accumulator(clear_accumulator), .valid(valid), .done(done)
    );

endmodule
'''
    
    def _generate_control_fsm(self) -> str:
        """Generate control FSM module"""
        n, p, D, B = self.config.n, self.config.p, self.config.dsp_count, self.config.batch_count
        
        return f'''// Parameter-Level TDM Control FSM

module parameter_tdm_control #(
    parameter N_SOLUTIONS = {n},
    parameter N_PARAMETERS = {p},
    parameter DSP_COUNT = {D},
    parameter BATCH_COUNT = {B}
) (
    input wire clk, rst_n, start,
    output reg [31:0] solution_idx, batch_idx, param_start, param_end,
    output reg load_enable, multiply_enable, accumulate_enable,
    output reg add_offset_enable, clear_accumulator,
    output reg valid, done
);

    localparam IDLE = 3'd0, LOAD = 3'd1, MULTIPLY = 3'd2, ACCUMULATE = 3'd3, DONE = 3'd4;
    
    reg [2:0] state, next_state;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            solution_idx <= 0;
            batch_idx <= 0;
        end else begin
            state <= next_state;
            
            if (state == IDLE && start) begin
                solution_idx <= 0;
                batch_idx <= 0;
            end else if (state == ACCUMULATE) begin
                if (batch_idx < BATCH_COUNT - 1) begin
                    batch_idx <= batch_idx + 1;
                end else begin
                    batch_idx <= 0;
                    if (solution_idx < N_SOLUTIONS - 1)
                        solution_idx <= solution_idx + 1;
                end
            end
        end
    end
    
    always @(*) begin
        case (state)
            IDLE: next_state = start ? LOAD : IDLE;
            LOAD: next_state = MULTIPLY;
            MULTIPLY: next_state = ACCUMULATE;
            ACCUMULATE: next_state = (batch_idx == BATCH_COUNT-1 && solution_idx == N_SOLUTIONS-1) ? DONE : LOAD;
            DONE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
    
    always @(*) begin
        param_start = batch_idx * DSP_COUNT;
        param_end = (param_start + DSP_COUNT > N_PARAMETERS) ? N_PARAMETERS : param_start + DSP_COUNT;
        
        load_enable = (state == LOAD);
        multiply_enable = (state == MULTIPLY);
        accumulate_enable = (state == ACCUMULATE);
        add_offset_enable = (state == ACCUMULATE) && (batch_idx == BATCH_COUNT - 1);
        clear_accumulator = (state == LOAD) && (batch_idx == 0);
        valid = (state == ACCUMULATE) && (batch_idx == BATCH_COUNT - 1);
        done = (state == DONE);
    end

endmodule
'''
    
    def _generate_datapath(self) -> str:
        """Generate datapath module"""
        D = self.config.dsp_count
        
        # Generate DSP product summation based on DSP_COUNT
        products = " + ".join([f"product[{i}]" for i in range(min(D, 8))])
        
        return f'''// Parameter-Level TDM Datapath

module parameter_tdm_datapath #(
    parameter DATA_WIDTH = {self.data_width},
    parameter THETA_WIDTH = {self.theta_width},
    parameter N_SOLUTIONS = {self.config.n},
    parameter N_PARAMETERS = {self.config.p},
    parameter DSP_COUNT = {D}
) (
    input wire clk, rst_n,
    input wire [THETA_WIDTH-1:0] theta [0:N_PARAMETERS-1],
    input wire [31:0] solution_idx, batch_idx, param_start, param_end,
    input wire load_enable, multiply_enable, accumulate_enable,
    input wire add_offset_enable, clear_accumulator,
    output reg [DATA_WIDTH-1:0] x_out [0:N_SOLUTIONS-1]
);

    reg [DATA_WIDTH-1:0] F_matrix [0:N_SOLUTIONS-1][0:N_PARAMETERS-1];
    reg [DATA_WIDTH-1:0] f_offset [0:N_SOLUTIONS-1];
    
    reg [DATA_WIDTH-1:0] f_reg [0:DSP_COUNT-1];
    reg [THETA_WIDTH-1:0] theta_reg [0:DSP_COUNT-1];
    reg [DATA_WIDTH-1:0] product [0:DSP_COUNT-1];
    reg [DATA_WIDTH-1:0] accumulator;
    
    integer i;
    
    always @(posedge clk) begin
        if (load_enable) begin
            for (i = 0; i < DSP_COUNT; i = i + 1) begin
                if (param_start + i < param_end) begin
                    f_reg[i] <= F_matrix[solution_idx][param_start + i];
                    theta_reg[i] <= theta[param_start + i];
                end else begin
                    f_reg[i] <= 0;
                    theta_reg[i] <= 0;
                end
            end
        end
    end
    
    always @(posedge clk) begin
        if (multiply_enable) begin
            for (i = 0; i < DSP_COUNT; i = i + 1)
                product[i] <= f_reg[i] * theta_reg[i];
        end
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
        end else if (clear_accumulator) begin
            accumulator <= 0;
        end else if (accumulate_enable) begin
            accumulator <= accumulator + {products};
            if (add_offset_enable)
                x_out[solution_idx] <= accumulator + f_offset[solution_idx];
        end
    end

endmodule
'''


def print_parameter_tdm_report(config: ParameterTDMConfig) -> None:
    """Print configuration report"""
    print("="*60)
    print("Parameter-Level TDM Configuration")
    print("="*60)
    print(f"Problem: n={config.n}, p={config.p}")
    print(f"DSP: {config.dsp_count}, Batches: {config.batch_count}")
    print(f"Cycles/solution: {config.cycles_per_solution}, Total: {config.total_cycles}")
    macs = config.n * config.p
    eff = macs / (config.dsp_count * config.total_cycles)
    print(f"DSP Efficiency: {eff:.2%}")
    print("="*60)


def generate_parameter_tdm_solver(config_file: Path, output_dir: Path, 
                                  dsp_count: int = None, verbose: bool = False) -> int:
    """Generate Parameter-Level TDM BST solver hardware"""
    try:
        import sys
        SCRIPT_DIR = Path(__file__).parent
        PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        
        from scripts.bst_sol.common.config_parser import VerilogConfigParser
        
        parser = VerilogConfigParser(str(config_file))
        cfg = parser.get_config()
        
        n, p = cfg.n_solutions, cfg.n_parameters
        
        if verbose:
            print(f"\nParameter-Level TDM Generation")
            print(f"Config: {config_file}, n={n}, p={p}")
        
        if dsp_count is None:
            dsp_count = max(1, min(p - 1, int(math.sqrt(p))))
        
        if not (0 < dsp_count < p):
            print(f"❌ Invalid DSP count {dsp_count} (need 0 < D < {p})")
            return 1
        
        tdm_config = configure_parameter_tdm(n, p, dsp_count)
        
        if verbose:
            print_parameter_tdm_report(tdm_config)
        
        generator = ParameterTDMGenerator(tdm_config, cfg)
        if generator.generate_rtl(output_dir):
            print(f"✓ Parameter-TDM RTL generated: {output_dir}")
            print(f"  Cycles: {tdm_config.total_cycles}, Batches: {tdm_config.batch_count}")
            return 0
        else:
            print(f"❌ RTL generation failed")
            return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parameter-Level TDM Generator")
    parser.add_argument('config_file', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--dsp-count', type=int, default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exit(generate_parameter_tdm_solver(args.config_file, args.output_dir, 
                                       args.dsp_count, args.verbose))