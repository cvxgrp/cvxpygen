"""
polyhedral_checker.py
=====================
Polyhedral Constraint Checker for Cached BST Solver

Verifies if a candidate solution satisfies polyhedral constraints:
    A·x ≤ b
    
where:
    - A is constraint matrix (p×m)
    - x is candidate solution vector (m×1)
    - b is constraint bound vector (p×1)
    - p is number of constraints
    - m is number of parameters

Features:
- Parallel constraint evaluation
- Fixed-point arithmetic (format from config.vh)
- Pipelined matrix-vector multiplication
- Efficient constraint satisfaction checking
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import math


class PolyhedralCheckerGenerator:
    """Generate polyhedral constraint checker module"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize polyhedral checker generator
        
        Args:
            config: Solver configuration containing:
                - n_parameters: Number of parameters (m)
                - n_constraints: Number of constraints (p)
                - int_bits: Integer bits (from config.vh)
                - frac_bits: Fractional bits (from config.vh)
                - data_width: Total width
                - project_name: Design name
                - constraint_matrix: Optional A matrix
                - constraint_bounds: Optional b vector
        """
        self.config = config
        self.m = config['n_parameters']
        self.p = config['n_constraints']
        
        # Read Q format from config
        self.int_bits = config.get('int_bits')
        self.frac_bits = config.get('frac_bits')
        
        if self.int_bits is None or self.frac_bits is None:
            raise ValueError("int_bits and frac_bits must be specified in config")
        
        # Total width = sign bit + integer bits + fractional bits
        self.data_width = 1 + self.int_bits + self.frac_bits
        
        self.project_name = config.get('project_name', 'solver')
        
        # Pipeline configuration
        self.pipeline_stages = self._calculate_pipeline_stages()
        
        # Internal precision for matrix-vector products
        # Product: Q_int.frac * Q_int.frac = Q_(2*int+1).(2*frac)
        self.product_width = 2 * self.data_width
        self.product_int_bits = 2 * self.int_bits + 1
        self.product_frac_bits = 2 * self.frac_bits
        
        # Accumulator for dot product (A[i]·x)
        # Needs extra bits for sum of m products
        self.accumulator_width = self.product_width + self._bits_needed(self.m)
        
        # Constraint matrices (if provided)
        self.A_matrix = config.get('constraint_matrix', None)
        self.b_vector = config.get('constraint_bounds', None)
        
        # Parallelism configuration
        self.parallel_constraints = self._calculate_parallel_constraints()
    
    def _bits_needed(self, n: int) -> int:
        """Calculate bits needed to represent sum of n values"""
        if n <= 1:
            return 0
        return (n - 1).bit_length()
    
    def _calculate_pipeline_stages(self) -> int:
        """Determine optimal pipeline stages"""
        # Stage breakdown:
        # 1. Multiply (A[i][j] * x[j])
        # 2. Accumulate (sum products)
        # 3. Compare (A·x ≤ b)
        # 4. Aggregate (all constraints satisfied)
        
        if self.m <= 4:
            return 3  # multiply + accumulate + compare
        else:
            return 4  # multiply + partial_sum + accumulate + compare
    
    def _calculate_parallel_constraints(self) -> int:
        """Calculate how many constraints to check in parallel"""
        # Trade-off between parallelism and resource usage
        if self.p <= 4:
            return self.p  # Check all in parallel
        elif self.p <= 16:
            return 4       # Check 4 at a time
        else:
            return 8       # Check 8 at a time for large problems
    
    def generate(self, output_dir: str) -> str:
        """
        Generate polyhedral checker Verilog module
        
        Returns:
            Path to generated file
        """
        output_path = Path(output_dir) / f'{self.project_name}_polyhedral_checker.v'
        
        verilog = self._generate_module()
        
        with open(output_path, 'w') as f:
            f.write(verilog)
        
        return str(output_path)
    
    def _generate_module(self) -> str:
        """Generate complete Verilog module"""
        return f"""{self._file_header()}

`include "config.vh"

module {self.project_name}_polyhedral_checker #(
    parameter M = {self.m},                    // Number of parameters
    parameter P = {self.p},                    // Number of constraints
    parameter PARALLEL = {self.parallel_constraints}  // Constraints checked in parallel
) (
    input  wire clk,
    input  wire rst_n,
    
    // Control
    input  wire start,              // Start constraint checking
    output reg  valid,              // Result valid
    output reg  busy,               // Checking in progress
    
    // Candidate solution input
    input  wire [`DATA_WIDTH-1:0] solution [0:M-1],
    
    // Constraint matrix A (p×m) - can be hardcoded or from memory
    input  wire [`DATA_WIDTH-1:0] A_matrix [0:P-1][0:M-1],
    
    // Constraint bounds b (p×1)
    input  wire [`DATA_WIDTH-1:0] b_vector [0:P-1],
    
    // Constraint satisfaction output
    output reg  feasible,           // 1 if all constraints satisfied
    output reg  [P-1:0] constraint_status  // Per-constraint status (for debugging)
);

{self._generate_internal_signals()}

{self._generate_control_logic()}

{self._generate_datapath()}

endmodule
"""
    
    def _file_header(self) -> str:
        """Generate file header"""
        q_format = f"Q{self.int_bits}.{self.frac_bits}"
        q_product_format = f"Q{self.product_int_bits}.{self.product_frac_bits}"
        
        return f"""//==============================================================================
// Polyhedral Constraint Checker - Fixed-Point Implementation
//==============================================================================
// Auto-generated by CachedBSTSolver
//
// Verifies polyhedral constraints: A·x ≤ b
//
// Configuration:
//   Parameters (m):       {self.m}
//   Constraints (p):      {self.p}
//   Parallel checks:      {self.parallel_constraints}
//   Input format:         {q_format} ({self.data_width} bits)
//   Product format:       {q_product_format} ({self.product_width} bits)
//   Accumulator width:    {self.accumulator_width} bits
//   Pipeline stages:      {self.pipeline_stages}
//
// Latency: {self.get_latency()} cycles
//
// Operation:
//   For each constraint i:
//     1. Compute dot product: sum_j(A[i][j] * x[j])
//     2. Compare: dot_product ≤ b[i]
//     3. Aggregate: feasible = AND(all constraints)
//
// Note: Comparison is done in Q{self.product_int_bits}.{self.product_frac_bits} format
//       Both sides are aligned before comparison
//==============================================================================
"""
    
    def _generate_internal_signals(self) -> str:
        """Generate internal signal declarations"""
        
        signals = f"""    // Configuration from config.vh
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam INT_BITS = `INT_BITS;
    localparam FRAC_BITS = `FRAC_BITS;
    
    // Derived parameters
    localparam PRODUCT_WIDTH = 2 * DATA_WIDTH;
    localparam ACCUM_WIDTH = {self.accumulator_width};
    localparam PIPE_DEPTH = {self.pipeline_stages};
    
    // Number of iterations needed if not all constraints checked in parallel
    localparam NUM_ITERATIONS = (P + PARALLEL - 1) / PARALLEL;
    
    // Pipeline stage signals
    reg  [PRODUCT_WIDTH-1:0] products [0:PARALLEL-1][0:M-1];   // Stage 1: A[i][j]*x[j]
    reg  [ACCUM_WIDTH-1:0] dot_products [0:PARALLEL-1];        // Stage 2: sum(products)
    reg  [ACCUM_WIDTH-1:0] b_extended [0:PARALLEL-1];          // Stage 2: b extended
    reg  [PARALLEL-1:0] constraint_satisfied;                  // Stage 3: comparisons
    
    // Iteration control
    reg  [$clog2(NUM_ITERATIONS+1)-1:0] iteration_cnt;
    reg  [P-1:0] constraint_results;  // Accumulated constraint results
    
    // Pipeline valid tracking
    reg  [PIPE_DEPTH:0] valid_pipe;
    
    // State machine
    typedef enum logic [1:0] {{
        IDLE = 2'b00,
        COMPUTING = 2'b01,
        AGGREGATING = 2'b10,
        DONE = 2'b11
    }} state_t;
    
    state_t state, next_state;
    
    // Current constraint batch being processed
    reg  [$clog2(P)-1:0] constraint_base;
"""
        
        return signals
    
    def _generate_control_logic(self) -> str:
        """Generate FSM control logic"""
        
        return f"""    //==========================================================================
    // Control Logic - State Machine for Constraint Checking
    //==========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            iteration_cnt <= '0;
            constraint_base <= '0;
            busy <= 1'b0;
            valid <= 1'b0;
            feasible <= 1'b0;
            constraint_status <= '0;
            constraint_results <= '1;  // Initially assume all satisfied
            valid_pipe <= '0;
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        busy <= 1'b1;
                        valid <= 1'b0;
                        iteration_cnt <= '0;
                        constraint_base <= '0;
                        constraint_results <= '1;  // Reset to all satisfied
                        valid_pipe <= {{{{1'b1, {{PIPE_DEPTH{{1'b0}}}}}}}};
                    end
                end
                
                COMPUTING: begin
                    // Pipeline advancement
                    valid_pipe <= {{valid_pipe[PIPE_DEPTH-1:0], 1'b0}};
                    
                    // Check if current batch completed
                    if (valid_pipe[PIPE_DEPTH]) begin
                        // Store results for current batch
                        for (integer i = 0; i < PARALLEL; i = i + 1) begin
                            if (constraint_base + i < P) begin
                                constraint_results[constraint_base + i] <= constraint_satisfied[i];
                            end
                        end
                        
                        // Move to next batch
                        constraint_base <= constraint_base + PARALLEL;
                        iteration_cnt <= iteration_cnt + 1;
                        
                        // Check if all batches processed
                        if (iteration_cnt + 1 >= NUM_ITERATIONS) begin
                            // All constraints checked, move to aggregation
                            valid_pipe <= '0;
                        end else begin
                            // Start next batch
                            valid_pipe <= {{{{1'b1, {{PIPE_DEPTH{{1'b0}}}}}}}};
                        end
                    end
                end
                
                AGGREGATING: begin
                    // Aggregate all constraint results
                    feasible <= &constraint_results;  // AND of all bits
                    constraint_status <= constraint_results;
                    valid <= 1'b1;
                end
                
                DONE: begin
                    busy <= 1'b0;
                    valid <= 1'b0;
                end
            endcase
        end
    end
    
    // Next state logic
    always @(*) begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start) begin
                    next_state = COMPUTING;
                end
            end
            
            COMPUTING: begin
                if (valid_pipe[PIPE_DEPTH] && (iteration_cnt + 1 >= NUM_ITERATIONS)) begin
                    next_state = AGGREGATING;
                end
            end
            
            AGGREGATING: begin
                next_state = DONE;
            end
            
            DONE: begin
                next_state = IDLE;
            end
        endcase
    end
"""
    
    def _generate_datapath(self) -> str:
        """Generate computational datapath"""
        
        stage1 = self._generate_stage1_multiply()
        stage2 = self._generate_stage2_accumulate()
        stage3 = self._generate_stage3_compare()
        
        return f"""    //==========================================================================
    // Datapath - Pipelined Constraint Evaluation
    //==========================================================================
    
{stage1}

{stage2}

{stage3}
"""
    
    def _generate_stage1_multiply(self) -> str:
        """Generate Stage 1: Matrix-vector multiplication"""
        
        code = f"""    // Stage 1: Multiply A[i][j] * x[j]
    //--------------------------------------------------------------------------
    // Compute products for current constraint batch
    // Format: Q_int.frac * Q_int.frac = Q_(2*int+1).(2*frac)
    
    always @(posedge clk) begin
        if (state == COMPUTING && (start || valid_pipe[0])) begin
            for (integer i = 0; i < PARALLEL; i = i + 1) begin
                if (constraint_base + i < P) begin
                    for (integer j = 0; j < M; j = j + 1) begin
                        // Signed multiplication
                        products[i][j] <= $signed(A_matrix[constraint_base + i][j]) * 
                                         $signed(solution[j]);
                    end
                end else begin
                    // Padding for partial batches
                    for (integer j = 0; j < M; j = j + 1) begin
                        products[i][j] <= '0;
                    end
                end
            end
        end
    end
"""
        return code
    
    def _generate_stage2_accumulate(self) -> str:
        """Generate Stage 2: Accumulate dot products"""
        
        if self.m <= 4:
            accumulate_code = self._generate_direct_accumulate()
        else:
            accumulate_code = self._generate_tree_accumulate()
        
        code = f"""    // Stage 2: Accumulate dot products (sum_j A[i][j]*x[j])
    //--------------------------------------------------------------------------
    // Sum products to get constraint left-hand side
    // Also extend b[i] to match accumulator format
    
{accumulate_code}
    
    // Extend b_vector to accumulator format for comparison
    always @(posedge clk) begin
        if (state == COMPUTING && valid_pipe[1]) begin
            for (integer i = 0; i < PARALLEL; i = i + 1) begin
                if (constraint_base + i < P) begin
                    // Shift b[i] left by FRAC_BITS to match Q_(2*int+1).(2*frac) format
                    // b is in Q_int.frac, need to shift to Q_(2*int+1).(2*frac)
                    b_extended[i] <= {{{{$signed(b_vector[constraint_base + i]), 
                                        {{FRAC_BITS{{1'b0}}}}}}}};
                end else begin
                    b_extended[i] <= '0;
                end
            end
        end
    end
"""
        return code
    
    def _generate_direct_accumulate(self) -> str:
        """Direct accumulation for small m"""
        
        code = """    always @(posedge clk) begin
        if (state == COMPUTING && valid_pipe[1]) begin
            for (integer i = 0; i < PARALLEL; i = i + 1) begin
                // Direct sum of all products
                dot_products[i] <= """
        
        # Generate sum for each constraint
        sum_expr = " + \n                                   ".join(
            [f"products[i][{j}]" for j in range(self.m)]
        )
        
        code += sum_expr + """;
            end
        end
    end
"""
        return code
    
    def _generate_tree_accumulate(self) -> str:
        """Tree accumulation for large m"""
        num_level1 = (self.m + 1) // 2
        
        code = f"""    // Tree reduction for m > 4
    reg  [ACCUM_WIDTH-1:0] level1_sums [0:PARALLEL-1][0:{num_level1-1}];
    
    always @(posedge clk) begin
        if (state == COMPUTING && valid_pipe[1]) begin
            for (integer i = 0; i < PARALLEL; i = i + 1) begin
                // Level 1: Pairwise addition
"""
        
        for j in range(0, self.m, 2):
            if j + 1 < self.m:
                code += f"                level1_sums[i][{j//2}] <= products[i][{j}] + products[i][{j+1}];\n"
            else:
                code += f"                level1_sums[i][{j//2}] <= products[i][{j}];\n"
        
        code += """            end
        end
        
        if (state == COMPUTING && valid_pipe[2]) begin
            for (integer i = 0; i < PARALLEL; i = i + 1) begin
                // Level 2: Sum all level1 results
                dot_products[i] <= """
        
        level1_terms = " + \n                                       ".join(
            [f"level1_sums[i][{k}]" for k in range(num_level1)]
        )
        
        code += level1_terms + """;
            end
        end
    end
"""
        return code
    
    def _generate_stage3_compare(self) -> str:
        """Generate Stage 3: Compare A·x ≤ b"""
        
        stage_idx = 2 if self.m <= 4 else 3
        
        code = f"""    // Stage 3: Compare A·x ≤ b
    //--------------------------------------------------------------------------
    // Check if constraint is satisfied
    // Both sides are in Q_(2*int+1).(2*frac) format
    
    always @(posedge clk) begin
        if (state == COMPUTING && valid_pipe[{stage_idx}]) begin
            for (integer i = 0; i < PARALLEL; i = i + 1) begin
                // Signed comparison: A·x ≤ b
                // satisfied if dot_product <= b_extended
                constraint_satisfied[i] <= ($signed(dot_products[i]) <= $signed(b_extended[i]));
            end
        end
    end
"""
        return code
    
    def get_latency(self) -> int:
        """Return total latency in cycles"""
        # Pipeline latency per batch
        pipeline_latency = self.pipeline_stages + 1
        
        # Number of batches
        num_batches = math.ceil(self.p / self.parallel_constraints)
        
        # Total latency
        total_latency = pipeline_latency * num_batches + 2  # +2 for aggregation
        
        return total_latency
    
    def get_resource_estimate(self) -> Dict[str, int]:
        """Estimate FPGA resource usage"""
        
        # Multipliers per batch
        multipliers_per_batch = self.parallel_constraints * self.m
        
        # Adders for accumulation
        if self.m <= 4:
            adders_per_constraint = self.m - 1
        else:
            level1_adders = (self.m + 1) // 2
            level2_adders = level1_adders - 1
            adders_per_constraint = level1_adders + level2_adders
        
        total_adders = self.parallel_constraints * adders_per_constraint
        
        # Comparators
        comparators = self.parallel_constraints
        
        # Registers
        product_regs = self.parallel_constraints * self.m * self.product_width
        dot_product_regs = self.parallel_constraints * self.accumulator_width
        b_extended_regs = self.parallel_constraints * self.accumulator_width
        control_regs = 64 + self.p  # State machine + constraint results
        
        if self.m > 4:
            level1_regs = self.parallel_constraints * ((self.m+1)//2) * self.accumulator_width
        else:
            level1_regs = 0
        
        total_regs = product_regs + dot_product_regs + b_extended_regs + control_regs + level1_regs
        
        return {
            'DSP_blocks': multipliers_per_batch,
            'Adders_LUT': total_adders * (self.accumulator_width // 4),
            'Comparators_LUT': comparators * (self.accumulator_width // 4),
            'Registers_FF': total_regs,
            'Parallel_constraints': self.parallel_constraints,
            'Batches': math.ceil(self.p / self.parallel_constraints),
            'Latency_cycles': self.get_latency(),
            'Max_frequency_est_MHz': 150  # Lower than distance checker due to longer paths
        }
    
    def get_format_info(self) -> Dict[str, Any]:
        """Return fixed-point format information"""
        return {
            'input_format': f'Q{self.int_bits}.{self.frac_bits}',
            'input_width': self.data_width,
            'product_format': f'Q{self.product_int_bits}.{self.product_frac_bits}',
            'product_width': self.product_width,
            'accumulator_width': self.accumulator_width,
            'comparison_format': f'Q{self.product_int_bits}.{self.product_frac_bits}'
        }


# Factory function
def create_polyhedral_checker(config: Dict[str, Any]) -> PolyhedralCheckerGenerator:
    """
    Factory function to create polyhedral checker generator
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PolyhedralCheckerGenerator instance
    """
    return PolyhedralCheckerGenerator(config)


# Example usage
if __name__ == '__main__':
    # Test configuration
    test_config = {
        'n_parameters': 3,
        'n_constraints': 6,
        'int_bits': 7,
        'frac_bits': 8,
        'project_name': 'test_solver'
    }
    
    generator = create_polyhedral_checker(test_config)
    
    print("Polyhedral Checker Configuration:")
    print(f"  Parameters (m): {generator.m}")
    print(f"  Constraints (p): {generator.p}")
    print(f"  Parallel checks: {generator.parallel_constraints}")
    
    format_info = generator.get_format_info()
    print(f"\n  Input format: {format_info['input_format']} ({format_info['input_width']} bits)")
    print(f"  Product format: {format_info['product_format']} ({format_info['product_width']} bits)")
    print(f"  Comparison format: {format_info['comparison_format']}")
    
    print(f"\n  Pipeline stages: {generator.pipeline_stages}")
    print(f"  Total latency: {generator.get_latency()} cycles")
    
    print("\nResource Estimates:")
    resources = generator.get_resource_estimate()
    for key, value in resources.items():
        print(f"  {key}: {value}")
    
    # Generate module
    output = generator.generate('.')
    print(f"\nGenerated: {output}")