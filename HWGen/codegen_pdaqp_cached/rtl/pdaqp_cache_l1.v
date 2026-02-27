// L1 Point Cache - Single Entry with Two-Step Verification
module pdaqp_cache_l1 #(
    parameter N_PARAMETERS = 2,
    parameter DATA_WIDTH = 32,
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
            cached_theta <= {(N_PARAMETERS*DATA_WIDTH){1'b0}};
            cached_region_idx <= 16'd0;
            valid <= 1'b0;
            hit <= 1'b0;
            region_idx <= 16'd0;
            query_done <= 1'b0;
            state <= IDLE;
            constraint_check_req <= 1'b0;
            constraint_region_idx <= 16'd0;
            distance_sq <= {DATA_WIDTH{1'b0}};
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
