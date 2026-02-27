// L2 Neighbor Cache - Stores adjacency list N(k_cache)
module pdaqp_cache_l2 #(
    parameter N_PARAMETERS = 2,
    parameter DATA_WIDTH = 32,
    parameter MAX_NEIGHBORS = 16
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
