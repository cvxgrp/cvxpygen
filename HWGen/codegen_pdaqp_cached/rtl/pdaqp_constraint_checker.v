// Constraint Checker - Evaluates G*theta <= g
module pdaqp_constraint_checker #(
    parameter N_PARAMETERS = 2,
    parameter DATA_WIDTH = 32,
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
                        rom_addr <= {region_idx[7:0], constraint_idx};
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
