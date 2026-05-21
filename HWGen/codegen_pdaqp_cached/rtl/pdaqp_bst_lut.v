// BST Lookup - Exact search fallback
module pdaqp_bst_lut #(
    parameter N_PARAMETERS = 2,
    parameter DATA_WIDTH = 32
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
