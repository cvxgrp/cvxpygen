// PDAQP Top - Paper-Compliant Two-Level Cache
module pdaqp_top #(
    parameter N_PARAMETERS = 2,
    parameter N_SOLUTIONS = 3,
    parameter DATA_WIDTH = 32
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
    wire [N_PARAMETERS*DATA_WIDTH-1:0] rom_G_row = {(N_PARAMETERS*DATA_WIDTH){1'b0}};
    wire [DATA_WIDTH-1:0] rom_g_val = {DATA_WIDTH{1'b0}};
    wire [7:0] rom_num_constraints = 8'd4;
    wire [N_SOLUTIONS*DATA_WIDTH-1:0] rom_solution = {(N_SOLUTIONS*DATA_WIDTH){1'b0}};
    
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
            result <= {(N_SOLUTIONS*DATA_WIDTH){1'b0}};
            query_reg <= {(N_PARAMETERS*DATA_WIDTH){1'b0}};
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
