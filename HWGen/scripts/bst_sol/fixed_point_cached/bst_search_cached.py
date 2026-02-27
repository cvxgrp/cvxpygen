# scripts/bst_sol/fixed_point_cached/bst_search.py
"""
BST Search Module Generator
Generates binary search tree traversal logic for region lookup
"""

from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter


class BSTSearchGenerator:
    """Generate BST search module"""
    
    def __init__(self, config: PDQAPConfig):
        self.config = config
        self.n_params = self._get_param_dimension()
        self.param_width = config.data_width
        
        # BST parameters
        self.max_depth = config.max_bst_depth if hasattr(config, 'max_bst_depth') else 20
        self.n_nodes = config.n_bst_nodes if hasattr(config, 'n_bst_nodes') else 1024
        self.rom_latency = 1  # ROM read latency in cycles
    
    def _get_param_dimension(self) -> int:
        """Extract parameter dimension"""
        if hasattr(self.config, 'n_parameters'):
            return self.config.n_parameters
        elif hasattr(self.config, 'param_dim'):
            return self.config.param_dim
        return 2
    
    def generate_bst_module(self, writer: RTLWriter):
        """Generate complete BST search module"""
        writer.write_section("BST Search Module")
        writer.write_blank()
        
        # Generate all components
        self._generate_state_machine(writer)
        self._generate_node_comparison(writer)
        self._generate_fsm_logic(writer)
        self._generate_datapath(writer)
        self._generate_output_signals(writer)
    
    def _generate_state_machine(self, writer: RTLWriter):
        """Generate BST traversal FSM"""
        writer.write_comment("BST Traversal State Machine")
        writer.write_blank()
        
        writer.write_line("localparam [2:0]")
        writer.write_line("    BST_IDLE        = 3'b000,")
        writer.write_line("    BST_LOAD_NODE   = 3'b001,")
        writer.write_line("    BST_COMPARE     = 3'b010,")
        writer.write_line("    BST_DECIDE      = 3'b011,")
        writer.write_line("    BST_LEAF_FOUND  = 3'b100,")
        writer.write_line("    BST_ERROR       = 3'b101;")
        writer.write_blank()
        
        writer.write_reg("bst_state", width="2:0")
        writer.write_reg("bst_next_state", width="2:0")
        writer.write_blank()
        
        # Traversal registers
        node_idx_bits = max(10, (self.n_nodes - 1).bit_length())
        depth_bits = max(5, (self.max_depth - 1).bit_length())
        
        writer.write_comment("Traversal registers")
        writer.write_reg("current_node", width=f"{node_idx_bits-1}:0")
        writer.write_reg("traversal_depth", width=f"{depth_bits-1}:0")
        writer.write_reg("comparison_result")
        writer.write_reg("is_leaf")
        writer.write_blank()
        
        # Node data registers
        writer.write_comment("Current node data")
        split_dim_bits = max(1, (self.n_params-1).bit_length())
        writer.write_reg("node_split_dim", width=f"{split_dim_bits-1}:0")
        writer.write_reg("node_split_value", width=f"{self.param_width-1}:0", signed=True)
        writer.write_reg("node_left_child", width=f"{node_idx_bits-1}:0")
        writer.write_reg("node_right_child", width=f"{node_idx_bits-1}:0")
        writer.write_reg("node_region_idx", width=f"{node_idx_bits-1}:0")
        writer.write_blank()
    
    def _generate_node_comparison(self, writer: RTLWriter):
        """Generate node comparison logic"""
        writer.write_comment("BST Node Comparison")
        writer.write_blank()
        
        writer.write_line(f"reg signed [{self.param_width-1}:0] query_param_at_split;")
        writer.write_blank()
        
        writer.write_line("always @(*) begin")
        writer.indent()
        writer.write_line("case (node_split_dim)")
        writer.indent()
        for i in range(self.n_params):
            writer.write_line(f"{i}: query_param_at_split = theta_new_{i};")
        writer.write_line("default: query_param_at_split = 0;")
        writer.dedent()
        writer.write_line("endcase")
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        writer.write_line("wire go_left = (query_param_at_split < node_split_value);")
        writer.write_blank()
    
    def _generate_fsm_logic(self, writer: RTLWriter):
        """Generate FSM state transition"""
        writer.write_comment("BST State Register")
        writer.write_line("always @(posedge clk or negedge rst_n) begin")
        writer.indent()
        writer.begin_if("!rst_n")
        writer.write_line("bst_state <= BST_IDLE;")
        writer.begin_else()
        writer.write_line("bst_state <= bst_next_state;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        writer.write_comment("BST Next State Logic")
        writer.write_line("always @(*) begin")
        writer.indent()
        writer.write_line("bst_next_state = bst_state;")
        writer.write_blank()
        
        writer.write_line("case (bst_state)")
        writer.indent()
        
        # IDLE
        writer.write_line("BST_IDLE: begin")
        writer.indent()
        writer.begin_if("cache_miss && query_valid")
        writer.write_line("bst_next_state = BST_LOAD_NODE;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # LOAD_NODE
        writer.write_line("BST_LOAD_NODE: begin")
        writer.write_line("    bst_next_state = BST_COMPARE;")
        writer.write_line("end")
        writer.write_blank()
        
        # COMPARE
        writer.write_line("BST_COMPARE: begin")
        writer.write_line("    bst_next_state = BST_DECIDE;")
        writer.write_line("end")
        writer.write_blank()
        
        # DECIDE
        writer.write_line("BST_DECIDE: begin")
        writer.indent()
        writer.begin_if("is_leaf")
        writer.write_line("bst_next_state = BST_LEAF_FOUND;")
        writer.begin_elseif(f"traversal_depth >= {self.max_depth}")
        writer.write_line("bst_next_state = BST_ERROR;")
        writer.begin_else()
        writer.write_line("bst_next_state = BST_LOAD_NODE;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # LEAF_FOUND
        writer.write_line("BST_LEAF_FOUND: begin")
        writer.write_line("    bst_next_state = BST_IDLE;")
        writer.write_line("end")
        writer.write_blank()
        
        # ERROR
        writer.write_line("BST_ERROR: begin")
        writer.write_line("    bst_next_state = BST_IDLE;")
        writer.write_line("end")
        writer.write_blank()
        
        writer.write_line("default: bst_next_state = BST_IDLE;")
        
        writer.dedent()
        writer.write_line("endcase")
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
    
    def _generate_datapath(self, writer: RTLWriter):
        """Generate traversal datapath"""
        writer.write_comment("BST Datapath Logic")
        writer.write_line("always @(posedge clk or negedge rst_n) begin")
        writer.indent()
        
        writer.begin_if("!rst_n")
        writer.write_line("current_node <= 0;")
        writer.write_line("traversal_depth <= 0;")
        writer.write_line("comparison_result <= 0;")
        
        writer.begin_else()
        
        writer.write_line("case (bst_state)")
        writer.indent()
        
        # IDLE
        writer.write_line("BST_IDLE: begin")
        writer.indent()
        writer.begin_if("cache_miss && query_valid")
        writer.write_line("current_node <= 0;  // Start at root")
        writer.write_line("traversal_depth <= 0;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # LOAD_NODE
        writer.write_line("BST_LOAD_NODE: begin")
        writer.write_line("    // Node data loaded via ROM interface")
        writer.write_line("end")
        writer.write_blank()
        
        # COMPARE
        writer.write_line("BST_COMPARE: begin")
        writer.write_line("    comparison_result <= go_left ? 0 : 1;")
        writer.write_line("end")
        writer.write_blank()
        
        # DECIDE
        writer.write_line("BST_DECIDE: begin")
        writer.indent()
        writer.begin_if("!is_leaf")
        writer.begin_if("comparison_result == 0")
        writer.write_line("current_node <= node_left_child;")
        writer.begin_else()
        writer.write_line("current_node <= node_right_child;")
        writer.end_if()
        writer.write_line("traversal_depth <= traversal_depth + 1;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        
        writer.dedent()
        writer.write_line("endcase")
        
        writer.end_if()
        
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
    
    def _generate_output_signals(self, writer: RTLWriter):
        """Generate output signals"""
        writer.write_comment("BST Output Signals")
        writer.write_line("assign bst_search_done = (bst_state == BST_LEAF_FOUND);")
        writer.write_line("assign bst_result_region = node_region_idx;")
        writer.write_line("assign bst_error = (bst_state == BST_ERROR);")
        writer.write_blank()