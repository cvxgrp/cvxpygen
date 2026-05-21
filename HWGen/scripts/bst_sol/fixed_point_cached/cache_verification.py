# scripts/bst_sol/fixed_point_cached/cache_verification.py
"""
Two-Step Verification Logic Generator
Implements distance filtering and polyhedral verification
"""

from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter
from .memory_layout import CachedMemoryLayoutGenerator


class CacheVerificationGenerator:
    """Generate two-step verification logic"""
    
    def __init__(self, config: PDQAPConfig, mem_gen: CachedMemoryLayoutGenerator):
        self.config = config
        self.mem_gen = mem_gen
        self.mem_config = mem_gen.get_memory_config()
        
        # Parameter dimensions
        self.n_params = self._get_param_dimension()
        self.param_width = config.data_width
        
        # Verification threshold
        self.distance_threshold_sq = self._compute_distance_threshold()
        
        # Halfplane layout
        self.elements_per_halfplane = self.n_params + 1
    
    def _get_param_dimension(self) -> int:
        """Extract parameter dimension p from config"""
        if hasattr(self.config, 'n_parameters'):
            return self.config.n_parameters
        elif hasattr(self.config, 'param_dim'):
            return self.config.param_dim
        elif hasattr(self.config, 'p'):
            return self.config.p
        else:
            return 2
    
    def _compute_distance_threshold(self) -> int:
        """Compute ε² threshold for distance filtering"""
        if hasattr(self.config, 'distance_threshold'):
            return self.config.distance_threshold
        
        # Default: ε² = 2^-20
        frac_bits = self.config.frac_width if hasattr(self.config, 'frac_width') else 16
        epsilon_sq_fixed = (1 << (2 * frac_bits - 20))
        return epsilon_sq_fixed
    
    def generate_verification_state_machine(self, writer: RTLWriter):
        """Generate verification FSM declarations"""
        writer.write_section("Two-Step Verification FSM")
        writer.write_blank()
        
        writer.write_comment("Verification states")
        writer.write_line("localparam [2:0]")
        writer.write_line("    VERIFY_IDLE           = 3'b000,")
        writer.write_line("    VERIFY_DISTANCE       = 3'b001,  // Step 1: ||θ_new - θ_cache||₂ ≤ ε")
        writer.write_line("    VERIFY_POLYHEDRAL     = 3'b010,  // Step 2: G_k * θ_new ≤ g_k")
        writer.write_line("    VERIFY_L2_SEARCH      = 3'b011,  // Search L2 neighbors")
        writer.write_line("    VERIFY_FULL_SEARCH    = 3'b100,  // Fall back to exact BST")
        writer.write_line("    VERIFY_DONE           = 3'b101;")
        writer.write_blank()
        
        writer.write_reg("verify_state", width="2:0")
        writer.write_reg("verify_next_state", width="2:0")
        writer.write_blank()
        
        # Distance computation registers
        writer.write_comment("Distance filtering registers")
        for i in range(self.n_params):
            diff_width = self.param_width + 1
            writer.write_reg(f"theta_diff_{i}", width=f"{diff_width-1}:0", signed=True)
        
        sum_width = 2 * (self.param_width + 1) + self.n_params.bit_length()
        writer.write_reg("distance_squared", width=f"{sum_width-1}:0")
        writer.write_reg("distance_valid")
        writer.write_blank()
        
        # Polyhedral verification registers
        writer.write_comment("Polyhedral verification registers")
        n_halfplanes = self.config.n_halfplanes
        hp_idx_bits = max(1, (n_halfplanes - 1).bit_length())
        writer.write_reg("poly_check_idx", width=f"{hp_idx_bits-1}:0")
        writer.write_reg("poly_all_satisfied")
        writer.write_blank()
        
        # L2 search registers
        way_bits = max(1, (self.mem_config.l2_ways - 1).bit_length())
        writer.write_comment("L2 neighbor search registers")
        writer.write_reg("l2_search_idx", width=f"{way_bits-1}:0")
        writer.write_reg("l2_found")
        writer.write_reg("l2_found_way", width=f"{way_bits-1}:0")
        writer.write_blank()
        
        # Input parameter registers
        writer.write_comment("Input parameter θ_new (captured from query)")
        for i in range(self.n_params):
            writer.write_reg(f"theta_new_{i}", width=f"{self.param_width-1}:0", signed=True)
        writer.write_blank()
    
    def generate_distance_filtering(self, writer: RTLWriter):
        """Generate Step 1: Distance Filtering Logic"""
        writer.write_section("Step 1: Distance Filtering")
        writer.write_comment(f"Check: ||θ_new - θ_cache||₂² ≤ ε² = {self.distance_threshold_sq}")
        writer.write_blank()
        
        # Compute differences
        writer.write_comment("Compute parameter differences: Δθ_i = θ_new_i - θ_cache_i")
        writer.write_line("always @(posedge clk or negedge rst_n) begin")
        writer.indent()
        writer.begin_if("!rst_n")
        for i in range(self.n_params):
            writer.write_line(f"theta_diff_{i} <= 0;")
        writer.begin_else()
        writer.begin_if("verify_state == VERIFY_DISTANCE")
        for i in range(self.n_params):
            writer.write_line(f"theta_diff_{i} <= $signed(theta_new_{i}) - $signed(l1_theta_{i});")
        writer.end_if()
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # Compute squared distance
        writer.write_comment("Compute squared Euclidean distance: ||Δθ||₂² = Σ(Δθ_i)²")
        writer.write_line("always @(posedge clk or negedge rst_n) begin")
        writer.indent()
        writer.begin_if("!rst_n")
        writer.write_line("distance_squared <= 0;")
        writer.write_line("distance_valid <= 0;")
        writer.begin_else()
        writer.begin_if("verify_state == VERIFY_DISTANCE")
        
        writer.write_line("distance_squared <= ")
        writer.indent()
        for i in range(self.n_params):
            if i == 0:
                writer.write_line(f"($signed(theta_diff_{i}) * $signed(theta_diff_{i}))")
            else:
                writer.write_line(f"+ ($signed(theta_diff_{i}) * $signed(theta_diff_{i}))")
        writer.write_line(";")
        writer.dedent()
        writer.write_line("distance_valid <= 1;")
        
        writer.begin_else()
        writer.write_line("distance_valid <= 0;")
        writer.end_if()
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # Distance check result
        writer.write_comment("Distance check result")
        writer.write_line(f"wire distance_pass = distance_valid && (distance_squared <= {self.distance_threshold_sq});")
        writer.write_line("wire distance_fail = distance_valid && !distance_pass;")
        writer.write_blank()
    
    def generate_polyhedral_verification(self, writer: RTLWriter):
        """Generate Step 2: Polyhedral Verification Logic"""
        writer.write_section("Step 2: Polyhedral Verification")
        writer.write_comment("Check: G_k * θ_new ≤ g_k for all halfplanes")
        writer.write_blank()
        
        n_halfplanes = self.config.n_halfplanes
        
        # Dot product computation
        dot_width = 2 * self.param_width + self.n_params.bit_length()
        writer.write_line(f"wire signed [{dot_width-1}:0] dot_products [0:{n_halfplanes-1}];")
        writer.write_line(f"wire constraint_satisfied [0:{n_halfplanes-1}];")
        writer.write_blank()
        
        # Generate constraint checkers
        writer.write_line("genvar hp_i;")
        writer.write_line("generate")
        writer.indent()
        writer.write_line(f"for (hp_i = 0; hp_i < {n_halfplanes}; hp_i = hp_i + 1) begin: halfplane_check")
        writer.indent()
        
        # Extract coefficients
        writer.write_comment("Extract halfplane coefficients [a_0, ..., a_{p-1}, b]")
        for j in range(self.n_params):
            idx = f"hp_i * {self.elements_per_halfplane} + {j}"
            writer.write_line(f"wire signed [{self.param_width-1}:0] a_{j} = l1_halfplane_{{{idx}}};")
        
        b_idx = f"hp_i * {self.elements_per_halfplane} + {self.n_params}"
        writer.write_line(f"wire signed [{self.param_width-1}:0] b = l1_halfplane_{{{b_idx}}};")
        writer.write_blank()
        
        # Compute dot product
        writer.write_comment("Dot product: a^T * θ_new")
        writer.write_line("assign dot_products[hp_i] = ")
        writer.indent()
        for j in range(self.n_params):
            if j == 0:
                writer.write_line(f"($signed(a_{j}) * $signed(theta_new_{j}))")
            else:
                writer.write_line(f"+ ($signed(a_{j}) * $signed(theta_new_{j}))")
        writer.write_line(";")
        writer.dedent()
        writer.write_blank()
        
        # Check constraint
        writer.write_comment("Check if a^T * θ_new ≤ b")
        writer.write_line("assign constraint_satisfied[hp_i] = (dot_products[hp_i] <= $signed(b));")
        
        writer.dedent()
        writer.write_line("end")
        writer.dedent()
        writer.write_line("endgenerate")
        writer.write_blank()
        
        # Check all constraints
        writer.write_comment("All constraints must be satisfied")
        writer.write_line("always @(posedge clk or negedge rst_n) begin")
        writer.indent()
        writer.begin_if("!rst_n")
        writer.write_line("poly_all_satisfied <= 0;")
        writer.begin_else()
        writer.begin_if("verify_state == VERIFY_POLYHEDRAL")
        
        writer.write_line("poly_all_satisfied <= ")
        writer.indent()
        for i in range(n_halfplanes):
            if i == 0:
                writer.write_line(f"constraint_satisfied[{i}]")
            else:
                writer.write_line(f"&& constraint_satisfied[{i}]")
        writer.write_line(";")
        writer.dedent()
        
        writer.end_if()
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        writer.write_line("wire polyhedral_pass = poly_all_satisfied;")
        writer.write_line("wire polyhedral_fail = (verify_state == VERIFY_POLYHEDRAL) && !poly_all_satisfied;")
        writer.write_blank()
    
    def generate_verification_fsm_logic(self, writer: RTLWriter):
        """Generate verification FSM state transition logic"""
        writer.write_section("Verification FSM State Transition")
        writer.write_blank()
        
        # State register
        writer.write_line("always @(posedge clk or negedge rst_n) begin")
        writer.indent()
        writer.begin_if("!rst_n")
        writer.write_line("verify_state <= VERIFY_IDLE;")
        writer.begin_else()
        writer.write_line("verify_state <= verify_next_state;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # Next state logic
        writer.write_line("always @(*) begin")
        writer.indent()
        writer.write_line("verify_next_state = verify_state;")
        writer.write_blank()
        
        writer.write_line("case (verify_state)")
        writer.indent()
        
        # IDLE
        writer.write_line("VERIFY_IDLE: begin")
        writer.indent()
        writer.begin_if("query_valid && l1_valid")
        writer.write_line("verify_next_state = VERIFY_DISTANCE;")
        writer.begin_elseif("query_valid && !l1_valid")
        writer.write_line("verify_next_state = VERIFY_FULL_SEARCH;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # DISTANCE
        writer.write_line("VERIFY_DISTANCE: begin")
        writer.indent()
        writer.begin_if("distance_pass")
        writer.write_line("verify_next_state = VERIFY_POLYHEDRAL;")
        writer.begin_elseif("distance_fail")
        writer.write_line("verify_next_state = VERIFY_L2_SEARCH;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # POLYHEDRAL
        writer.write_line("VERIFY_POLYHEDRAL: begin")
        writer.indent()
        writer.begin_if("polyhedral_pass")
        writer.write_line("verify_next_state = VERIFY_DONE;")
        writer.begin_elseif("polyhedral_fail")
        writer.write_line("verify_next_state = VERIFY_L2_SEARCH;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # L2_SEARCH
        writer.write_line("VERIFY_L2_SEARCH: begin")
        writer.indent()
        writer.begin_if("l2_found")
        writer.write_line("verify_next_state = VERIFY_DONE;")
        writer.begin_elseif(f"l2_search_idx >= {self.mem_config.l2_ways}")
        writer.write_line("verify_next_state = VERIFY_FULL_SEARCH;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # FULL_SEARCH
        writer.write_line("VERIFY_FULL_SEARCH: begin")
        writer.write_line("    verify_next_state = VERIFY_DONE;")
        writer.write_line("end")
        writer.write_blank()
        
        # DONE
        writer.write_line("VERIFY_DONE: begin")
        writer.write_line("    verify_next_state = VERIFY_IDLE;")
        writer.write_line("end")
        writer.write_blank()
        
        writer.write_line("default: verify_next_state = VERIFY_IDLE;")
        
        writer.dedent()
        writer.write_line("endcase")
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # Output signals
        writer.write_comment("Verification result signals")
        writer.write_line("assign cache_hit = (verify_state == VERIFY_DONE) && (polyhedral_pass || l2_found);")
        writer.write_line("assign cache_miss = (verify_state == VERIFY_FULL_SEARCH);")
        writer.write_line("assign verification_done = (verify_state == VERIFY_DONE);")
        writer.write_blank()
    
    def generate_l2_search_logic(self, writer: RTLWriter):
        """Generate L2 neighbor search logic"""
        writer.write_section("L2 Neighbor Search Logic")
        writer.write_blank()
        
        writer.write_line("always @(posedge clk or negedge rst_n) begin")
        writer.indent()
        writer.begin_if("!rst_n")
        writer.write_line("l2_search_idx <= 0;")
        writer.write_line("l2_found <= 0;")
        writer.write_line("l2_found_way <= 0;")
        writer.begin_else()
        
        writer.begin_if("verify_state == VERIFY_L2_SEARCH")
        writer.begin_if(f"l2_search_idx < {self.mem_config.l2_ways}")
        
        writer.begin_if("l2_valid[l2_search_idx]")
        writer.write_comment("TODO: Add polyhedral check for L2 entry")
        writer.write_comment("For now, accept first valid neighbor")
        writer.write_line("l2_found <= 1;")
        writer.write_line("l2_found_way <= l2_search_idx;")
        writer.end_if()
        
        writer.write_line("l2_search_idx <= l2_search_idx + 1;")
        writer.end_if()
        
        writer.begin_else()
        writer.write_line("l2_search_idx <= 0;")
        writer.write_line("l2_found <= 0;")
        writer.end_if()
        
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()