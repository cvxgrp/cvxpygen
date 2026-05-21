# scripts/bst_sol/fixed_point_cached/cache_prefetch.py
"""
Cache Prefetch FSM Generator
Handles L1/L2 cache population from ROM
"""

from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter
from .memory_layout import CachedMemoryLayoutGenerator


class CachePrefetchGenerator:
    """Generate cache prefetch FSM"""
    
    def __init__(self, config: PDQAPConfig, mem_gen: CachedMemoryLayoutGenerator):
        self.config = config
        self.mem_gen = mem_gen
        self.mem_config = mem_gen.get_memory_config()
        
        self.l2_ways = self.mem_config.l2_ways
        self.max_neighbors = self.mem_config.max_neighbors
        self.rom_latency = 1  # ROM read cycles
        
        # Parameter dimension
        self.n_params = self._get_param_dimension()
    
    def _get_param_dimension(self) -> int:
        if hasattr(self.config, 'n_parameters'):
            return self.config.n_parameters
        elif hasattr(self.config, 'param_dim'):
            return self.config.param_dim
        return 2
    
    def generate_prefetch_state_machine(self, writer: RTLWriter):
        """Generate prefetch FSM declarations"""
        writer.write_section("Prefetch FSM")
        writer.write_blank()
        
        writer.write_comment("Prefetch states")
        writer.write_line("localparam [1:0]")
        writer.write_line("    PREFETCH_IDLE     = 2'b00,")
        writer.write_line("    PREFETCH_L1_LOAD  = 2'b01,")
        writer.write_line("    PREFETCH_L2_LOAD  = 2'b10,")
        writer.write_line("    PREFETCH_L2_WAIT  = 2'b11;")
        writer.write_blank()
        
        writer.write_reg("prefetch_state", width="1:0")
        writer.write_reg("prefetch_next_state", width="1:0")
        
        index_range = f"{self.mem_config.index_width-1}:0"
        neighbor_bits = max(1, (self.max_neighbors - 1).bit_length())
        way_bits = max(1, (self.l2_ways - 1).bit_length())
        
        writer.write_reg("prefetch_counter", width=f"{neighbor_bits-1}:0")
        writer.write_reg("prefetch_hp_idx", width=index_range)
        writer.write_reg("prefetch_way", width=f"{way_bits-1}:0")
        writer.write_reg("prefetch_wait_cnt", width="1:0")
        writer.write_blank()
        
        # L1 load pipeline
        writer.write_reg("l1_load_pending")
        writer.write_reg("l1_load_hp_idx", width=index_range)
        writer.write_blank()
    
    def generate_rom_interface(self, writer: RTLWriter):
        """Generate ROM read interface"""
        writer.write_section("ROM Interface for Cache Loading")
        writer.write_blank()
        
        data_range = f"{self.config.data_width-1}:0"
        index_range = f"{self.mem_config.index_width-1}:0"
        
        writer.write_comment("ROM read control")
        writer.write_wire("rom_read_enable")
        writer.write_wire("rom_read_addr", width=index_range)
        writer.write_blank()
        
        writer.write_comment("ROM data outputs")
        for i in range(self.mem_config.halfplane_stride):
            writer.write_reg(f"rom_data_halfplane_{i}", width=data_range, signed=True)
        for i in range(self.mem_config.feedback_stride):
            writer.write_reg(f"rom_data_feedback_{i}", width=data_range, signed=True)
        writer.write_blank()
        
        writer.write_line(
            "assign rom_read_enable = "
            "(prefetch_state == PREFETCH_L1_LOAD && prefetch_wait_cnt == 0) || "
            "(prefetch_state == PREFETCH_L2_LOAD);"
        )
        writer.write_blank()
        
        writer.write_line("assign rom_read_addr = ")
        writer.write_line("    (prefetch_state == PREFETCH_L1_LOAD) ? l1_load_hp_idx :")
        writer.write_line("    (prefetch_state == PREFETCH_L2_LOAD) ? neighbor_list[prefetch_counter] :")
        writer.write_line("    0;")
        writer.write_blank()
    
    def generate_prefetch_fsm_logic(self, writer: RTLWriter):
        """Generate prefetch FSM state transition logic"""
        writer.write_section("Prefetch FSM State Transition")
        writer.write_blank()
        
        # State register
        writer.write_line("always @(posedge clk or negedge rst_n) begin")
        writer.indent()
        writer.begin_if("!rst_n")
        writer.write_line("prefetch_state <= PREFETCH_IDLE;")
        writer.begin_else()
        writer.write_line("prefetch_state <= prefetch_next_state;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # Next state logic
        writer.write_line("always @(*) begin")
        writer.indent()
        writer.write_line("prefetch_next_state = prefetch_state;")
        writer.write_blank()
        
        writer.write_line("case (prefetch_state)")
        writer.indent()
        
        writer.write_line("PREFETCH_IDLE: begin")
        writer.indent()
        writer.begin_if("verification_done && cache_miss")
        writer.write_line("prefetch_next_state = PREFETCH_L1_LOAD;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        writer.write_line("PREFETCH_L1_LOAD: begin")
        writer.indent()
        writer.begin_if(f"prefetch_wait_cnt >= {self.rom_latency}")
        writer.write_line("prefetch_next_state = PREFETCH_L2_LOAD;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        writer.write_line("PREFETCH_L2_LOAD: begin")
        writer.indent()
        writer.write_line("prefetch_next_state = PREFETCH_L2_WAIT;")
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        writer.write_line("PREFETCH_L2_WAIT: begin")
        writer.indent()
        writer.begin_if(f"prefetch_wait_cnt >= {self.rom_latency}")
        writer.begin_if("prefetch_counter >= neighbor_count")
        writer.write_line("prefetch_next_state = PREFETCH_IDLE;")
        writer.begin_else()
        writer.write_line("prefetch_next_state = PREFETCH_L2_LOAD;")
        writer.end_if()
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        writer.write_line("default: prefetch_next_state = PREFETCH_IDLE;")
        
        writer.dedent()
        writer.write_line("endcase")
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
    
    def generate_prefetch_datapath(self, writer: RTLWriter):
        """Generate prefetch datapath logic"""
        writer.write_section("Prefetch Datapath")
        writer.write_blank()
        
        writer.write_line("always @(posedge clk or negedge rst_n) begin")
        writer.indent()
        
        writer.begin_if("!rst_n")
        self._generate_reset_logic(writer)
        
        writer.begin_else()
        
        writer.write_line("case (prefetch_state)")
        writer.indent()
        
        # IDLE
        writer.write_line("PREFETCH_IDLE: begin")
        writer.indent()
        writer.begin_if("verification_done && cache_miss")
        writer.write_line("l1_load_hp_idx <= bst_result_region;")
        writer.write_line("l1_load_pending <= 1;")
        writer.write_line("prefetch_counter <= 0;")
        writer.write_line("prefetch_wait_cnt <= 0;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # L1_LOAD
        writer.write_line("PREFETCH_L1_LOAD: begin")
        writer.indent()
        writer.write_line("prefetch_wait_cnt <= prefetch_wait_cnt + 1;")
        writer.begin_if(f"prefetch_wait_cnt == {self.rom_latency - 1}")
        self._generate_l1_latch_logic(writer)
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # L2_LOAD
        writer.write_line("PREFETCH_L2_LOAD: begin")
        writer.indent()
        writer.begin_if("prefetch_counter < neighbor_count")
        writer.write_line("prefetch_hp_idx <= neighbor_list[prefetch_counter];")
        way_bits = max(1, (self.l2_ways - 1).bit_length())
        writer.write_line(f"prefetch_way <= prefetch_counter[{way_bits-1}:0];")
        writer.write_line("prefetch_wait_cnt <= 0;")
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # L2_WAIT
        writer.write_line("PREFETCH_L2_WAIT: begin")
        writer.indent()
        writer.write_line("prefetch_wait_cnt <= prefetch_wait_cnt + 1;")
        writer.begin_if(f"prefetch_wait_cnt == {self.rom_latency - 1}")
        self._generate_l2_latch_logic(writer)
        writer.end_if()
        writer.dedent()
        writer.write_line("end")
        
        writer.dedent()
        writer.write_line("endcase")
        
        writer.end_if()
        
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
    
    def _generate_reset_logic(self, writer: RTLWriter):
        """Generate reset logic for prefetch"""
        writer.write_line("l1_valid <= 0;")
        writer.write_line("l1_tag <= 0;")
        
        for i in range(self.n_params):
            writer.write_line(f"l1_theta_{i} <= 0;")
        
        writer.begin_for("i = 0", f"i < {self.l2_ways}", "i = i + 1")
        writer.write_line("l2_valid[i] <= 0;")
        writer.write_line("l2_tags[i] <= 0;")
        writer.write_line("l2_lru[i] <= i;")
        writer.end_for()
        
        writer.write_line("prefetch_counter <= 0;")
        writer.write_line("prefetch_wait_cnt <= 0;")
        writer.write_line("prefetch_hp_idx <= 0;")
        writer.write_line("prefetch_way <= 0;")
        writer.write_line("l1_load_pending <= 0;")
    
    def _generate_l1_latch_logic(self, writer: RTLWriter):
        """Generate L1 cache latch logic"""
        writer.write_line("l1_tag <= l1_load_hp_idx;")
        writer.write_line("l1_valid <= 1;")
        writer.write_blank()
        
        writer.write_comment("Update θ_cache")
        for i in range(self.n_params):
            writer.write_line(f"l1_theta_{i} <= theta_new_{i};")
        writer.write_blank()
        
        for i in range(self.mem_config.halfplane_stride):
            writer.write_line(f"l1_halfplane_{i} <= rom_data_halfplane_{i};")
        for i in range(self.mem_config.feedback_stride):
            writer.write_line(f"l1_feedback_{i} <= rom_data_feedback_{i};")
        
        writer.write_line("l1_load_pending <= 0;")
        writer.write_line("prefetch_wait_cnt <= 0;")
    
    def _generate_l2_latch_logic(self, writer: RTLWriter):
        """Generate L2 cache latch logic"""
        writer.write_line("l2_tags[prefetch_way] <= prefetch_hp_idx;")
        writer.write_line("l2_valid[prefetch_way] <= 1;")
        writer.write_blank()
        
        writer.write_line("case (prefetch_way)")
        writer.indent()
        for way in range(self.l2_ways):
            writer.write_line(f"{way}: begin")
            writer.indent()
            for i in range(self.mem_config.halfplane_stride):
                writer.write_line(f"l2_hp_{way}_{i} <= rom_data_halfplane_{i};")
            for i in range(self.mem_config.feedback_stride):
                writer.write_line(f"l2_fb_{way}_{i} <= rom_data_feedback_{i};")
            writer.dedent()
            writer.write_line("end")
        writer.dedent()
        writer.write_line("endcase")
        
        writer.write_line("prefetch_counter <= prefetch_counter + 1;")
        writer.write_line("prefetch_wait_cnt <= 0;")