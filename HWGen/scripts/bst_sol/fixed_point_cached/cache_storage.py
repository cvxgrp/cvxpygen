# scripts/bst_sol/fixed_point_cached/cache_storage.py
"""
Cache Storage Structure Generator
Generates L1/L2 cache storage declarations and neighbor lookup tables
"""

from typing import List
from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter
from .memory_layout import CachedMemoryLayoutGenerator


class CacheStorageGenerator:
    """Generate cache storage structures"""
    
    def __init__(self, config: PDQAPConfig, mem_gen: CachedMemoryLayoutGenerator):
        self.config = config
        self.mem_gen = mem_gen
        self.mem_config = mem_gen.get_memory_config()
        
        # Cache configuration
        self.l2_ways = self.mem_config.l2_ways
        self.max_neighbors = self.mem_config.max_neighbors
        
        # Parameter dimensions
        self.n_params = self._get_param_dimension()
        self.param_width = config.data_width
        
        # Data layout
        self.elements_per_halfplane = self.n_params + 1  # [a_0, ..., a_{p-1}, b]
    
    def _get_param_dimension(self) -> int:
        """Extract parameter dimension p from config"""
        if hasattr(self.config, 'n_parameters'):
            return self.config.n_parameters
        elif hasattr(self.config, 'param_dim'):
            return self.config.param_dim
        elif hasattr(self.config, 'p'):
            return self.config.p
        else:
            print("WARNING: Cannot determine parameter dimension, defaulting to 2")
            return 2
    
    def generate_l1_cache(self, writer: RTLWriter):
        """Generate L1 Point Cache storage"""
        writer.write_section("L1 Point Cache: (θ_cache, k_cache)")
        writer.write_blank()
        
        data_range = f"{self.config.data_width-1}:0"
        index_range = f"{self.mem_config.index_width-1}:0"
        
        writer.write_comment("L1 Cache valid and tag")
        writer.write_reg("l1_valid")
        writer.write_reg("l1_tag", width=index_range, comment="k_cache: region index")
        writer.write_blank()
        
        # Cached parameters θ_cache
        writer.write_comment(f"L1 cached parameter θ_cache (dimension p={self.n_params})")
        for i in range(self.n_params):
            writer.write_reg(f"l1_theta_{i}", width=f"{self.param_width-1}:0", signed=True)
        writer.write_blank()
        
        # Halfplane constraints
        writer.write_comment("L1 cached halfplane constraints: G_k_cache * θ ≤ g_k_cache")
        for i in range(self.mem_config.halfplane_stride):
            writer.write_reg(f"l1_halfplane_{i}", width=data_range, signed=True)
        writer.write_blank()
        
        # Feedback coefficients
        writer.write_comment("L1 cached feedback coefficients K_k_cache")
        for i in range(self.mem_config.feedback_stride):
            writer.write_reg(f"l1_feedback_{i}", width=data_range, signed=True)
        writer.write_blank()
    
    def generate_l2_cache(self, writer: RTLWriter):
        """Generate L2 Neighbor Cache storage"""
        writer.write_section(f"L2 Neighbor Cache: N(k_cache) with {self.l2_ways} ways")
        writer.write_blank()
        
        data_range = f"{self.config.data_width-1}:0"
        index_range = f"{self.mem_config.index_width-1}:0"
        way_bits = max(1, (self.l2_ways - 1).bit_length())
        
        # L2 metadata
        writer.write_comment("L2 Cache metadata")
        writer.write_reg("l2_valid", array=f"0:{self.l2_ways-1}")
        writer.write_reg("l2_tags", width=index_range, array=f"0:{self.l2_ways-1}")
        writer.write_reg("l2_lru", width=f"{way_bits-1}:0", array=f"0:{self.l2_ways-1}")
        writer.write_blank()
        
        # L2 halfplane data
        writer.write_comment("L2 cached halfplane constraints for neighbor regions")
        for way in range(self.l2_ways):
            for i in range(self.mem_config.halfplane_stride):
                writer.write_reg(f"l2_hp_{way}_{i}", width=data_range, signed=True)
        writer.write_blank()
        
        # L2 feedback data
        writer.write_comment("L2 cached feedback coefficients for neighbor regions")
        for way in range(self.l2_ways):
            for i in range(self.mem_config.feedback_stride):
                writer.write_reg(f"l2_fb_{way}_{i}", width=data_range, signed=True)
        writer.write_blank()
    
    def generate_neighbor_lookup(self, writer: RTLWriter):
        """Generate neighbor adjacency table lookup"""
        writer.write_section("Neighbor Adjacency Lookup: N(k_cache)")
        writer.write_blank()
        
        project = self.config.project_name
        neighbor_bits = max(1, (self.max_neighbors - 1).bit_length())
        index_range = f"{self.mem_config.index_width-1}:0"
        
        # Neighbor count ROM
        writer.write_comment("Neighbor count lookup: |N(k)|")
        writer.write_reg(f"neighbor_count_rom [{neighbor_bits-1}:0]", 
                        array=f"0:{self.mem_config.n_regions-1}")
        writer.write_line("initial begin")
        writer.indent()
        writer.write_line(f'$readmemh("../include/{project}_neighbor_count.mem", neighbor_count_rom);')
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        writer.write_comment("Neighbor list lookup: N(k_cache) = {j1, j2, ..., jd}")
        total_entries = self.mem_config.n_regions * self.max_neighbors
        writer.write_reg(f"neighbor_list_rom [{index_range}]", 
                        array=f"0:{total_entries-1}")
        writer.write_line("initial begin")
        writer.indent()
        writer.write_line(f'$readmemh("../include/{project}_neighbor_list.mem", neighbor_list_rom);')
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # Output wires
        writer.write_comment("Neighbor lookup outputs")
        writer.write_wire("neighbor_count", width=f"{neighbor_bits-1}:0")
        writer.write_wire("neighbor_list", width=index_range, 
                         array=f"0:{self.max_neighbors-1}")
        writer.write_blank()
        
        writer.write_line("assign neighbor_count = neighbor_count_rom[l1_tag];")
        writer.write_blank()
        
        # Unpack neighbor list
        writer.write_comment("Unpack neighbor list for current k_cache")
        writer.write_line("genvar nb_idx;")
        writer.write_line("generate")
        writer.indent()
        writer.write_line(f"for (nb_idx = 0; nb_idx < {self.max_neighbors}; nb_idx = nb_idx + 1) begin: unpack_neighbors")
        writer.indent()
        writer.write_line(
            f"assign neighbor_list[nb_idx] = "
            f"neighbor_list_rom[l1_tag * {self.max_neighbors} + nb_idx];"
        )
        writer.dedent()
        writer.write_line("end")
        writer.dedent()
        writer.write_line("endgenerate")
        writer.write_blank()
    
    def generate_data_output_mux(self, writer: RTLWriter):
        """Generate cache data output multiplexer"""
        writer.write_section("Cache Data Output Multiplexing")
        writer.write_blank()
        
        data_range = f"{self.config.data_width-1}:0"
        
        # Halfplane outputs
        writer.write_comment("Halfplane coefficient outputs")
        for i in range(self.mem_config.halfplane_stride):
            writer.write_line(f"reg signed [{data_range}] cached_halfplane_{i};")
        writer.write_blank()
        
        writer.write_line("always @(*) begin")
        writer.indent()
        writer.begin_if("polyhedral_pass")
        writer.write_comment("L1 cache hit - use L1 data")
        for i in range(self.mem_config.halfplane_stride):
            writer.write_line(f"cached_halfplane_{i} = l1_halfplane_{i};")
        
        writer.begin_elseif("l2_found")
        writer.write_comment("L2 cache hit - use L2 data from found way")
        writer.write_line("case (l2_found_way)")
        writer.indent()
        for way in range(self.l2_ways):
            writer.write_line(f"{way}: begin")
            writer.indent()
            for i in range(self.mem_config.halfplane_stride):
                writer.write_line(f"cached_halfplane_{i} = l2_hp_{way}_{i};")
            writer.dedent()
            writer.write_line("end")
        writer.write_line("default: begin")
        writer.indent()
        for i in range(self.mem_config.halfplane_stride):
            writer.write_line(f"cached_halfplane_{i} = 0;")
        writer.dedent()
        writer.write_line("end")
        writer.dedent()
        writer.write_line("endcase")
        
        writer.begin_else()
        writer.write_comment("Cache miss - output zeros (BST will provide data)")
        for i in range(self.mem_config.halfplane_stride):
            writer.write_line(f"cached_halfplane_{i} = 0;")
        writer.end_if()
        
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # Feedback outputs
        writer.write_comment("Feedback coefficient outputs")
        for i in range(self.mem_config.feedback_stride):
            writer.write_line(f"reg signed [{data_range}] cached_feedback_{i};")
        writer.write_blank()
        
        writer.write_line("always @(*) begin")
        writer.indent()
        writer.begin_if("polyhedral_pass")
        for i in range(self.mem_config.feedback_stride):
            writer.write_line(f"cached_feedback_{i} = l1_feedback_{i};")
        
        writer.begin_elseif("l2_found")
        writer.write_line("case (l2_found_way)")
        writer.indent()
        for way in range(self.l2_ways):
            writer.write_line(f"{way}: begin")
            writer.indent()
            for i in range(self.mem_config.feedback_stride):
                writer.write_line(f"cached_feedback_{i} = l2_fb_{way}_{i};")
            writer.dedent()
            writer.write_line("end")
        writer.write_line("default: begin")
        writer.indent()
        for i in range(self.mem_config.feedback_stride):
            writer.write_line(f"cached_feedback_{i} = 0;")
        writer.dedent()
        writer.write_line("end")
        writer.dedent()
        writer.write_line("endcase")
        
        writer.begin_else()
        for i in range(self.mem_config.feedback_stride):
            writer.write_line(f"cached_feedback_{i} = 0;")
        writer.end_if()
        
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()