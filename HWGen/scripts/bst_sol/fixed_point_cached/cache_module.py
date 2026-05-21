# scripts/bst_sol/fixed_point_cached/cache_module.py
"""
Two-Level Cache Module - Main Entry Point
Coordinates cache storage, verification, and prefetch modules
"""

from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter
from .memory_layout import CachedMemoryLayoutGenerator
from .cache_storage import CacheStorageGenerator
from .cache_verification import CacheVerificationGenerator
from .cache_prefetch import CachePrefetchGenerator


class TwoLevelCacheGenerator:
    """Main coordinator for two-level cache generation"""
    
    def __init__(self, config: PDQAPConfig, mem_gen: CachedMemoryLayoutGenerator):
        self.config = config
        self.mem_gen = mem_gen
        self.mem_config = mem_gen.get_memory_config()
        
        # Instantiate sub-generators
        self.storage_gen = CacheStorageGenerator(config, mem_gen)
        self.verify_gen = CacheVerificationGenerator(config, mem_gen)
        self.prefetch_gen = CachePrefetchGenerator(config, mem_gen)
    
    def generate_cache_module(self, writer: RTLWriter):
        """Generate complete cache module"""
        writer.write_section("Two-Level Cache Module with Two-Step Verification")
        writer.write_blank()
        
        # Storage structures
        self.storage_gen.generate_l1_cache(writer)
        self.storage_gen.generate_l2_cache(writer)
        self.storage_gen.generate_neighbor_lookup(writer)
        
        # Verification logic
        self.verify_gen.generate_verification_state_machine(writer)
        self.verify_gen.generate_distance_filtering(writer)
        self.verify_gen.generate_polyhedral_verification(writer)
        self.verify_gen.generate_verification_fsm_logic(writer)
        self.verify_gen.generate_l2_search_logic(writer)
        
        # Prefetch FSM
        self.prefetch_gen.generate_prefetch_state_machine(writer)
        self.prefetch_gen.generate_rom_interface(writer)
        self.prefetch_gen.generate_prefetch_fsm_logic(writer)
        self.prefetch_gen.generate_prefetch_datapath(writer)
        
        # Data output
        self.storage_gen.generate_data_output_mux(writer)


# Integration helper
def integrate_cache_into_bst(writer: RTLWriter, config: PDQAPConfig, 
                             mem_gen: CachedMemoryLayoutGenerator):
    """
    Integration helper: Insert cache logic into BST module
    
    Args:
        writer: RTL writer instance
        config: PDQAP configuration
        mem_gen: Memory layout generator
    
    Returns:
        TwoLevelCacheGenerator instance
    """
    cache_gen = TwoLevelCacheGenerator(config, mem_gen)
    cache_gen.generate_cache_module(writer)
    return cache_gen