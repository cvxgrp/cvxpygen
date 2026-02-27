# scripts/bst_sol/fixed_point_cached/memory_layout.py
"""
Memory Layout and Configuration Generator
Computes cache parameters, memory sizes, and address ranges
"""

from dataclasses import dataclass
from typing import Dict, List
from ..common.config_parser import PDQAPConfig


@dataclass
class CacheConfig:
    """Cache configuration parameters"""
    # L1 Cache
    l1_entries: int = 1
    
    # L2 Cache
    l2_ways: int = 4
    l2_sets: int = 1
    
    # Neighbor graph
    max_neighbors: int = 8
    avg_neighbors: float = 4.0
    
    # Address widths
    index_width: int = 10
    tag_width: int = 10
    
    # Data strides
    halfplane_stride: int = 0
    feedback_stride: int = 0
    
    # Total regions
    n_regions: int = 0


@dataclass
class MemoryAddressMap:
    """Memory address allocation map"""
    # ROM base addresses
    halfplane_rom_base: int = 0x0000
    feedback_rom_base: int = 0x4000
    neighbor_count_base: int = 0x8000
    neighbor_list_base: int = 0x9000
    
    # ROM sizes
    halfplane_rom_size: int = 0
    feedback_rom_size: int = 0
    neighbor_count_size: int = 0
    neighbor_list_size: int = 0
    
    # Total ROM size
    total_rom_size: int = 0


class CachedMemoryLayoutGenerator:
    """
    Memory layout and configuration generator
    Computes cache parameters, memory allocation, and timing
    """
    
    def __init__(self, config: PDQAPConfig):
        self.config = config
        self.n_params = self._get_param_dimension()
        self.data_width = config.data_width
        
        # Compute configurations
        self.cache_config = self._compute_cache_config()
        self.memory_map = self._compute_memory_map()
        self.timing_params = self._compute_timing_parameters()
        
        # Validate
        self._validate()
    
    def _get_param_dimension(self) -> int:
        """Extract parameter dimension from config"""
        if hasattr(self.config, 'n_parameters'):
            return self.config.n_parameters
        elif hasattr(self.config, 'param_dim'):
            return self.config.param_dim
        elif hasattr(self.config, 'p'):
            return self.config.p
        return 2
    
    def _compute_cache_config(self) -> CacheConfig:
        """Compute cache configuration parameters"""
        # Extract from config or use defaults
        n_regions = self.config.n_regions if hasattr(self.config, 'n_regions') else 1024
        max_neighbors = self.config.max_neighbors if hasattr(self.config, 'max_neighbors') else 8
        l2_ways = self.config.l2_ways if hasattr(self.config, 'l2_ways') else 4
        
        # Compute address widths
        index_width = max(10, (n_regions - 1).bit_length())
        tag_width = index_width  # For simplicity
        
        # Compute data strides
        n_halfplanes = self.config.n_halfplanes
        elements_per_halfplane = self.n_params + 1  # [a_0, ..., a_{p-1}, b]
        halfplane_stride = n_halfplanes * elements_per_halfplane
        
        n_states = self.config.n_states if hasattr(self.config, 'n_states') else 1
        n_inputs = self.config.n_inputs if hasattr(self.config, 'n_inputs') else 1
        feedback_stride = n_states * n_inputs
        
        return CacheConfig(
            l1_entries=1,
            l2_ways=l2_ways,
            l2_sets=1,
            max_neighbors=max_neighbors,
            avg_neighbors=max_neighbors * 0.5,
            index_width=index_width,
            tag_width=tag_width,
            halfplane_stride=halfplane_stride,
            feedback_stride=feedback_stride,
            n_regions=n_regions
        )
    
    def _compute_memory_map(self) -> MemoryAddressMap:
        """Compute memory address allocation"""
        n_regions = self.cache_config.n_regions
        words_per_entry = self.data_width // 8
        
        # Halfplane ROM size
        halfplane_entries = n_regions * self.cache_config.halfplane_stride
        halfplane_rom_size = halfplane_entries * words_per_entry
        
        # Feedback ROM size
        feedback_entries = n_regions * self.cache_config.feedback_stride
        feedback_rom_size = feedback_entries * words_per_entry
        
        # Neighbor count ROM size
        neighbor_count_size = n_regions * words_per_entry
        
        # Neighbor list ROM size
        neighbor_list_entries = n_regions * self.cache_config.max_neighbors
        neighbor_list_size = neighbor_list_entries * words_per_entry
        
        # Compute base addresses with alignment
        halfplane_base = 0x0000
        feedback_base = self._align_address(halfplane_base + halfplane_rom_size)
        neighbor_count_base = self._align_address(feedback_base + feedback_rom_size)
        neighbor_list_base = self._align_address(neighbor_count_base + neighbor_count_size)
        
        total_size = neighbor_list_base + neighbor_list_size
        
        return MemoryAddressMap(
            halfplane_rom_base=halfplane_base,
            feedback_rom_base=feedback_base,
            neighbor_count_base=neighbor_count_base,
            neighbor_list_base=neighbor_list_base,
            halfplane_rom_size=halfplane_rom_size,
            feedback_rom_size=feedback_rom_size,
            neighbor_count_size=neighbor_count_size,
            neighbor_list_size=neighbor_list_size,
            total_rom_size=total_size
        )
    
    def _align_address(self, addr: int, alignment: int = 1024) -> int:
        """Align address to boundary"""
        return ((addr + alignment - 1) // alignment) * alignment
    
    def _compute_timing_parameters(self) -> Dict[str, int]:
        """Compute timing parameters for cache operations"""
        target_freq = self.config.clock_freq_mhz if hasattr(self.config, 'clock_freq_mhz') else 100
        
        # ROM access latency (cycles)
        rom_latency = 1 if target_freq <= 100 else 2
        
        # L2 search cycles (worst case)
        l2_search_cycles = self.cache_config.l2_ways + 1
        
        # Distance computation cycles
        distance_cycles = self.n_params + 2
        
        # Polyhedral check cycles
        poly_cycles = self.config.n_halfplanes + 1
        
        return {
            'rom_latency': rom_latency,
            'l2_search_cycles': l2_search_cycles,
            'distance_cycles': distance_cycles,
            'polyhedral_cycles': poly_cycles,
            'total_hit_latency': distance_cycles + poly_cycles,
            'total_miss_latency': distance_cycles + l2_search_cycles + rom_latency * 2
        }
    
    def _validate(self):
        """Validate memory layout"""
        warnings = []
        errors = []
        
        # Check address overlaps
        ranges = [
            ('halfplane', self.memory_map.halfplane_rom_base, 
             self.memory_map.halfplane_rom_base + self.memory_map.halfplane_rom_size),
            ('feedback', self.memory_map.feedback_rom_base,
             self.memory_map.feedback_rom_base + self.memory_map.feedback_rom_size),
            ('neighbor_count', self.memory_map.neighbor_count_base,
             self.memory_map.neighbor_count_base + self.memory_map.neighbor_count_size),
            ('neighbor_list', self.memory_map.neighbor_list_base,
             self.memory_map.neighbor_list_base + self.memory_map.neighbor_list_size),
        ]
        
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                name1, start1, end1 = ranges[i]
                name2, start2, end2 = ranges[j]
                if not (end1 <= start2 or end2 <= start1):
                    errors.append(f"Address overlap: {name1} overlaps {name2}")
        
        # Check size limits
        max_rom_size = 1 << 20  # 1MB
        if self.memory_map.total_rom_size > max_rom_size:
            warnings.append(f"Total ROM size {self.memory_map.total_rom_size} bytes exceeds {max_rom_size} bytes")
        
        if errors:
            print("❌ Memory layout validation failed:")
            for error in errors:
                print(f"  ERROR: {error}")
            raise ValueError("Memory layout validation failed")
        
        if warnings:
            print("⚠️  Memory layout warnings:")
            for warning in warnings:
                print(f"  WARNING: {warning}")
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration"""
        return self.cache_config
    
    def get_memory_map(self) -> MemoryAddressMap:
        """Get memory address map"""
        return self.memory_map
    
    def get_timing_parameters(self) -> Dict[str, int]:
        """Get timing parameters"""
        return self.timing_params
    
    def print_summary(self):
        """Print memory layout summary"""
        print("\n" + "="*70)
        print("Memory Layout Summary")
        print("="*70)
        
        print("\n📦 Cache Configuration:")
        print(f"  L1 Entries: {self.cache_config.l1_entries}")
        print(f"  L2 Ways: {self.cache_config.l2_ways}")
        print(f"  L2 Sets: {self.cache_config.l2_sets}")
        print(f"  Max Neighbors: {self.cache_config.max_neighbors}")
        print(f"  Total Regions: {self.cache_config.n_regions}")
        
        print("\n📍 Address Map:")
        print(f"  Halfplane ROM:    [{self.memory_map.halfplane_rom_base:#06x} - "
              f"{self.memory_map.halfplane_rom_base + self.memory_map.halfplane_rom_size:#06x}]")
        print(f"  Feedback ROM:     [{self.memory_map.feedback_rom_base:#06x} - "
              f"{self.memory_map.feedback_rom_base + self.memory_map.feedback_rom_size:#06x}]")
        print(f"  Neighbor Count:   [{self.memory_map.neighbor_count_base:#06x} - "
              f"{self.memory_map.neighbor_count_base + self.memory_map.neighbor_count_size:#06x}]")
        print(f"  Neighbor List:    [{self.memory_map.neighbor_list_base:#06x} - "
              f"{self.memory_map.neighbor_list_base + self.memory_map.neighbor_list_size:#06x}]")
        print(f"  Total ROM Size:   {self.memory_map.total_rom_size} bytes "
              f"({self.memory_map.total_rom_size / 1024:.1f} KB)")
        
        print("\n⏱️  Timing Parameters:")
        for key, value in self.timing_params.items():
            print(f"  {key}: {value} cycles")
        
        print("\n" + "="*70 + "\n")