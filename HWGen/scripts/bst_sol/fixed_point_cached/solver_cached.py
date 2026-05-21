# scripts/bst_sol/fixed_point_cached/solver_cached.py

import math
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import json

from ..common.config_parser import VerilogConfigParser, PDQAPConfig

logger = logging.getLogger(__name__)


class AdaptiveCacheConfig:
    """Auto-configure two-level cache based on problem size"""
    
    def __init__(self, n_regions: int, n_solutions: int, data_width: int):
        self.n_regions = n_regions
        self.n_solutions = n_solutions
        self.data_width = data_width
        self.entry_size = n_solutions * data_width
        
        self.l1_config = self._compute_l1_config()
        self.l2_config = self._compute_l2_config()
    
    def _compute_l1_config(self) -> dict:
        """L1: on-chip SRAM, direct-mapped, covers 1-5% hot regions"""
        if self.n_regions <= 16:
            lines = self.n_regions
        elif self.n_regions <= 256:
            lines = 16
        elif self.n_regions <= 1024:
            lines = 32
        else:
            lines = min(64, int(math.log2(self.n_regions) * 4))
        
        lines = 2 ** int(math.ceil(math.log2(lines)))
        index_bits = int(math.log2(lines))
        region_addr_bits = int(math.ceil(math.log2(self.n_regions)))
        tag_bits = max(1, region_addr_bits - index_bits)
        
        # Estimate area and power
        bits_per_line = self.entry_size + tag_bits + 1  # data + tag + valid
        total_bits = lines * bits_per_line
        area_mm2 = total_bits * 0.0001  # rough estimate
        power_mw = total_bits * 0.00005  # rough estimate
        
        return {
            'lines': lines,
            'mapping': 'direct',
            'index_bits': index_bits,
            'tag_bits': tag_bits,
            'bits_per_line': bits_per_line,
            'total_bits': total_bits,
            'area_mm2': area_mm2,
            'power_mw': power_mw,
        }
    
    def _compute_l2_config(self) -> dict:
        """L2: off-chip DRAM, set-associative, covers 10-25% working set"""
        l1_lines = self.l1_config['lines']
        
        if self.n_regions <= 64:
            return {'enabled': False}
        elif self.n_regions <= 512:
            lines = l1_lines * 4
            ways = 2
        elif self.n_regions <= 2048:
            lines = l1_lines * 8
            ways = 4
        else:
            lines = min(256, l1_lines * 16)
            ways = 4
        
        lines = 2 ** int(math.ceil(math.log2(lines)))
        sets = lines // ways
        index_bits = int(math.log2(sets))
        region_addr_bits = int(math.ceil(math.log2(self.n_regions)))
        tag_bits = max(1, region_addr_bits - index_bits)
        
        # Estimate area and power
        bits_per_line = self.entry_size + tag_bits + 1 + int(math.log2(ways))
        total_bits = lines * bits_per_line
        area_mm2 = total_bits * 0.00005  # lower density for DRAM
        power_mw = total_bits * 0.00002
        
        return {
            'enabled': True,
            'lines': lines,
            'ways': ways,
            'sets': sets,
            'index_bits': index_bits,
            'tag_bits': tag_bits,
            'bits_per_line': bits_per_line,
            'total_bits': total_bits,
            'area_mm2': area_mm2,
            'power_mw': power_mw,
        }


class L1Cache:
    """Direct-mapped L1 cache implementation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.lines = config['lines']
        self.index_bits = config['index_bits']
        self.tag_bits = config['tag_bits']
        
        # Cache storage: {index: (tag, valid, data)}
        self.storage = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def access(self, address: int) -> Tuple[bool, Optional[np.ndarray]]:
        """Access cache line, returns (hit, data)"""
        index = address & ((1 << self.index_bits) - 1)
        tag = address >> self.index_bits
        
        if index in self.storage:
            stored_tag, valid, data = self.storage[index]
            if valid and stored_tag == tag:
                self.hits += 1
                return True, data
        
        self.misses += 1
        return False, None
    
    def insert(self, address: int, data: np.ndarray):
        """Insert data into cache"""
        index = address & ((1 << self.index_bits) - 1)
        tag = address >> self.index_bits
        
        if index in self.storage and self.storage[index][1]:
            self.evictions += 1
        
        self.storage[index] = (tag, True, data.copy())
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'utilization': len(self.storage) / self.lines
        }


class L2Cache:
    """Set-associative L2 cache implementation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        
        if not self.enabled:
            return
        
        self.lines = config['lines']
        self.ways = config['ways']
        self.sets = config['sets']
        self.index_bits = config['index_bits']
        self.tag_bits = config['tag_bits']
        
        # Cache storage: {set_index: [(tag, valid, data, lru_counter), ...]}
        self.storage = {i: [] for i in range(self.sets)}
        self.lru_counter = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def access(self, address: int) -> Tuple[bool, Optional[np.ndarray]]:
        """Access cache line, returns (hit, data)"""
        if not self.enabled:
            return False, None
        
        set_index = address & ((1 << self.index_bits) - 1)
        tag = address >> self.index_bits
        
        cache_set = self.storage[set_index]
        
        for i, (stored_tag, valid, data, lru) in enumerate(cache_set):
            if valid and stored_tag == tag:
                self.hits += 1
                # Update LRU
                cache_set[i] = (stored_tag, valid, data, self.lru_counter)
                self.lru_counter += 1
                return True, data
        
        self.misses += 1
        return False, None
    
    def insert(self, address: int, data: np.ndarray):
        """Insert data into cache using LRU replacement"""
        if not self.enabled:
            return
        
        set_index = address & ((1 << self.index_bits) - 1)
        tag = address >> self.index_bits
        
        cache_set = self.storage[set_index]
        
        # Check if already exists
        for i, (stored_tag, valid, _, _) in enumerate(cache_set):
            if valid and stored_tag == tag:
                cache_set[i] = (tag, True, data.copy(), self.lru_counter)
                self.lru_counter += 1
                return
        
        # Find empty way or evict LRU
        if len(cache_set) < self.ways:
            cache_set.append((tag, True, data.copy(), self.lru_counter))
        else:
            # Find LRU entry
            lru_idx = min(range(len(cache_set)), key=lambda i: cache_set[i][3])
            if cache_set[lru_idx][1]:  # if valid
                self.evictions += 1
            cache_set[lru_idx] = (tag, True, data.copy(), self.lru_counter)
        
        self.lru_counter += 1
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.enabled:
            return {'enabled': False}
        
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        occupied_ways = sum(len(s) for s in self.storage.values())
        utilization = occupied_ways / self.lines
        
        return {
            'enabled': True,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'utilization': utilization
        }


class CachedFixedPointSolver:
    """Fixed-point solver with adaptive two-level cache"""
    
    def __init__(self, verilog_path: Path, config: PDQAPConfig):
        self.verilog_path = verilog_path
        self.config = config
        
        parser = VerilogConfigParser()
        self.params = parser.parse_verilog(verilog_path)
        
        # Extract key parameters
        self.n_regions = self.params['N_REGIONS']
        self.n_solutions = self.params['N_SOLUTIONS']
        self.data_width = self.params['DATA_WIDTH']
        self.frac_bits = self.params['FRAC_BITS']
        
        # Auto-configure cache
        cache_cfg = AdaptiveCacheConfig(
            self.n_regions, self.n_solutions, self.data_width
        )
        self.l1_config = cache_cfg.l1_config
        self.l2_config = cache_cfg.l2_config
        
        # Initialize caches
        self.l1_cache = L1Cache(self.l1_config)
        self.l2_cache = L2Cache(self.l2_config)
        
        logger.info(f"L1 cache: {self.l1_config['lines']} lines (direct-mapped)")
        if self.l2_config['enabled']:
            logger.info(
                f"L2 cache: {self.l2_config['lines']} lines, "
                f"{self.l2_config['ways']}-way set-associative"
            )
    
    def solve(self, max_iterations: int = 100, tolerance: float = 1e-6) -> Dict:
        """Run fixed-point iteration with caching"""
        
        # Initialize solution vectors
        x = self._initialize_solution()
        x_history = [x.copy()]
        
        stats = {
            'iterations': 0,
            'converged': False,
            'convergence_history': []
        }
        
        for iteration in range(max_iterations):
            x_new = np.copy(x)
            
            # Update each region
            for region_id in range(self.n_regions):
                result = self._compute_with_cache(region_id, x)
                x_new[region_id] = result
            
            # Check convergence
            diff = np.linalg.norm(x_new - x)
            stats['convergence_history'].append(diff)
            
            x = x_new
            x_history.append(x.copy())
            stats['iterations'] = iteration + 1
            
            if diff < tolerance:
                stats['converged'] = True
                logger.info(f"Converged after {iteration + 1} iterations (diff={diff:.2e})")
                break
            
            if (iteration + 1) % 10 == 0:
                logger.info(f"Iteration {iteration + 1}: diff={diff:.2e}")
        
        # Collect final statistics
        stats['final_diff'] = diff
        stats['l1_stats'] = self.l1_cache.get_stats()
        stats['l2_stats'] = self.l2_cache.get_stats()
        
        # Overall hit rate
        total_l1 = stats['l1_stats']['hits'] + stats['l1_stats']['misses']
        if self.l2_config['enabled']:
            total_l2 = stats['l2_stats']['hits'] + stats['l2_stats']['misses']
            overall_hits = stats['l1_stats']['hits'] + stats['l2_stats']['hits']
            overall_total = total_l1
            stats['overall_hit_rate'] = overall_hits / overall_total if overall_total > 0 else 0
        else:
            stats['overall_hit_rate'] = stats['l1_stats']['hit_rate']
        
        return {
            'solution': x,
            'solution_history': x_history,
            'stats': stats,
            'l1_config': self.l1_config,
            'l2_config': self.l2_config
        }
    
    def _initialize_solution(self) -> np.ndarray:
        """Initialize solution vector with reasonable values"""
        # Use small random values around 0.5
        x = 0.5 + 0.1 * np.random.randn(self.n_regions, self.n_solutions)
        # Normalize to [0, 1]
        x = np.clip(x, 0, 1)
        return x
    
    def _compute_with_cache(self, region_id: int, x: np.ndarray) -> np.ndarray:
        """Compute region update using cache hierarchy"""
        
        # Generate address from region context
        address = self._compute_address(region_id, x)
        
        # Try L1 cache
        hit, data = self.l1_cache.access(address)
        if hit:
            return data
        
        # Try L2 cache
        hit, data = self.l2_cache.access(address)
        if hit:
            # Promote to L1
            self.l1_cache.insert(address, data)
            return data
        
        # Compute new value
        result = self._compute_region_update(region_id, x)
        
        # Insert into both caches
        self.l1_cache.insert(address, result)
        self.l2_cache.insert(address, result)
        
        return result
    
    def _compute_address(self, region_id: int, x: np.ndarray) -> int:
        """Compute cache address from region context"""
        # Simple hash: combine region_id with quantized neighbor values
        neighbors = self._get_neighbor_regions(region_id)
        
        addr = region_id
        for i, n in enumerate(neighbors[:3]):
            # Quantize neighbor values to reduce address space
            quantized = int(np.mean(x[n]) * 15)  # 4-bit quantization
            addr = (addr << 4) | quantized
        
        return addr
    
    def _get_neighbor_regions(self, region_id: int) -> List[int]:
        """Get neighboring regions based on topology"""
        # Simple 1D ring topology
        neighbors = [
            (region_id - 1) % self.n_regions,
            (region_id + 1) % self.n_regions
        ]
        
        # Add diagonal neighbors for 2D grid (if applicable)
        grid_size = int(math.sqrt(self.n_regions))
        if grid_size * grid_size == self.n_regions:
            row = region_id // grid_size
            col = region_id % grid_size
            
            if row > 0:
                neighbors.append((row - 1) * grid_size + col)
            if row < grid_size - 1:
                neighbors.append((row + 1) * grid_size + col)
        
        return neighbors
    
    def _compute_region_update(self, region_id: int, x: np.ndarray) -> np.ndarray:
        """Compute updated value for a region using BST dynamics"""
        neighbors = self._get_neighbor_regions(region_id)
        
        # BST-inspired update: weighted combination with nonlinearity
        result = np.zeros(self.n_solutions)
        
        # Self influence with decay
        alpha = 0.6
        result += alpha * x[region_id]
        
        # Neighbor influence
        beta = 0.4 / len(neighbors)
        for n in neighbors:
            result += beta * x[n]
        
        # Apply nonlinearity (tanh for smoothness)
        result = np.tanh(result)
        
        # Normalize to [0, 1]
        result = (result + 1) / 2
        
        return result
    
    def generate_cache_report(self, output_dir: Path, results: Dict):
        """Generate comprehensive cache performance report"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Text report
        report_path = output_dir / "cache_report.txt"
        self._write_text_report(report_path, results)
        
        # JSON report
        json_path = output_dir / "cache_stats.json"
        self._write_json_report(json_path, results)
        
        # Convergence plot data
        conv_path = output_dir / "convergence.csv"
        self._write_convergence_data(conv_path, results)
        
        logger.info(f"Reports saved to {output_dir}")
    
    def _write_text_report(self, path: Path, results: Dict):
        """Write human-readable text report"""
        stats = results['stats']
        
        with open(path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CACHED FIXED-POINT SOLVER - PERFORMANCE REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Problem configuration
            f.write("PROBLEM CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Regions:                {self.n_regions}\n")
            f.write(f"Solutions per region:   {self.n_solutions}\n")
            f.write(f"Data width:             {self.data_width} bits\n")
            f.write(f"Fractional bits:        {self.frac_bits}\n\n")
            
            # Cache configuration
            f.write("CACHE CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write("L1 Cache (On-chip SRAM):\n")
            f.write(f"  Lines:                {self.l1_config['lines']}\n")
            f.write(f"  Mapping:              {self.l1_config['mapping']}\n")
            f.write(f"  Index bits:           {self.l1_config['index_bits']}\n")
            f.write(f"  Tag bits:             {self.l1_config['tag_bits']}\n")
            f.write(f"  Bits per line:        {self.l1_config['bits_per_line']}\n")
            f.write(f"  Total bits:           {self.l1_config['total_bits']:,}\n")
            f.write(f"  Estimated area:       {self.l1_config['area_mm2']:.3f} mm²\n")
            f.write(f"  Estimated power:      {self.l1_config['power_mw']:.3f} mW\n\n")
            
            if self.l2_config['enabled']:
                f.write("L2 Cache (Off-chip DRAM):\n")
                f.write(f"  Lines:                {self.l2_config['lines']}\n")
                f.write(f"  Ways:                 {self.l2_config['ways']}\n")
                f.write(f"  Sets:                 {self.l2_config['sets']}\n")
                f.write(f"  Index bits:           {self.l2_config['index_bits']}\n")
                f.write(f"  Tag bits:             {self.l2_config['tag_bits']}\n")
                f.write(f"  Bits per line:        {self.l2_config['bits_per_line']}\n")
                f.write(f"  Total bits:           {self.l2_config['total_bits']:,}\n")
                f.write(f"  Estimated area:       {self.l2_config['area_mm2']:.3f} mm²\n")
                f.write(f"  Estimated power:      {self.l2_config['power_mw']:.3f} mW\n\n")
            else:
                f.write("L2 Cache:               Disabled\n\n")
            
            # Convergence results
            f.write("CONVERGENCE RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Iterations:             {stats['iterations']}\n")
            f.write(f"Converged:              {stats['converged']}\n")
            f.write(f"Final difference:       {stats['final_diff']:.6e}\n\n")
            
            # Cache performance
            f.write("CACHE PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            l1_stats = stats['l1_stats']
            f.write("L1 Cache:\n")
            f.write(f"  Hits:                 {l1_stats['hits']:,}\n")
            f.write(f"  Misses:               {l1_stats['misses']:,}\n")
            f.write(f"  Evictions:            {l1_stats['evictions']:,}\n")
            f.write(f"  Hit rate:             {l1_stats['hit_rate']:.2%}\n")
            f.write(f"  Utilization:          {l1_stats['utilization']:.2%}\n\n")
            
            if self.l2_config['enabled']:
                l2_stats = stats['l2_stats']
                f.write("L2 Cache:\n")
                f.write(f"  Hits:                 {l2_stats['hits']:,}\n")
                f.write(f"  Misses:               {l2_stats['misses']:,}\n")
                f.write(f"  Evictions:            {l2_stats['evictions']:,}\n")
                f.write(f"  Hit rate:             {l2_stats['hit_rate']:.2%}\n")
                f.write(f"  Utilization:          {l2_stats['utilization']:.2%}\n\n")
            
            f.write(f"Overall hit rate:       {stats['overall_hit_rate']:.2%}\n\n")
            
            # Performance summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 70 + "\n")
            total_accesses = l1_stats['hits'] + l1_stats['misses']
            f.write(f"Total memory accesses:  {total_accesses:,}\n")
            
            # Estimate latency savings
            l1_hit_cycles = l1_stats['hits'] * 1  # 1 cycle for L1 hit
            if self.l2_config['enabled']:
                l2_hit_cycles = stats['l2_stats']['hits'] * 10  # 10 cycles for L2 hit
                l2_miss_cycles = stats['l2_stats']['misses'] * 100  # 100 cycles for DRAM
                total_cycles = l1_hit_cycles + l2_hit_cycles + l2_miss_cycles
            else:
                l1_miss_cycles = l1_stats['misses'] * 100  # direct to DRAM
                total_cycles = l1_hit_cycles + l1_miss_cycles
            
            baseline_cycles = total_accesses * 100  # no cache
            speedup = baseline_cycles / total_cycles if total_cycles > 0 else 1
            
            f.write(f"Estimated cycles:       {total_cycles:,}\n")
            f.write(f"Baseline (no cache):    {baseline_cycles:,}\n")
            f.write(f"Speedup:                {speedup:.2f}x\n")
            
            f.write("\n" + "=" * 70 + "\n")
    
    def _write_json_report(self, path: Path, results: Dict):
        """Write machine-readable JSON report"""
        stats = results['stats']
        
        report = {
            'problem': {
                'n_regions': self.n_regions,
                'n_solutions': self.n_solutions,
                'data_width': self.data_width,
                'frac_bits': self.frac_bits
            },
            'cache_config': {
                'l1': self.l1_config,
                'l2': self.l2_config
            },
            'convergence': {
                'iterations': stats['iterations'],
                'converged': stats['converged'],
                'final_diff': float(stats['final_diff']),
                'history': [float(x) for x in stats['convergence_history']]
            },
            'performance': {
                'l1_stats': stats['l1_stats'],
                'l2_stats': stats['l2_stats'],
                'overall_hit_rate': stats['overall_hit_rate']
            }
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _write_convergence_data(self, path: Path, results: Dict):
        """Write convergence history as CSV"""
        history = results['stats']['convergence_history']
        
        with open(path, 'w') as f:
            f.write("iteration,difference\n")
            for i, diff in enumerate(history, 1):
                f.write(f"{i},{diff:.6e}\n")


def main():
    """Example usage and testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example paths
    verilog_path = Path("hardware/rtl/bst_solver_cached.sv")
    output_dir = Path("results/cached_solver")
    
    # Create config
    config = PDQAPConfig(
        n_regions=64,
        n_solutions=8,
        data_width=32,
        frac_bits=16
    )
    
    # Run solver
    logger.info("Initializing cached fixed-point solver...")
    solver = CachedFixedPointSolver(verilog_path, config)
    
    logger.info("Starting fixed-point iteration...")
    results = solver.solve(max_iterations=100, tolerance=1e-6)
    
    # Generate reports
    logger.info("Generating performance reports...")
    solver.generate_cache_report(output_dir, results)
    
    # Print summary
    stats = results['stats']
    logger.info(f"Converged: {stats['converged']}")
    logger.info(f"Iterations: {stats['iterations']}")
    logger.info(f"L1 hit rate: {stats['l1_stats']['hit_rate']:.2%}")
    if solver.l2_config['enabled']:
        logger.info(f"L2 hit rate: {stats['l2_stats']['hit_rate']:.2%}")
    logger.info(f"Overall hit rate: {stats['overall_hit_rate']:.2%}")


if __name__ == "__main__":
    main()