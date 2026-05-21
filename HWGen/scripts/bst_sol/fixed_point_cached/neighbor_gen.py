# scripts/bst_sol/fixed_point_cached/neighbor_gen.py
"""
Neighbor Adjacency Generation
Offline analysis to generate neighbor relationships for cache prefetching
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import logging

from ..common.config_parser import PDQAPConfig


@dataclass
class NeighborInfo:
    """Neighbor information for a region"""
    region_id: int
    hp_idx: int
    neighbors: List[int]  # List of neighbor hp_idx
    neighbor_count: int
    confidence: List[float]  # Confidence score for each neighbor (0-1)
    
    def __str__(self):
        return f"Region {self.region_id} (hp_idx={self.hp_idx}): {self.neighbor_count} neighbors"


class NeighborGenerator:
    """Generate neighbor adjacency tables for cache prefetching"""
    
    def __init__(self, config: PDQAPConfig, max_neighbors: int = 4):
        self.config = config
        self.max_neighbors = max_neighbors
        self.logger = logging.getLogger(__name__)
        
        # Load tree structure
        self.tree_data = self._load_tree_structure()
        
        # Load halfplane definitions for geometric analysis
        self.halfplanes = self._load_halfplanes()
        
        # Region mapping
        self.n_regions = None  # Will be set by _build_region_mapping
        self.hp_idx_to_region = {}  # hp_idx -> region_id
        self.region_to_hp_idx = {}  # region_id -> hp_idx
        self.region_to_node = {}    # region_id -> tree node
        
        self._build_region_mapping()
        
        # Build decision path for each region
        self.region_paths = self._build_region_paths()
    
    def _load_tree_structure(self) -> Dict:
        """Load BST structure from pickle file"""
        pkl_path = self.config.get_output_path(f"{self.config.project_name}_bst.pkl")
        
        if not pkl_path.exists():
            raise FileNotFoundError(f"BST pickle not found: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            tree_data = pickle.load(f)
        
        self.logger.info(f"Loaded tree structure: {len(tree_data['nodes'])} nodes")
        return tree_data
    
    def _load_halfplanes(self) -> Optional[np.ndarray]:
        """Load halfplane definitions (A matrix and b vector)"""
        try:
            # Try to load from memory files
            include_dir = self.config.get_output_path("../include")
            hp_list_path = include_dir / f"{self.config.project_name}_hp_list.mem"
            
            if not hp_list_path.exists():
                self.logger.warning("Halfplane list file not found, geometric analysis disabled")
                return None
            
            # Parse hp_list.mem (format: hex values for A and b)
            # This is simplified - actual parsing depends on your format
            self.logger.info("Halfplane definitions loaded for geometric analysis")
            return None  # Placeholder - implement actual parsing if needed
            
        except Exception as e:
            self.logger.warning(f"Failed to load halfplanes: {e}")
            return None
    
    def _build_region_mapping(self):
        """Build mapping between region_id and hp_idx"""
        # Find all leaf nodes
        leaves = [node for node in self.tree_data['nodes'] if node['is_leaf']]
        
        self.logger.info(f"Found {len(leaves)} leaf nodes (regions)")
        
        # Sort by hp_idx to get consistent region_id ordering
        leaves_sorted = sorted(leaves, key=lambda x: x['hp_idx'])
        
        for region_id, leaf in enumerate(leaves_sorted):
            hp_idx = leaf['hp_idx']
            self.hp_idx_to_region[hp_idx] = region_id
            self.region_to_hp_idx[region_id] = hp_idx
            self.region_to_node[region_id] = leaf
        
        self.n_regions = len(leaves)
        self.logger.info(f"Built mapping for {self.n_regions} regions")
    
    def _build_region_paths(self) -> Dict[int, List[Tuple[int, bool]]]:
        """
        Build decision path for each region
        Returns dict: region_id -> [(node_id, went_left), ...]
        """
        paths = {}
        
        for region_id in range(self.n_regions):
            node = self.region_to_node[region_id]
            node_id = node['node_id']
            
            # Trace path from root to this leaf
            path = []
            current_id = node_id
            
            while current_id > 1:  # Stop at root (node_id=1)
                parent_id = current_id // 2
                went_left = (current_id == parent_id * 2)
                path.append((parent_id, went_left))
                current_id = parent_id
            
            paths[region_id] = list(reversed(path))  # Root to leaf order
        
        return paths
    
    def _compute_path_similarity(self, path1: List[Tuple[int, bool]], 
                                 path2: List[Tuple[int, bool]]) -> float:
        """
        Compute similarity between two decision paths
        Returns score 0-1 (1 = most similar, likely neighbors)
        """
        # Find longest common prefix
        common_length = 0
        for i in range(min(len(path1), len(path2))):
            if path1[i] == path2[i]:
                common_length += 1
            else:
                break
        
        # Regions that diverge late in the tree are more likely neighbors
        max_depth = max(len(path1), len(path2))
        if max_depth == 0:
            return 0.0
        
        # Score based on when paths diverge
        similarity = common_length / max_depth
        
        # Bonus: if they diverge at the last step (siblings)
        if common_length == max_depth - 1:
            similarity += 0.5
        
        return min(similarity, 1.0)
    
    def generate_neighbors(self) -> Dict[int, NeighborInfo]:
        """
        Generate neighbor relationships using multiple methods
        
        Returns:
            Dict mapping region_id -> NeighborInfo
        """
        self.logger.info("Generating neighbor adjacency...")
        
        neighbor_map = {}
        
        for region_id in range(self.n_regions):
            hp_idx = self.region_to_hp_idx[region_id]
            
            # Collect candidate neighbors with confidence scores
            candidates = {}  # hp_idx -> confidence score
            
            # Method 1: Tree structure (siblings and cousins)
            tree_neighbors = self._get_tree_neighbors(region_id)
            for neighbor_hp_idx, confidence in tree_neighbors.items():
                candidates[neighbor_hp_idx] = max(
                    candidates.get(neighbor_hp_idx, 0.0), 
                    confidence
                )
            
            # Method 2: Path similarity
            path_neighbors = self._get_path_neighbors(region_id)
            for neighbor_hp_idx, confidence in path_neighbors.items():
                candidates[neighbor_hp_idx] = max(
                    candidates.get(neighbor_hp_idx, 0.0),
                    confidence * 0.8  # Slightly lower weight
                )
            
            # Method 3: Sequential hp_idx (fallback)
            seq_neighbors = self._get_sequential_neighbors(hp_idx)
            for neighbor_hp_idx in seq_neighbors:
                if neighbor_hp_idx not in candidates:
                    candidates[neighbor_hp_idx] = 0.3  # Low confidence
            
            # Method 4: Centroid distance (if available)
            centroid_neighbors = self._get_centroid_neighbors(hp_idx)
            for neighbor_hp_idx, confidence in centroid_neighbors.items():
                candidates[neighbor_hp_idx] = max(
                    candidates.get(neighbor_hp_idx, 0.0),
                    confidence * 0.7
                )
            
            # Remove self if present
            candidates.pop(hp_idx, None)
            
            # Sort by confidence and take top N
            sorted_candidates = sorted(
                candidates.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.max_neighbors]
            
            neighbors_list = [hp for hp, _ in sorted_candidates]
            confidence_list = [conf for _, conf in sorted_candidates]
            
            neighbor_map[region_id] = NeighborInfo(
                region_id=region_id,
                hp_idx=hp_idx,
                neighbors=neighbors_list,
                neighbor_count=len(neighbors_list),
                confidence=confidence_list
            )
            
            self.logger.debug(
                f"Region {region_id}: {len(neighbors_list)} neighbors "
                f"(conf: {[f'{c:.2f}' for c in confidence_list]})"
            )
        
        self._print_statistics(neighbor_map)
        return neighbor_map
    
    def _get_tree_neighbors(self, region_id: int) -> Dict[int, float]:
        """
        Method 1: Find neighbors via tree structure
        Returns dict: hp_idx -> confidence (0-1)
        """
        neighbors = {}
        
        node = self.region_to_node[region_id]
        node_id = node['node_id']
        
        # Find sibling
        parent_id = node_id // 2
        if parent_id > 0:
            sibling_id = node_id + 1 if node_id % 2 == 0 else node_id - 1
            
            # Check if sibling exists and is a leaf
            for other_node in self.tree_data['nodes']:
                if (other_node['node_id'] == sibling_id and 
                    other_node['is_leaf']):
                    neighbors[other_node['hp_idx']] = 1.0  # High confidence
                    break
                elif other_node['node_id'] == sibling_id:
                    # Sibling is internal node, find its leaf children
                    leaf_descendants = self._get_leaf_descendants(sibling_id)
                    for leaf_hp in leaf_descendants:
                        neighbors[leaf_hp] = 0.8  # Medium-high confidence
        
        # Find cousins (children of parent's sibling)
        if parent_id > 1:
            grandparent_id = parent_id // 2
            parent_sibling_id = parent_id + 1 if parent_id % 2 == 0 else parent_id - 1
            
            # Find leaves under parent's sibling
            cousin_leaves = self._get_leaf_descendants(parent_sibling_id)
            for leaf_hp in cousin_leaves:
                if leaf_hp not in neighbors:
                    neighbors[leaf_hp] = 0.6  # Medium confidence
        
        return neighbors
    
    def _get_leaf_descendants(self, node_id: int) -> List[int]:
        """Get all leaf hp_idx under a given node"""
        leaves = []
        
        # BFS to find all descendants
        queue = [node_id]
        
        while queue:
            current_id = queue.pop(0)
            
            # Find node
            current_node = None
            for node in self.tree_data['nodes']:
                if node['node_id'] == current_id:
                    current_node = node
                    break
            
            if current_node is None:
                continue
            
            if current_node['is_leaf']:
                leaves.append(current_node['hp_idx'])
            else:
                # Add children
                left_child = current_id * 2
                right_child = current_id * 2 + 1
                queue.extend([left_child, right_child])
        
        return leaves
    
    def _get_path_neighbors(self, region_id: int) -> Dict[int, float]:
        """
        Method 2: Find neighbors based on path similarity
        Returns dict: hp_idx -> confidence
        """
        neighbors = {}
        my_path = self.region_paths[region_id]
        
        # Compare with all other regions
        similarities = []
        for other_id in range(self.n_regions):
            if other_id == region_id:
                continue
            
            other_path = self.region_paths[other_id]
            similarity = self._compute_path_similarity(my_path, other_path)
            
            if similarity > 0.5:  # Only consider reasonably similar paths
                other_hp = self.region_to_hp_idx[other_id]
                similarities.append((other_hp, similarity))
        
        # Take top candidates
        similarities.sort(key=lambda x: x[1], reverse=True)
        for hp_idx, sim in similarities[:self.max_neighbors * 2]:
            neighbors[hp_idx] = sim
        
        return neighbors
    
    def _get_sequential_neighbors(self, hp_idx: int) -> Set[int]:
        """
        Method 3: Sequential neighbors in hp_idx space
        Fallback method when geometric analysis unavailable
        """
        neighbors = set()
        
        # Check adjacent hp_idx values
        for offset in [-1, 1, -2, 2]:
            candidate = hp_idx + offset
            if candidate in self.hp_idx_to_region:
                neighbors.add(candidate)
        
        return neighbors
    
    def _get_centroid_neighbors(self, hp_idx: int) -> Dict[int, float]:
        """
        Method 4: Find neighbors by centroid distance in parameter space
        Returns dict: hp_idx -> confidence
        """
        neighbors = {}
        
        # Load region sample points if available
        samples_path = self.config.get_output_path(
            f"{self.config.project_name}_samples.pkl"
        )
        
        if not samples_path.exists():
            return neighbors
        
        try:
            with open(samples_path, 'rb') as f:
                samples_data = pickle.load(f)
            
            # Compute centroids
            centroids = {}
            for region_hp_idx, points in samples_data.items():
                if len(points) > 0:
                    centroids[region_hp_idx] = np.mean(points, axis=0)
            
            if hp_idx not in centroids:
                return neighbors
            
            # Find K nearest centroids
            my_centroid = centroids[hp_idx]
            distances = []
            
            for other_hp_idx, other_centroid in centroids.items():
                if other_hp_idx != hp_idx:
                    dist = np.linalg.norm(my_centroid - other_centroid)
                    distances.append((dist, other_hp_idx))
            
            # Sort by distance
            distances.sort()
            
            # Convert distance to confidence (closer = higher confidence)
            max_dist = distances[-1][0] if distances else 1.0
            for dist, neighbor_hp_idx in distances[:self.max_neighbors * 2]:
                # Normalize distance to 0-1 confidence
                confidence = 1.0 - (dist / max_dist)
                neighbors[neighbor_hp_idx] = confidence
        
        except Exception as e:
            self.logger.warning(f"Centroid method failed: {e}")
        
        return neighbors
    
    def _print_statistics(self, neighbor_map: Dict[int, NeighborInfo]):
        """Print neighbor statistics"""
        counts = [info.neighbor_count for info in neighbor_map.values()]
        avg_confidence = np.mean([
            np.mean(info.confidence) if info.confidence else 0.0
            for info in neighbor_map.values()
        ])
        
        self.logger.info("=== Neighbor Statistics ===")
        self.logger.info(f"Total regions: {len(neighbor_map)}")
        self.logger.info(f"Avg neighbors: {np.mean(counts):.2f}")
        self.logger.info(f"Min neighbors: {np.min(counts)}")
        self.logger.info(f"Max neighbors: {np.max(counts)}")
        self.logger.info(f"Avg confidence: {avg_confidence:.3f}")
        
        # Distribution
        for n in range(self.max_neighbors + 1):
            count = sum(1 for c in counts if c == n)
            pct = count / len(counts) * 100 if counts else 0
            self.logger.info(f"  {n} neighbors: {count} regions ({pct:.1f}%)")
    
    def export_memory_files(self, neighbor_map: Dict[int, NeighborInfo]):
        """
        Export neighbor tables to .mem files for Verilog
        
        Generates:
        - <project>_neighbor_list.mem: [region_id][slot] -> neighbor_hp_idx
        - <project>_neighbor_count.mem: [region_id] -> count
        """
        output_dir = self.config.get_output_path("../include")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        project = self.config.project_name
        
        # ========== neighbor_list.mem ==========
        list_path = output_dir / f"{project}_neighbor_list.mem"
        with open(list_path, 'w') as f:
            f.write("// Neighbor list: [region_id * max_neighbors + slot] -> neighbor_hp_idx\n")
            f.write(f"// Max neighbors per region: {self.max_neighbors}\n")
            f.write(f"// Total regions: {self.n_regions}\n\n")
            
            for region_id in range(self.n_regions):
                info = neighbor_map[region_id]
                
                # Write all slots (pad with 0 if fewer neighbors)
                for slot in range(self.max_neighbors):
                    if slot < info.neighbor_count:
                        neighbor_hp_idx = info.neighbors[slot]
                        f.write(f"{neighbor_hp_idx:04X}\n")
                    else:
                        f.write("0000\n")  # Padding
        
        self.logger.info(f"Exported: {list_path}")
        
        # ========== neighbor_count.mem ==========
        count_path = output_dir / f"{project}_neighbor_count.mem"
        with open(count_path, 'w') as f:
            f.write("// Neighbor count: [region_id] -> number of neighbors\n\n")
            
            for region_id in range(self.n_regions):
                info = neighbor_map[region_id]
                f.write(f"{info.neighbor_count:01X}\n")
        
        self.logger.info(f"Exported: {count_path}")
        
        # ========== Debug info file ==========
        debug_path = output_dir / f"{project}_neighbors_debug.txt"
        with open(debug_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Neighbor Adjacency Table (for debugging)\n")
            f.write("=" * 70 + "\n\n")
            
            for region_id in range(self.n_regions):
                info = neighbor_map[region_id]
                f.write(f"Region {region_id:3d} (hp_idx={info.hp_idx:4d}): ")
                f.write(f"{info.neighbor_count} neighbors\n")
                
                for i, (neighbor_hp, conf) in enumerate(zip(info.neighbors, info.confidence)):
                    neighbor_region = self.hp_idx_to_region.get(neighbor_hp, -1)
                    f.write(f"  [{i}] hp_idx={neighbor_hp:4d} (region={neighbor_region:3d}) ")
                    f.write(f"confidence={conf:.3f}\n")
                
                f.write("\n")
        
        self.logger.info(f"Exported debug info: {debug_path}")


def generate_neighbor_tables(config: PDQAPConfig, max_neighbors: int = 4):
    """
    Main entry point for neighbor table generation
    
    Args:
        config: PDQAP configuration
        max_neighbors: Maximum neighbors per region
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    gen = NeighborGenerator(config, max_neighbors)
    neighbor_map = gen.generate_neighbors()
    gen.export_memory_files(neighbor_map)
    
    return neighbor_map


# ========== Testing Utilities ==========

def verify_neighbor_tables(config: PDQAPConfig, max_neighbors: int = 4):
    """Verify generated neighbor tables are valid"""
    project = config.project_name
    include_dir = config.get_output_path("../include")
    
    list_path = include_dir / f"{project}_neighbor_list.mem"
    count_path = include_dir / f"{project}_neighbor_count.mem"
    
    if not list_path.exists() or not count_path.exists():
        print("❌ Neighbor table files not found")
        return False
    
    # Read files (skip comment lines)
    with open(list_path, 'r') as f:
        list_data = [int(line.strip(), 16) for line in f 
                     if line.strip() and not line.strip().startswith('//')]
    
    with open(count_path, 'r') as f:
        count_data = [int(line.strip(), 16) for line in f 
                      if line.strip() and not line.strip().startswith('//')]
    
    # Determine n_regions from count data
    n_regions = len(count_data)
    
    expected_list_entries = n_regions * max_neighbors
    
    print(f"📊 Verification Results:")
    print(f"   Regions: {n_regions}")
    print(f"   Max neighbors: {max_neighbors}")
    print(f"   List entries: {len(list_data)} (expected {expected_list_entries})")
    print(f"   Count entries: {len(count_data)}")
    
    if len(list_data) != expected_list_entries:
        print(f"❌ List size mismatch")
        return False
    
    # Verify counts match list
    errors = 0
    for region_id in range(n_regions):
        count = count_data[region_id]
        
        if count > max_neighbors:
            print(f"❌ Region {region_id}: count {count} exceeds max {max_neighbors}")
            errors += 1
            continue
        
        # Check that first 'count' entries are non-zero
        base = region_id * max_neighbors
        for i in range(count):
            if list_data[base + i] == 0:
                print(f"⚠️  Region {region_id}: slot {i} is zero but count={count}")
                errors += 1
    
    if errors > 0:
        print(f"❌ Found {errors} errors")
        return False
    
    print("✅ Neighbor tables verified successfully")
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python neighbor_gen.py <config.json>")
        sys.exit(1)
    
    config = PDQAPConfig.from_json(sys.argv[1])
    neighbor_map = generate_neighbor_tables(config)
    
    # Verify
    if verify_neighbor_tables(config):
        print("\n✅ Neighbor generation completed successfully")
    else:
        print("\n❌ Verification failed")
        sys.exit(1)