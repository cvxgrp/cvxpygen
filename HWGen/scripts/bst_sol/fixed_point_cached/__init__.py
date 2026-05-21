# scripts/bst_sol/fixed_point_cached/__init__.py

"""
Fixed-Point Cached BST Solver Package
======================================

Provides cached BST-based hardware solver with fixed-point arithmetic.

Main Components:
    - CachedFixedPointSolver: Main solver class with cache support
    - AdaptiveCacheConfig: Auto cache configuration
    - L1Cache, L2Cache: Cache implementations

Example:
    >>> from scripts.bst_sol.fixed_point_cached import CachedFixedPointSolver
    >>> from pathlib import Path
    >>> solver = CachedFixedPointSolver(Path('design.sv'), config)
    >>> results = solver.solve()
"""

from .solver_cached import (
    CachedFixedPointSolver,
    AdaptiveCacheConfig,
    L1Cache,
    L2Cache
)

__all__ = [
    'CachedFixedPointSolver',
    'AdaptiveCacheConfig',
    'L1Cache',
    'L2Cache',
]

__version__ = '1.0.0'