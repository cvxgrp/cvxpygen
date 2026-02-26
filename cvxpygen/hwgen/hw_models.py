from dataclasses import dataclass

from numpy import uint16
from numpy.typing import NDArray

from cvxpygen.hwgen.models import MultiplyAddFix16, MultiplyAddFP32, ProblemSize


@dataclass
class TreeWalkerFSM:
    """A tree walking module implemented as finite state machine."""

    tree_nodes: NDArray[uint16]


@dataclass
class DotProductLeq:
    """Combined vector dot product with less-than-or-equal
    (LEQ) operation."""

    param: MultiplyAddFP32 | MultiplyAddFix16


@dataclass
class GEMV:
    """Combined matrix-vector multiplication and addition."""

    param: MultiplyAddFP32 | MultiplyAddFix16


@dataclass
class PDAQPHWConfig:
    """High-level architecture of the PDAQP arithmetic accelerator."""

    problem_size: ProblemSize

    tree_walker: TreeWalkerFSM
    half_planes_module: DotProductLeq
    feedback_module: GEMV

    @property
    def n_tree_nodes(self) -> int:
        return self.tree_walker.tree_nodes.size

    @property
    def n_half_planes(self) -> int:
        return self.half_planes_module.param.scale.shape[0]

    @property
    def n_feedbacks(self) -> int:
        return self.feedback_module.param.scale.shape[0]
