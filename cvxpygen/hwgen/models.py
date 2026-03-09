from dataclasses import dataclass
from enum import Enum

from numpy import float32, int16, uint16
from numpy.typing import NDArray


@dataclass
class ProblemSize:
    n_parameter: int
    n_solution: int


@dataclass
class IntArray:
    name: str
    data: NDArray[int16]


@dataclass
class FloatArray:
    name: str
    data: NDArray[float32]


class DataFormat(Enum):
    FP32 = 0
    FIXQ2_14 = 1


@dataclass
class MultiplyAddFP32:
    scale: NDArray[float32]
    offset: NDArray[float32]


@dataclass
class MultiplyAddFix16:
    scale: NDArray[int16]
    offset: NDArray[int16]
    Q: int = 14


@dataclass
class PDAQPAlgoConfig:
    problem_size: ProblemSize
    tree_nodes: NDArray[uint16]
    half_planes: MultiplyAddFP32
    feedbacks: MultiplyAddFP32

    @property
    def n_tree_nodes(self) -> int:
        return self.tree_nodes.size

    @property
    def n_half_planes(self) -> int:
        return self.half_planes.scale.shape[0]

    @property
    def n_feedbacks(self) -> int:
        return self.feedbacks.scale.shape[0]
