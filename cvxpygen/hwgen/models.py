from dataclasses import dataclass
from enum import Enum

from numpy import float32, int16
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


class FloatFormat(Enum):
    FP32 = 0
    FP16 = 1
    FIX16 = 2


@dataclass
class PDAQPConfig:
    problem_size: ProblemSize
    tree_nodes: IntArray
    half_planes: FloatArray
    feedbacks: FloatArray
    data_format: FloatFormat = FloatFormat.FIX16
