from pathlib import Path

from numpy import float32, uint16
from numpy.random import randint, randn
from numpy.typing import NDArray

from cvxpygen.hwgen.codegen import generateTestbenchCode
from cvxpygen.hwgen.firmware_models import LINUX_AMD64
from cvxpygen.hwgen.models import (
    BinaryTree,
    MultiplyAddFP32,
    PDAQPAlgoConfig,
    ProblemSize,
)


def test_codegen() -> None:
    problem_size = ProblemSize(2, 3)
    n_tree_nodes = 5
    n_feedbacks = 3

    def generate_rand(size: tuple) -> NDArray[float32]:
        vmax = 20_000
        return randn(*size).astype(float32)

    config = PDAQPAlgoConfig(
        problem_size,
        tree_nodes=BinaryTree(
            halfplane_or_feedback_id=randint(0, n_tree_nodes, size=n_tree_nodes).astype(
                uint16
            ),
            jump=randint(0, n_tree_nodes, size=n_tree_nodes).astype(uint16),
        ),
        half_planes=MultiplyAddFP32(
            generate_rand(
                (n_tree_nodes - n_feedbacks, problem_size.n_parameter),
            ),
            generate_rand((n_tree_nodes - n_feedbacks,)),
        ),
        feedbacks=MultiplyAddFP32(
            generate_rand(
                (n_feedbacks, problem_size.n_solution, problem_size.n_parameter),
            ),
            generate_rand(
                (
                    n_feedbacks,
                    problem_size.n_solution,
                ),
            ),
        ),
    )

    fixed_point_precision = 14
    constants_hpp, problem_def_hpp = generateTestbenchCode(
        config, fixed_point_precision, LINUX_AMD64, Path("/tmp")
    )
    assert constants_hpp.exists()
    assert problem_def_hpp.exists()
