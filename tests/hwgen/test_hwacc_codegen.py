from pathlib import Path

from numpy import int16, uint16
from numpy.random import randint
from numpy.typing import NDArray

from cvxpygen.hwgen.codegen import generateHWAccCode
from cvxpygen.hwgen.hw_models import GEMV, DotProductLeq, PDAQPHWConfig, TreeWalkerFSM
from cvxpygen.hwgen.models import BinaryTree, MultiplyAddFix16, ProblemSize


def test_codegen() -> None:
    problem_size = ProblemSize(2, 3)
    n_tree_nodes = 5
    n_feedbacks = 3

    def generate_randint(size: tuple) -> NDArray[int16]:
        vmax = 20_000
        return randint(-vmax, vmax, size=size, dtype=int16).astype(int16)

    config = PDAQPHWConfig(
        problem_size,
        tree_walker=TreeWalkerFSM(
            BinaryTree(
                halfplane_or_feedback_id=randint(
                    0, n_tree_nodes, size=n_tree_nodes
                ).astype(uint16),
                jump=randint(0, n_tree_nodes, size=n_tree_nodes).astype(uint16),
            )
        ),
        half_planes_module=DotProductLeq(
            MultiplyAddFix16(
                generate_randint(
                    (n_tree_nodes - n_feedbacks, problem_size.n_parameter),
                ),
                generate_randint((n_tree_nodes - n_feedbacks,)),
            )
        ),
        feedback_module=GEMV(
            MultiplyAddFix16(
                generate_randint(
                    (n_feedbacks, problem_size.n_solution, problem_size.n_parameter),
                ),
                generate_randint(
                    (
                        n_feedbacks,
                        problem_size.n_solution,
                    ),
                ),
            )
        ),
    )

    outfile = generateHWAccCode(config, Path("/tmp"))
    assert outfile.exists()
