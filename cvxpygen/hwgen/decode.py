from pathlib import Path

from numpy import uint16

from cvxpygen.hwgen.c_parsing.array_visitor import FloatArrayVisitor, IntArrayVisitor
from cvxpygen.hwgen.c_parsing.grammars import pdaqp_c_grammar, pdaqp_header_grammar
from cvxpygen.hwgen.c_parsing.header_visitor import HeaderVisitor
from cvxpygen.hwgen.models import MultiplyAddFP32, PDAQPAlgoConfig


def decode(c_path: Path, h_path: Path) -> PDAQPAlgoConfig:
    header_visitor = HeaderVisitor()
    header_visitor.grammar = pdaqp_header_grammar

    with open(h_path, "r") as h_file:
        header_visitor.parse(h_file.read())

    with open(c_path, "r") as c_file:
        ast = pdaqp_c_grammar.parse(c_file.read())

    int_arrays = IntArrayVisitor()
    int_arrays.visit(ast)
    assert "pdaqp_hp_list" in int_arrays.arrays

    float_arrays = FloatArrayVisitor()
    float_arrays.visit(ast)

    n_parameter = header_visitor.problem_size.n_parameter
    n_solution = header_visitor.problem_size.n_solution

    assert "pdaqp_halfplanes" in float_arrays.arrays
    assert float_arrays.arrays["pdaqp_halfplanes"].data.size % (n_parameter + 1) == 0
    halfplanes = float_arrays.arrays["pdaqp_halfplanes"].data.reshape(
        -1, n_parameter + 1
    )

    assert "pdaqp_feedbacks" in float_arrays.arrays
    assert (
        float_arrays.arrays["pdaqp_feedbacks"].data.size
        % ((n_parameter + 1) * n_solution)
        == 0
    )
    feedbacks = float_arrays.arrays["pdaqp_feedbacks"].data.reshape(
        -1, n_parameter + 1, n_solution
    )

    assert "pdaqp_hp_list" in int_arrays.arrays
    assert int_arrays.arrays["pdaqp_hp_list"].data.max() <= 65535

    return PDAQPAlgoConfig(
        header_visitor.problem_size,
        tree_nodes=int_arrays.arrays["pdaqp_hp_list"].data.astype(uint16),
        half_planes=MultiplyAddFP32(
            halfplanes[:, :n_parameter], halfplanes[:, n_parameter]
        ),
        feedbacks=MultiplyAddFP32(
            feedbacks[:, :n_parameter], feedbacks[:, n_parameter]
        ),
    )
