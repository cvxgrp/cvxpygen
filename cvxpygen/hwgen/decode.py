from pathlib import Path

from cvxpygen.hwgen.c_parsing.array_visitor import FloatArrayVisitor, IntArrayVisitor
from cvxpygen.hwgen.c_parsing.grammars import pdaqp_c_grammar, pdaqp_header_grammar
from cvxpygen.hwgen.c_parsing.header_visitor import HeaderVisitor
from cvxpygen.hwgen.models import PDAQPConfig


def decode(c_path: Path, h_path: Path) -> PDAQPConfig:
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
    assert "pdaqp_halfplanes" in float_arrays.arrays
    assert "pdaqp_feedbacks" in float_arrays.arrays

    return PDAQPConfig(
        header_visitor.problem_size,
        tree_nodes=int_arrays.arrays["pdaqp_hp_list"],
        half_planes=float_arrays.arrays["pdaqp_halfplanes"],
        feedbacks=float_arrays.arrays["pdaqp_feedbacks"],
    )
