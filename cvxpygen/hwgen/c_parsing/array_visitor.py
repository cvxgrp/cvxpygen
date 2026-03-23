from numpy import array, float32, int16
from parsimonious.nodes import NodeVisitor

from cvxpygen.hwgen.models import FloatArray, IntArray


class IntArrayVisitor(NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.arrays: dict[str, IntArray] = {}

    def visit_int_array(self, node, visited_children) -> None:
        _, name, _, _, items, _ = visited_children
        self.arrays[name] = IntArray(name, array(items, dtype=int16))

    def visit_int_item(self, node, visited_children) -> int:
        return visited_children[1]

    def visit_name(self, node, _) -> str:
        return node.text

    def visit_digits(self, node, _) -> int:
        return int(node.text)

    def generic_visit(self, node, visited_children):
        return visited_children or node


class FloatArrayVisitor(NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.arrays: dict[str, FloatArray] = {}

    def visit_float_array(self, node, visited_children) -> None:
        _, name, _, _, items, _ = visited_children
        self.arrays[name] = FloatArray(name, array(items, dtype=float32))

    def visit_float_item(self, node, visited_children) -> float:
        return visited_children[1]

    def visit_name(self, node, _) -> str:
        return node.text

    def visit_decimal(self, node, _) -> float:
        return float(node.text)

    def generic_visit(self, node, visited_children):
        return visited_children or node
