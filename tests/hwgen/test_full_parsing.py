from pathlib import Path

from cvxpygen.hwgen.decode import decode


def test_full_parsing() -> None:
    prefix = Path("test-vectors/pid")
    assert decode(prefix / "pdaqp.c", prefix / "pdaqp.h")
