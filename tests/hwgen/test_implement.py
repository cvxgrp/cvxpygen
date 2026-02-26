from pathlib import Path

from pytest import fixture

from cvxpygen.hwgen.decode import decode
from cvxpygen.hwgen.implement import quantize, specialize
from cvxpygen.hwgen.models import DataFormat, MultiplyAddFix16, PDAQPAlgoConfig


@fixture
def algo_config() -> PDAQPAlgoConfig:
    prefix = Path("test-vectors/pid")
    return decode(prefix / "pdaqp.c", prefix / "pdaqp.h")


def test_specialize(algo_config: PDAQPAlgoConfig) -> None:
    assert specialize(algo_config)


def test_quantize(algo_config: PDAQPAlgoConfig) -> None:
    hw_config = specialize(algo_config)

    hw_config_fix16 = quantize(hw_config, DataFormat.FIXQ2_14)
    assert isinstance(hw_config_fix16.feedback_module.param, MultiplyAddFix16)
    assert isinstance(hw_config_fix16.half_planes_module.param, MultiplyAddFix16)
