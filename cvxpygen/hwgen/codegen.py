from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape
from numpy import concatenate, hstack, newaxis

from cvxpygen.hwgen.firmware_models import (
    ExecutionEnvironment,
)
from cvxpygen.hwgen.hw_models import PDAQPHWConfig
from cvxpygen.hwgen.models import PDAQPAlgoConfig

env = Environment(
    loader=PackageLoader("cvxpygen.hwgen"), autoescape=select_autoescape()
)


def generateHWAccCode(config: PDAQPHWConfig, target_dir: Path) -> Path:
    """Given the high-level hardware accelerator design for
    PDA-QP solver, generate the design in the Chisel language."""

    template = env.get_template("hwacc/Constants.scala.j2")
    rendered = template.render(config=config)

    outfile_path = target_dir / "Constants.scala"
    with open(outfile_path, "w") as f:
        f.write(rendered)

    return outfile_path


def generateTestbenchCode(
    config: PDAQPAlgoConfig,
    fixed_point_precision: int,
    execution_environment: ExecutionEnvironment,
    target_dir: Path,
) -> tuple[Path, Path]:
    """Given the high-level algorithm design for
    PDA-QP solver, generate the design in the Chisel language."""

    constants_hpp_path = target_dir / "constants.hpp"
    with open(constants_hpp_path, "w") as f:
        rendered = env.get_template("testbench/constants.hpp.j2").render(
            problem_size=config.problem_size,
            Q=fixed_point_precision,
        )
        f.write(rendered)

    problem_def_hpp_path = target_dir / "problem-def.hpp"
    with open(problem_def_hpp_path, "w") as f:
        # TODO (antonysigma): The shape of the halfplanes normal and
        # offset are erased. Modify the C++ project to preserve the shape.

        assert config.feedbacks.scale.shape[0] == config.feedbacks.offset.shape[0]
        assert config.feedbacks.scale.shape[1] == config.feedbacks.offset.shape[1]
        assert config.feedbacks.scale.shape[2] == config.problem_size.n_parameter

        rendered = env.get_template("testbench/problem-def.hpp.j2").render(
            half_planes=hstack(
                (config.half_planes.scale, config.half_planes.offset[:, newaxis])
            ),
            feedbacks=concatenate(
                (config.feedbacks.scale, config.feedbacks.offset[..., newaxis]),
                axis=-1,
            ),
            tree_nodes=config.tree_nodes,
        )
        f.write(rendered)

    return constants_hpp_path, problem_def_hpp_path
