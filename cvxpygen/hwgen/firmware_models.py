from dataclasses import dataclass
from enum import Enum


class OperatingSystem(Enum):
    Linux = 0
    NoOS = 1


class HardwareCapability(Enum):
    BranchPrediction = 0
    SIMDInstructions = 1
    InstructionCache = 2


@dataclass
class ExecutionEnvironment:
    os: OperatingSystem
    cpu_bits: int
    hardware_capabilities: list[HardwareCapability]


LINUX_AMD64 = ExecutionEnvironment(
    os=OperatingSystem.Linux,
    cpu_bits=64,
    hardware_capabilities=[
        HardwareCapability.SIMDInstructions,
        HardwareCapability.BranchPrediction,
        HardwareCapability.InstructionCache,
    ],
)

ARM64 = ExecutionEnvironment(
    os=OperatingSystem.NoOS,
    cpu_bits=32,
    hardware_capabilities=[
        HardwareCapability.SIMDInstructions,
        HardwareCapability.BranchPrediction,
    ],
)

ATMETA8 = ExecutionEnvironment(
    os=OperatingSystem.NoOS,
    cpu_bits=8,
    hardware_capabilities=[],
)
