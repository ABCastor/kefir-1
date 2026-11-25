"""Kefir mixture-of-experts model and training engine."""

from .accounting import MoEParameterCounts, count_moe_parameters
from .compile_lane import COMPILE_MODES, DEFAULT_COMPILE_MODE, apply_compile_lane
from .config import MoEModelConfig
from .engine import MoETrainerEngine
from .model import DroplessTopKMoE, MoESmallLLM
from .optimizer import build_moe_optimizer, classify_moe_parameters
from .router import SwitchTopKRouter

__all__ = [
    "COMPILE_MODES",
    "DEFAULT_COMPILE_MODE",
    "DroplessTopKMoE",
    "MoEModelConfig",
    "MoEParameterCounts",
    "MoESmallLLM",
    "MoETrainerEngine",
    "SwitchTopKRouter",
    "apply_compile_lane",
    "build_moe_optimizer",
    "classify_moe_parameters",
    "count_moe_parameters",
]
