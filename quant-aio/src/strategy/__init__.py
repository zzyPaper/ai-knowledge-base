"""策略注册表 —— 通过名称获取策略实例。"""
from __future__ import annotations

from src.strategy.base import BaseStrategy, SectorScore
from src.strategy.v1_simple_momentum import V1SimpleMomentum
from src.strategy.v2_three_factor import V2ThreeFactor

# 策略注册表
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "v1": V1SimpleMomentum,
    "simple_momentum": V1SimpleMomentum,
    "v2": V2ThreeFactor,
    "three_factor": V2ThreeFactor,
}


def get_strategy(name: str = "v2") -> BaseStrategy:
    """根据名称获取策略实例。

    Parameters
    ----------
    name : 策略名称，可选 "v1", "simple_momentum", "v2", "three_factor"

    Returns
    -------
    BaseStrategy 实例
    """
    name_lower = name.lower().strip()
    if name_lower not in STRATEGY_REGISTRY:
        raise ValueError(
            f"未知策略 '{name}'，可选: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name_lower]()


def list_strategies() -> list[dict]:
    """列出所有可用策略。"""
    seen = set()
    result = []
    for key, cls in STRATEGY_REGISTRY.items():
        if cls.name not in seen:
            seen.add(cls.name)
            result.append({
                "key": key,
                "name": cls.name,
                "version": cls.version,
                "description": cls.description,
            })
    return result


__all__ = [
    "BaseStrategy",
    "SectorScore",
    "V1SimpleMomentum",
    "V2ThreeFactor",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "list_strategies",
]
