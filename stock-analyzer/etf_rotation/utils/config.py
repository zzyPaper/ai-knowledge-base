"""配置管理 - 借鉴QLib的YAML配置思路"""
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """轻量级配置管理"""

    def __init__(self, config_path: str = None):
        self._config: Dict[str, Any] = {}
        if config_path:
            self.load(config_path)

    def load(self, path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """支持点分隔的key: config.get('data.source')"""
        keys = key.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
                if val is None:
                    return default
            else:
                return default
        return val

    def set(self, key: str, value: Any):
        keys = key.split(".")
        target = self._config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    @property
    def raw(self) -> dict:
        return self._config

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
