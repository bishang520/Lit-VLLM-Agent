"""轻量配置加载：YAML 文件 + 字典覆盖，路径支持相对项目根。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，返回 dict。"""
    if yaml is None:
        raise RuntimeError(
            "缺少 PyYAML，请先安装：pip install -r requirements/requirements-base.txt"
        )
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 YAML 映射: {p}")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个 dict，override 优先。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(dict(out[k]), dict(v))
        else:
            out[k] = v
    return out


def load_config(
    path: str | Path, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    cfg = load_yaml(path)
    if overrides:
        cfg = deep_merge(cfg, dict(overrides))
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """把配置中的相对路径解析为相对项目根。"""
    p = Path(cfg[key])
    return p if p.is_absolute() else PROJECT_ROOT / p
