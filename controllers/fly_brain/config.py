"""Configuration loading.

Every gain, threshold and geometry lives in config/default.yaml. This module
loads it once and exposes it as attribute-accessible nested objects, so call
sites read `cfg.avoidance.turn_gain` rather than `cfg["avoidance"]["turn_gain"]`.

Rationale for a hard failure on a missing key: a typo in a gain name should
stop the drone at startup, not silently fall back to a default and produce
flight behaviour nobody can explain.
"""

import os
from typing import Any, Dict, Iterator

import yaml

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "default.yaml",
)


class ConfigNode:
    """Attribute access over a nested mapping, with useful failures."""

    __slots__ = ("_data", "_path")

    def __init__(self, data: Dict[str, Any], path: str = "") -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        path = object.__getattribute__(self, "_path")

        if name not in data:
            where = f"{path}.{name}" if path else name
            available = ", ".join(sorted(data.keys()))
            raise AttributeError(
                f"no config key '{where}'. Available here: {available}"
            )

        value = data[name]
        if isinstance(value, dict):
            return ConfigNode(value, f"{path}.{name}" if path else name)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "config is read-only at runtime — change config/default.yaml, or "
            "use load_config(overrides=...) for a test"
        )

    def __contains__(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_data")

    def __iter__(self) -> Iterator[str]:
        return iter(object.__getattribute__(self, "_data"))

    def get(self, name: str, default: Any = None) -> Any:
        """Dict-style access for genuinely optional keys."""
        data = object.__getattribute__(self, "_data")
        value = data.get(name, default)
        if isinstance(value, dict):
            return ConfigNode(value, name)
        return value

    def as_dict(self) -> Dict[str, Any]:
        return dict(object.__getattribute__(self, "_data"))

    def __repr__(self) -> str:
        path = object.__getattribute__(self, "_path") or "<root>"
        keys = ",".join(sorted(object.__getattribute__(self, "_data").keys()))
        return f"ConfigNode({path}: {keys})"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `override` into `base`, recursing into nested dicts."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str = None, overrides: Dict[str, Any] = None) -> ConfigNode:
    """Load configuration from YAML.

    Args:
        path: config file; defaults to config/default.yaml at the repo root.
        overrides: nested dict merged over the file, for tests that need to
            vary one gain without editing the shared config.

    Returns:
        ConfigNode supporting attribute access.
    """
    resolved = path or _DEFAULT_PATH
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"config not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if overrides:
        data = _deep_merge(data, overrides)

    return ConfigNode(data)


_cached: ConfigNode = None


def get_config() -> ConfigNode:
    """Process-wide config, loaded once.

    The control loop must not touch the filesystem per timestep.
    """
    global _cached
    if _cached is None:
        _cached = load_config()
    return _cached


def reset_config() -> None:
    """Drop the cache. For tests that load a modified config."""
    global _cached
    _cached = None
