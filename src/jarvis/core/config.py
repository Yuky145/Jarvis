"""Configuration loading.

Loads ``config/config.yaml`` once and exposes it as a nested dictionary with
convenient dotted-path access. Paths in the config are resolved relative to the
project root so scripts work regardless of the current working directory.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/jarvis/core/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """Thin wrapper around the parsed YAML config."""

    def __init__(self, data: dict[str, Any], root: Path = PROJECT_ROOT):
        self._data = data
        self.root = root

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a value using ``"a.b.c"`` dotted notation."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def path(self, dotted_key: str, default: str | None = None) -> Path:
        """Resolve a config value that is a path, relative to project root."""
        value = self.get(dotted_key, default)
        if value is None:
            raise KeyError(f"No path configured for '{dotted_key}'")
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    @property
    def data(self) -> dict[str, Any]:
        return self._data


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load and cache the project configuration."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data)
