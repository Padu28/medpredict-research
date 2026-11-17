"""Configuration loading and project paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.getenv("MEDPREDICT_ROOT", Path(__file__).resolve().parents[3]))
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Settings:
    dataset: dict[str, Any]
    target_cfg: dict[str, Any]
    evaluation: dict[str, Any]

    @property
    def target(self) -> str:
        return self.target_cfg["target"]

    @property
    def numeric(self) -> list[str]:
        return list(self.target_cfg["numeric_features"])

    @property
    def categorical(self) -> list[str]:
        return list(self.target_cfg["categorical_features"])

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical

    @property
    def db_path(self) -> Path:
        o = os.getenv("MEDPREDICT_DB")
        return Path(o) if o else PROJECT_ROOT / self.evaluation["storage"]["database"]

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / self.evaluation["storage"]["reports_dir"]

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / self.evaluation["storage"]["models_dir"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(_load("dataset.yaml"), _load("target.yaml"), _load("evaluation.yaml"))
