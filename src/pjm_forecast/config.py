from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProjectConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def project(self) -> dict[str, Any]:
        return self.raw["project"]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw["backtest"]

    @property
    def tuning(self) -> dict[str, Any]:
        return self.raw["tuning"]

    @property
    def models(self) -> dict[str, Any]:
        return self.raw["models"]

    @property
    def report(self) -> dict[str, Any]:
        return self.raw["report"]

    def resolve_path(self, relative_path: str) -> Path:
        override = os.environ.get("PJM_PROJECT_ROOT_OVERRIDE") or self.project.get("root_override")
        base_path = Path(override).resolve() if override else self.path.parent.parent
        return (base_path / relative_path).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ProjectConfig(raw=raw, path=config_path)
