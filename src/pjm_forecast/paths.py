from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig


def ensure_project_directories(config: ProjectConfig) -> dict[str, Path]:
    directories = {}
    for key, relative_path in config.project["directories"].items():
        path = config.resolve_path(relative_path)
        path.mkdir(parents=True, exist_ok=True)
        directories[key] = path
    return directories

