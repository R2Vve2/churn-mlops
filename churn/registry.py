"""A lightweight, file-based model registry.

Each trained model is stored under ``models/<version>/`` with the serialized pipeline and a
``metadata.json`` (metrics, threshold, feature list, data fingerprint, drift reference).
``models/latest.json`` points at the version to serve. MLflow keeps the full experiment
history; this registry is what the serving container ships with.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from churn.config import MODEL_DIR


class ModelRegistry:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or MODEL_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_version(self) -> str:
        return "v" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    def save(self, pipeline: Any, metadata: dict[str, Any], version: str | None = None, promote: bool = True) -> str:
        version = version or self.new_version()
        folder = self.root / version
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, folder / "model.joblib")
        metadata = {**metadata, "version": version}
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
        if promote:
            self.promote(version)
        return version

    def promote(self, version: str) -> None:
        if not (self.root / version / "model.joblib").exists():
            raise FileNotFoundError(f"no model artifact for version {version}")
        (self.root / "latest.json").write_text(json.dumps({"version": version}), encoding="utf-8")

    def latest_version(self) -> str | None:
        pointer = self.root / "latest.json"
        if not pointer.exists():
            return None
        return json.loads(pointer.read_text(encoding="utf-8"))["version"]

    def list_versions(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and (p / "metadata.json").exists())

    def load(self, version: str | None = None) -> tuple[Any, dict[str, Any]]:
        version = version or self.latest_version()
        if version is None:
            raise FileNotFoundError(f"no model registered under {self.root}; run `python -m churn.training.train`")
        folder = self.root / version
        pipeline = joblib.load(folder / "model.joblib")
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        return pipeline, metadata
