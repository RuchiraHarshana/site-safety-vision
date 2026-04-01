#config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    model_path: str = "models/best.pt"
    confidence_threshold: float = 0.25
    device: Optional[str] = None
    image_size: Optional[int] = None
    tracker: str = "bytetrack.yaml"


@dataclass
class RulesConfig:
    unsafe_trigger_seconds: float = 5.0
    recent_memory_seconds: float = 3.0


@dataclass
class MatcherConfig:
    helmet_min_overlap: float = 0.05
    vest_min_overlap: float = 0.10
    gloves_min_overlap: float = 0.01
    boots_min_overlap: float = 0.01


@dataclass
class OutputConfig:
    output_dir: str = "outputs/predictions"
    save_video: bool = True
    save_json: bool = True


@dataclass
class AppConfig:
    model: ModelConfig
    rules: RulesConfig
    matcher: MatcherConfig
    output: OutputConfig


def load_yaml_config(config_path: str | Path) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in: {path}")

    return data


def load_app_config(config_path: str | Path) -> AppConfig:
    data = load_yaml_config(config_path)

    model_data = data.get("model", {})
    rules_data = data.get("rules", {})
    matcher_data = data.get("matcher", {})
    output_data = data.get("output", {})

    return AppConfig(
        model=ModelConfig(
            model_path=model_data.get("model_path", "models/best.pt"),
            confidence_threshold=model_data.get("confidence_threshold", 0.25),
            device=model_data.get("device"),
            image_size=model_data.get("image_size"),
            tracker=model_data.get("tracker", "bytetrack.yaml"),
        ),
        rules=RulesConfig(
            unsafe_trigger_seconds=rules_data.get("unsafe_trigger_seconds", 5.0),
            recent_memory_seconds=rules_data.get("recent_memory_seconds", 3.0),
        ),
        matcher=MatcherConfig(
            helmet_min_overlap=matcher_data.get("helmet_min_overlap", 0.05),
            vest_min_overlap=matcher_data.get("vest_min_overlap", 0.10),
            gloves_min_overlap=matcher_data.get("gloves_min_overlap", 0.01),
            boots_min_overlap=matcher_data.get("boots_min_overlap", 0.01),
        ),
        output=OutputConfig(
            output_dir=output_data.get("output_dir", "outputs/predictions"),
            save_video=output_data.get("save_video", True),
            save_json=output_data.get("save_json", True),
        ),
    )


def resolve_repo_path(*parts: str) -> Path:
    """
    Resolve a path relative to the repository root.

    Assumes this file lives in:
    src/site_safety_vision/config.py
    """
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.joinpath(*parts)