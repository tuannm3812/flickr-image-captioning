from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    root: Path
    image_dir: Path
    captions_file: Path
    train_split: Path
    val_split: Path
    test_split: Path
    min_freq: int = 5
    max_length: int = 21
    num_workers: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    checkpoint_dir: Path = Path("models")
    log_every: int = 50


@dataclass(frozen=True)
class ModelConfig:
    embed_size: int = 256
    hidden_size: int = 512
    attention_size: int = 256
    dropout: float = 0.3
    encoder_trainable: bool = False


@dataclass(frozen=True)
class InferenceConfig:
    max_length: int = 21
    beam_size: int = 3


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    device: str
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    inference: InferenceConfig


def _pathify(section: dict[str, Any], *keys: str) -> dict[str, Any]:
    copied = dict(section)
    for key in keys:
        copied[key] = Path(copied[key])
    return copied


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return ProjectConfig(
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        data=DataConfig(
            **_pathify(
                raw["data"],
                "root",
                "image_dir",
                "captions_file",
                "train_split",
                "val_split",
                "test_split",
            )
        ),
        training=TrainingConfig(
            **_pathify(raw.get("training", {}), "checkpoint_dir")
        ),
        model=ModelConfig(**raw.get("model", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
    )
