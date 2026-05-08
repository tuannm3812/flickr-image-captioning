from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from flickr_captioning.config import ProjectConfig
from flickr_captioning.data import make_dataloaders
from flickr_captioning.models import DecoderRNN, DecoderWithAttention, EncoderCNN, SpatialEncoderCNN
from flickr_captioning.text import Vocabulary
from flickr_captioning.utils import ensure_dir, resolve_device, seed_everything


ModelKind = Literal["baseline", "attention"]


def _make_models(config: ProjectConfig, vocab_size: int, model_kind: ModelKind):
    if model_kind == "baseline":
        encoder = EncoderCNN(config.model.embed_size, trainable=config.model.encoder_trainable)
        decoder = DecoderRNN(
            config.model.embed_size,
            config.model.hidden_size,
            vocab_size,
            dropout=config.model.dropout,
        )
    else:
        encoder = SpatialEncoderCNN(trainable=config.model.encoder_trainable)
        decoder = DecoderWithAttention(
            vocab_size,
            config.model.embed_size,
            config.model.hidden_size,
            config.model.attention_size,
            dropout=config.model.dropout,
        )
    return encoder, decoder


def _loss_for_batch(
    encoder: nn.Module,
    decoder: nn.Module,
    batch: dict,
    criterion: nn.Module,
    model_kind: ModelKind,
    device: torch.device,
) -> torch.Tensor:
    images = batch["images"].to(device)
    captions = batch["captions"].to(device)

    features = encoder(images)
    if model_kind == "baseline":
        logits = decoder(features, captions)
    else:
        logits, _ = decoder(features, captions)

    targets = captions[:, 1:]
    return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def _run_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    model_kind: ModelKind,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    encoder.train(is_train)
    decoder.train(is_train)
    losses: list[float] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_batch(encoder, decoder, batch, criterion, model_kind, device)
            if is_train:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(
    path: str | Path,
    encoder: nn.Module,
    decoder: nn.Module,
    vocabulary: Vocabulary,
    config: ProjectConfig,
    model_kind: ModelKind,
    val_loss: float,
) -> None:
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "vocabulary": vocabulary,
            "config": config,
            "model_kind": model_kind,
            "val_loss": val_loss,
        },
        path,
    )


def train(config: ProjectConfig, model_kind: ModelKind = "baseline") -> Path:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    train_loader, val_loader, _, vocabulary = make_dataloaders(
        config.data,
        batch_size=config.training.batch_size,
    )
    encoder, decoder = _make_models(config, len(vocabulary), model_kind)
    encoder.to(device)
    decoder.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_idx)
    trainable_parameters = [
        parameter
        for parameter in list(encoder.parameters()) + list(decoder.parameters())
        if parameter.requires_grad
    ]
    optimizer = Adam(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    checkpoint_dir = ensure_dir(config.training.checkpoint_dir)
    best_path = checkpoint_dir / f"best_{model_kind}.pt"
    best_val = float("inf")

    for epoch in range(1, config.training.epochs + 1):
        train_loss = _run_epoch(encoder, decoder, train_loader, criterion, optimizer, model_kind, device)
        val_loss = _run_epoch(encoder, decoder, val_loader, criterion, None, model_kind, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, encoder, decoder, vocabulary, config, model_kind, val_loss)

    return best_path
