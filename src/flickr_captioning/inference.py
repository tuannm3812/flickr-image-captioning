from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from flickr_captioning.data import build_transforms
from flickr_captioning.models import DecoderRNN, DecoderWithAttention, EncoderCNN, SpatialEncoderCNN
from flickr_captioning.text import Vocabulary
from flickr_captioning.utils import resolve_device


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vocabulary = checkpoint["vocabulary"]
    model_kind = checkpoint.get("model_kind", "baseline")

    if model_kind == "baseline":
        encoder = EncoderCNN(config.model.embed_size, trainable=False)
        decoder = DecoderRNN(
            config.model.embed_size,
            config.model.hidden_size,
            len(vocabulary),
            dropout=config.model.dropout,
        )
    else:
        encoder = SpatialEncoderCNN(trainable=False)
        decoder = DecoderWithAttention(
            len(vocabulary),
            config.model.embed_size,
            config.model.hidden_size,
            config.model.attention_size,
            dropout=config.model.dropout,
        )

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder, vocabulary, config, model_kind


def load_image(image_path: str | Path, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return build_transforms(train=False)(image).unsqueeze(0).to(device)


@torch.no_grad()
def greedy_caption(
    encoder,
    decoder,
    image: torch.Tensor,
    vocabulary: Vocabulary,
    max_length: int,
    model_kind: str,
) -> str:
    features = encoder(image)
    token = torch.tensor([vocabulary.start_idx], device=image.device)
    indices: list[int] = []

    if model_kind == "baseline":
        first_output, states = decoder.lstm(features.unsqueeze(1))
        first_logits = decoder.fc(first_output.squeeze(1))
        token = first_logits.argmax(dim=-1)
        indices.append(int(token.item()))
        for _ in range(max_length - 1):
            logits, states = decoder.step(token, states)
            token = logits.argmax(dim=-1)
            next_idx = int(token.item())
            indices.append(next_idx)
            if next_idx == vocabulary.end_idx:
                break
    else:
        hidden, cell = decoder.init_hidden_state(features)
        for _ in range(max_length):
            logits, hidden, cell, _ = decoder.step(features, token, hidden, cell)
            token = logits.argmax(dim=-1)
            next_idx = int(token.item())
            indices.append(next_idx)
            if next_idx == vocabulary.end_idx:
                break

    return " ".join(vocabulary.decode(indices))


@torch.no_grad()
def beam_search_caption(
    encoder,
    decoder,
    image: torch.Tensor,
    vocabulary: Vocabulary,
    max_length: int,
    beam_size: int,
    model_kind: str,
) -> str:
    if model_kind != "attention" or beam_size <= 1:
        return greedy_caption(encoder, decoder, image, vocabulary, max_length, model_kind)

    features = encoder(image)
    hidden, cell = decoder.init_hidden_state(features)
    beams = [([vocabulary.start_idx], 0.0, hidden, cell)]
    completed: list[tuple[list[int], float]] = []

    for _ in range(max_length):
        candidates = []
        for sequence, score, beam_hidden, beam_cell in beams:
            token = torch.tensor([sequence[-1]], device=image.device)
            logits, next_hidden, next_cell, _ = decoder.step(features, token, beam_hidden, beam_cell)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            values, indices = torch.topk(log_probs, beam_size)
            for value, index in zip(values.tolist(), indices.tolist()):
                next_sequence = [*sequence, index]
                next_score = score + float(value)
                if index == vocabulary.end_idx:
                    completed.append((next_sequence, next_score))
                else:
                    candidates.append((next_sequence, next_score, next_hidden, next_cell))
        if not candidates:
            break
        beams = sorted(candidates, key=lambda item: item[1] / len(item[0]), reverse=True)[:beam_size]

    if completed:
        best_sequence = max(completed, key=lambda item: item[1] / len(item[0]))[0]
    else:
        best_sequence = max(beams, key=lambda item: item[1] / len(item[0]))[0]
    return " ".join(vocabulary.decode(best_sequence[1:]))


@torch.no_grad()
def predict(
    checkpoint_path: str | Path,
    image_path: str | Path,
    device_name: str = "auto",
    max_length: int | None = None,
    beam_size: int = 1,
) -> str:
    device = resolve_device(device_name)
    encoder, decoder, vocabulary, config, model_kind = load_checkpoint(checkpoint_path, device)
    image = load_image(image_path, device)
    return beam_search_caption(
        encoder,
        decoder,
        image,
        vocabulary,
        max_length=max_length or config.inference.max_length,
        beam_size=beam_size,
        model_kind=model_kind,
    )
