from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from PIL import Image
from tqdm import tqdm

from flickr_captioning.data import build_transforms, load_captions, load_split
from flickr_captioning.inference import beam_search_caption, load_checkpoint
from flickr_captioning.text import clean_caption
from flickr_captioning.utils import resolve_device


def _weights(n: int) -> tuple[float, float, float, float]:
    if n == 1:
        return (1.0, 0.0, 0.0, 0.0)
    if n == 2:
        return (0.5, 0.5, 0.0, 0.0)
    if n == 3:
        return (1 / 3, 1 / 3, 1 / 3, 0.0)
    return (0.25, 0.25, 0.25, 0.25)


@torch.no_grad()
def evaluate_bleu(
    checkpoint_path: str | Path,
    device_name: str = "auto",
    beam_size: int = 1,
    limit: int | None = None,
) -> dict[str, float]:
    device = resolve_device(device_name)
    encoder, decoder, vocabulary, config, model_kind = load_checkpoint(checkpoint_path, device)
    captions = load_captions(config.data.captions_file)
    test_ids = sorted(load_split(config.data.test_split))
    if limit is not None:
        test_ids = test_ids[:limit]

    references_by_image: dict[str, list[list[str]]] = defaultdict(list)
    for row in captions.itertuples(index=False):
        if row.image_id in test_ids:
            references_by_image[row.image_id].append(clean_caption(row.caption))

    transform = build_transforms(train=False)
    smoother = SmoothingFunction().method1
    totals = {f"BLEU-{n}": 0.0 for n in range(1, 5)}
    count = 0

    for image_id in tqdm(test_ids):
        image_path = config.data.image_dir / image_id
        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        caption = beam_search_caption(
            encoder,
            decoder,
            image,
            vocabulary,
            max_length=config.inference.max_length,
            beam_size=beam_size,
            model_kind=model_kind,
        )
        hypothesis = clean_caption(caption)
        references = references_by_image[image_id]
        for n in range(1, 5):
            totals[f"BLEU-{n}"] += sentence_bleu(
                references,
                hypothesis,
                weights=_weights(n),
                smoothing_function=smoother,
            )
        count += 1

    return {metric: value / max(count, 1) for metric, value in totals.items()}
