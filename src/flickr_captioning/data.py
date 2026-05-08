from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from flickr_captioning.config import DataConfig
from flickr_captioning.text import Vocabulary


def load_captions(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            image_ref, caption = line.split("\t", maxsplit=1)
            image_id = image_ref.split("#", maxsplit=1)[0]
            rows.append({"image_id": image_id, "caption": caption})
    return pd.DataFrame(rows)


def load_split(path: str | Path) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def split_captions(
    captions: pd.DataFrame,
    train_split: str | Path,
    val_split: str | Path,
    test_split: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ids = load_split(train_split)
    val_ids = load_split(val_split)
    test_ids = load_split(test_split)

    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Split leakage detected for image ids: {sample}")

    train_df = captions[captions["image_id"].isin(train_ids)].reset_index(drop=True)
    val_df = captions[captions["image_id"].isin(val_ids)].reset_index(drop=True)
    test_df = captions[captions["image_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def build_transforms(train: bool = False) -> Callable:
    ops: list[Callable] = [transforms.Resize((224, 224))]
    if train:
        ops.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(ops)


class FlickrCaptionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str | Path,
        vocabulary: Vocabulary,
        max_length: int,
        transform: Callable | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.transform = transform or build_transforms(train=False)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / row["image_id"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        caption_tensor = torch.tensor(
            self.vocabulary.encode(row["caption"], max_length=self.max_length),
            dtype=torch.long,
        )
        return image_tensor, caption_tensor, row["image_id"], row["caption"]


class CaptionCollator:
    def __init__(self, pad_idx: int) -> None:
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions, image_ids, raw_captions = zip(*batch)
        return {
            "images": torch.stack(images),
            "captions": pad_sequence(captions, batch_first=True, padding_value=self.pad_idx),
            "image_ids": list(image_ids),
            "raw_captions": list(raw_captions),
        }


def make_dataloaders(
    config: DataConfig,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    captions = load_captions(config.captions_file)
    train_df, val_df, test_df = split_captions(
        captions,
        config.train_split,
        config.val_split,
        config.test_split,
    )

    vocab = Vocabulary(min_freq=config.min_freq)
    vocab.fit(train_df["caption"].tolist())

    collate = CaptionCollator(vocab.pad_idx)
    train_dataset = FlickrCaptionDataset(
        train_df,
        config.image_dir,
        vocab,
        config.max_length,
        transform=build_transforms(train=True),
    )
    val_dataset = FlickrCaptionDataset(val_df, config.image_dir, vocab, config.max_length)
    test_dataset = FlickrCaptionDataset(test_df, config.image_dir, vocab, config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate,
    )
    return train_loader, val_loader, test_loader, vocab
