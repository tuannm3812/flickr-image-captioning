from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


cells = [
    md(
        """
# Flickr8k Image Captioning: ResNet Attention Improvement Notebook

This Kaggle notebook is the next experiment after the VGG16 baseline and VGG16 attention runs.

It tests a stronger image-captioning setup:

- ResNet50 spatial visual encoder instead of VGG16.
- Additive attention decoder over image regions.
- Mixed precision training on CUDA.
- Learning-rate scheduling on validation loss.
- Beam search with length normalization and repetition penalty.
- Exported checkpoints, BLEU scores, prediction CSV, inference image grid, and run metadata.

The notebook is self-contained so it can run on Kaggle without installing this repository.
"""
    ),
    md(
        """
## Runtime Configuration

Attach a Flickr8k Kaggle dataset before running. The path finder supports common Flickr8k layouts and falls back to a deterministic image-level 80/10/10 split if official split files are missing.
"""
    ),
    code(
        """
from pathlib import Path

RUN_TRAINING = True
RUN_EVALUATION = True
RUN_SAMPLE_INFERENCE = True

MODEL_KIND = "resnet_attention"
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 1e-3
MIN_FREQ = 5
MAX_LEN = 24
BEAM_SIZE = 5
LENGTH_ALPHA = 0.7
REPETITION_PENALTY = 1.15

# Kaggle notebooks are more stable with workers disabled.
NUM_WORKERS = 0
SEED = 42

# Use small values for quick smoke tests; use None for full experiments.
TRAIN_LIMIT = None
EVAL_LIMIT = 500

IS_KAGGLE = Path("/kaggle/input").exists()
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("outputs/resnet_attention")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Optional GitHub artifact publishing. Store GITHUB_TOKEN in Kaggle Secrets.
PUBLISH_TO_GITHUB = False
GITHUB_REPO = "tuannm3812/flickr-image-captioning"
GITHUB_BASE_BRANCH = "main"
GITHUB_TARGET_BRANCH = "kaggle-resnet-attention-artifacts"
GITHUB_COMMIT_NAME = "Kaggle Notebook"
GITHUB_COMMIT_EMAIL = "actions@users.noreply.github.com"

WORK_DIR
"""
    ),
    md("## Imports and Reproducibility"),
    code(
        r"""
import json
import math
import random
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm.auto import tqdm

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


seed_everything(SEED)
device = resolve_device()
USE_AMP = device.type == "cuda"
print("Device:", device)
print("AMP:", USE_AMP)
print("Work dir:", WORK_DIR)
"""
    ),
    md("## Locate Flickr8k Files"),
    code(
        r"""
def find_first(root: Path, names: list[str]) -> Optional[Path]:
    if not root.exists():
        return None
    targets = {name.lower() for name in names}
    for path in root.rglob("*"):
        if path.name.lower() in targets:
            return path
    return None


def find_image_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = []
    for path in root.rglob("*"):
        if path.is_dir():
            jpg_count = len(list(path.glob("*.jpg")))
            if jpg_count > 100:
                priority = 0 if path.name.lower() in {"flickr8k_dataset", "images", "flickr8k_images"} else 1
                candidates.append((priority, -jpg_count, path))
    return sorted(candidates)[0][2] if candidates else None


SEARCH_ROOT = Path("/kaggle/input") if IS_KAGGLE else Path("data/raw/flickr8k")

CAPTIONS_FILE = find_first(SEARCH_ROOT, ["Flickr8k.token.txt", "captions.txt"])
TRAIN_SPLIT = find_first(SEARCH_ROOT, ["Flickr_8k.trainImages.txt", "trainImages.txt"])
VAL_SPLIT = find_first(SEARCH_ROOT, ["Flickr_8k.devImages.txt", "devImages.txt", "valImages.txt"])
TEST_SPLIT = find_first(SEARCH_ROOT, ["Flickr_8k.testImages.txt", "testImages.txt"])
IMAGE_DIR = find_image_dir(SEARCH_ROOT)

print("Search root:", SEARCH_ROOT)
print("Image dir:", IMAGE_DIR)
print("Captions:", CAPTIONS_FILE)
print("Train split:", TRAIN_SPLIT)
print("Val split:", VAL_SPLIT)
print("Test split:", TEST_SPLIT)

if IMAGE_DIR is None or CAPTIONS_FILE is None:
    raise FileNotFoundError("Could not locate Flickr8k image directory and captions file.")
if TRAIN_SPLIT is None or VAL_SPLIT is None or TEST_SPLIT is None:
    print("Official split files not found. Using deterministic image-level 80/10/10 split.")
"""
    ),
    md("## Captions, Vocabulary, and Splits"),
    code(
        r"""
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
TOKEN_PATTERN = re.compile(r"[^a-z\s]")


def clean_caption(caption: str) -> list[str]:
    caption = caption.lower()
    caption = TOKEN_PATTERN.sub(" ", caption)
    return [token for token in caption.split() if len(token) > 1 or token == "a"]


def prepare_caption(caption: str) -> list[str]:
    return [START_TOKEN, *clean_caption(caption), END_TOKEN]


class Vocabulary:
    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.stoi = {}
        self.itos = []
        for token in [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]:
            self.add_token(token)

    def __len__(self):
        return len(self.itos)

    @property
    def pad_idx(self):
        return self.stoi[PAD_TOKEN]

    @property
    def unk_idx(self):
        return self.stoi[UNK_TOKEN]

    @property
    def start_idx(self):
        return self.stoi[START_TOKEN]

    @property
    def end_idx(self):
        return self.stoi[END_TOKEN]

    def add_token(self, token: str) -> int:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def fit(self, captions: list[str]) -> None:
        counts = Counter()
        for caption in captions:
            counts.update(clean_caption(caption))
        for token, count in sorted(counts.items()):
            if count >= self.min_freq:
                self.add_token(token)

    def encode(self, caption: str, max_length: Optional[int] = None) -> list[int]:
        tokens = prepare_caption(caption)
        if max_length is not None:
            tokens = tokens[:max_length]
            if tokens[-1] != END_TOKEN:
                tokens[-1] = END_TOKEN
        return [self.stoi.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: list[int], skip_special: bool = True) -> list[str]:
        special = {PAD_TOKEN, START_TOKEN, END_TOKEN} if skip_special else set()
        words = []
        for idx in indices:
            token = self.itos[idx] if 0 <= idx < len(self.itos) else UNK_TOKEN
            if skip_special and token == END_TOKEN:
                break
            if token not in special:
                words.append(token)
        return words

    def to_dict(self):
        return {"min_freq": self.min_freq, "itos": self.itos, "stoi": self.stoi}

    @classmethod
    def from_dict(cls, data):
        vocab = cls(min_freq=data.get("min_freq", 1))
        vocab.itos = list(data["itos"])
        vocab.stoi = dict(data["stoi"])
        return vocab


def load_captions(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        handle.seek(0)
        if "," in first and "caption" in first.lower() and "image" in first.lower():
            df = pd.read_csv(path)
            image_col = "image" if "image" in df.columns else "image_id"
            return pd.DataFrame({"image_id": df[image_col], "caption": df["caption"]})
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                image_ref, caption = line.split("\t", maxsplit=1)
            elif "," in line:
                image_ref, caption = line.split(",", maxsplit=1)
            else:
                continue
            rows.append({"image_id": image_ref.split("#", maxsplit=1)[0], "caption": caption})
    return pd.DataFrame(rows)


def load_split(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def split_captions(captions: pd.DataFrame):
    if TRAIN_SPLIT is not None and VAL_SPLIT is not None and TEST_SPLIT is not None:
        train_ids = load_split(TRAIN_SPLIT)
        val_ids = load_split(VAL_SPLIT)
        test_ids = load_split(TEST_SPLIT)
    else:
        image_ids = captions.image_id.drop_duplicates().tolist()
        rng = random.Random(SEED)
        rng.shuffle(image_ids)
        n_total = len(image_ids)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        train_ids = set(image_ids[:n_train])
        val_ids = set(image_ids[n_train:n_train + n_val])
        test_ids = set(image_ids[n_train + n_val:])

    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise ValueError(f"Split leakage detected: {sorted(overlap)[:5]}")

    train_df = captions[captions.image_id.isin(train_ids)].reset_index(drop=True)
    val_df = captions[captions.image_id.isin(val_ids)].reset_index(drop=True)
    test_df = captions[captions.image_id.isin(test_ids)].reset_index(drop=True)
    if TRAIN_LIMIT is not None:
        train_df = train_df.iloc[:TRAIN_LIMIT].reset_index(drop=True)
        val_df = val_df.iloc[: max(32, TRAIN_LIMIT // 5)].reset_index(drop=True)
    return train_df, val_df, test_df


captions_df = load_captions(CAPTIONS_FILE)
train_df, val_df, test_df = split_captions(captions_df)
vocab = Vocabulary(min_freq=MIN_FREQ)
vocab.fit(train_df.caption.tolist())

print("Captions:", len(captions_df))
print("Unique images:", captions_df.image_id.nunique())
print("Train/val/test rows:", len(train_df), len(val_df), len(test_df))
print("Vocabulary size:", len(vocab))
train_df.head()
"""
    ),
    md("## DataLoaders"),
    code(
        r"""
def build_transforms(train: bool = False):
    ops = [transforms.Resize((256, 256))]
    if train:
        ops.extend([
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    else:
        ops.append(transforms.CenterCrop((224, 224)))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)


class FlickrCaptionDataset(Dataset):
    def __init__(self, dataframe, image_dir: Path, vocabulary: Vocabulary, max_length: int, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.transform = transform or build_transforms(train=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(self.image_dir / row.image_id).convert("RGB")
        image = self.transform(image)
        caption = torch.tensor(self.vocabulary.encode(row.caption, self.max_length), dtype=torch.long)
        return image, caption, row.image_id, row.caption


class CaptionCollator:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions, image_ids, raw_captions = zip(*batch)
        return {
            "images": torch.stack(images),
            "captions": pad_sequence(captions, batch_first=True, padding_value=self.pad_idx),
            "image_ids": list(image_ids),
            "raw_captions": list(raw_captions),
        }


collate = CaptionCollator(vocab.pad_idx)
loader_kwargs = {
    "num_workers": NUM_WORKERS,
    "collate_fn": collate,
    "pin_memory": torch.cuda.is_available(),
}
if NUM_WORKERS > 0:
    loader_kwargs["persistent_workers"] = True

train_loader = DataLoader(
    FlickrCaptionDataset(train_df, IMAGE_DIR, vocab, MAX_LEN, build_transforms(train=True)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    **loader_kwargs,
)
val_loader = DataLoader(
    FlickrCaptionDataset(val_df, IMAGE_DIR, vocab, MAX_LEN, build_transforms(train=False)),
    batch_size=BATCH_SIZE,
    shuffle=False,
    **loader_kwargs,
)
test_loader = DataLoader(
    FlickrCaptionDataset(test_df, IMAGE_DIR, vocab, MAX_LEN, build_transforms(train=False)),
    batch_size=BATCH_SIZE,
    shuffle=False,
    **loader_kwargs,
)

batch = next(iter(train_loader))
print("Image batch:", tuple(batch["images"].shape))
print("Caption batch:", tuple(batch["captions"].shape))
"""
    ),
    md("## ResNet50 Spatial Encoder and Attention Decoder"),
    code(
        r"""
def load_resnet50():
    try:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        print("Loaded ResNet50 ImageNet weights.")
    except Exception as exc:
        print("Could not load pretrained ResNet50 weights; using random initialization.")
        print(type(exc).__name__, exc)
        model = resnet50(weights=None)
    return model


class ResNetSpatialEncoder(nn.Module):
    def __init__(self, trainable: bool = False):
        super().__init__()
        backbone = load_resnet50()
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        for parameter in self.features.parameters():
            parameter.requires_grad = trainable

    def forward(self, images):
        maps = self.features(images)  # [B, 2048, 7, 7]
        maps = maps.permute(0, 2, 3, 1)
        return maps.reshape(maps.size(0), -1, maps.size(-1))


class AdditiveAttention(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int, attention_size: int):
        super().__init__()
        self.feature_att = nn.Linear(feature_size, attention_size)
        self.hidden_att = nn.Linear(hidden_size, attention_size)
        self.full_att = nn.Linear(attention_size, 1)

    def forward(self, features, hidden):
        scores = self.full_att(torch.tanh(self.feature_att(features) + self.hidden_att(hidden).unsqueeze(1))).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = (features * alpha.unsqueeze(-1)).sum(dim=1)
        return context, alpha


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        attention_size: int = 256,
        feature_size: int = 2048,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.attention = AdditiveAttention(feature_size, hidden_size, attention_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.init_h = nn.Linear(feature_size, hidden_size)
        self.init_c = nn.Linear(feature_size, hidden_size)
        self.lstm_cell = nn.LSTMCell(embed_size + feature_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, features):
        mean_features = features.mean(dim=1)
        return self.init_h(mean_features), self.init_c(mean_features)

    def forward(self, features, captions):
        hidden, cell = self.init_hidden_state(features)
        embeddings = self.dropout(self.embed(captions[:, :-1]))
        predictions = []
        alphas = []
        for t in range(captions.size(1) - 1):
            context, alpha = self.attention(features, hidden)
            hidden, cell = self.lstm_cell(torch.cat([embeddings[:, t], context], dim=1), (hidden, cell))
            predictions.append(self.fc(self.dropout(hidden)))
            alphas.append(alpha)
        return torch.stack(predictions, dim=1), torch.stack(alphas, dim=1)

    def step(self, features, token, hidden, cell):
        context, alpha = self.attention(features, hidden)
        embedding = self.embed(token)
        hidden, cell = self.lstm_cell(torch.cat([embedding, context], dim=1), (hidden, cell))
        return self.fc(hidden), hidden, cell, alpha


encoder = ResNetSpatialEncoder(trainable=False).to(device)
decoder = DecoderWithAttention(vocab_size=len(vocab)).to(device)
print(type(encoder).__name__, type(decoder).__name__)
"""
    ),
    md("## Training with AMP, Scheduler, and Checkpointing"),
    code(
        r"""
def batch_loss(encoder, decoder, batch, criterion):
    images = batch["images"].to(device, non_blocking=True)
    captions = batch["captions"].to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", enabled=USE_AMP):
        features = encoder(images)
        logits, _ = decoder(features, captions)
        targets = captions[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return loss


def run_epoch(encoder, decoder, loader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    encoder.train(is_train)
    decoder.train(is_train)
    losses = []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(encoder, decoder, batch, criterion)
            if is_train:
                if scaler is not None and USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
                    optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


def save_checkpoint(path: Path, history: list[dict], best_val: float):
    torch.save(
        {
            "model_kind": MODEL_KIND,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "vocab": vocab.to_dict(),
            "settings": {
                "max_len": MAX_LEN,
                "min_freq": MIN_FREQ,
                "beam_size": BEAM_SIZE,
                "length_alpha": LENGTH_ALPHA,
                "repetition_penalty": REPETITION_PENALTY,
            },
            "history": history,
            "best_val_loss": best_val,
        },
        path,
    )


def train_model():
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    parameters = [p for p in list(encoder.parameters()) + list(decoder.parameters()) if p.requires_grad]
    optimizer = AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    checkpoint_path = WORK_DIR / "best_resnet_attention.pt"
    history = []
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = run_epoch(encoder, decoder, train_loader, criterion, optimizer, scaler)
        val_loss = run_epoch(encoder, decoder, val_loader, criterion)
        scheduler.step(val_loss)
        improved = val_loss < (best_val - EARLY_STOPPING_MIN_DELTA)
        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"best {best_val:.4f} | wait {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} | "
            f"lr {row['lr']:.2e} | {row['seconds']:.1f}s"
        )
        if improved:
            save_checkpoint(checkpoint_path, history, best_val)
            print(f"Saved checkpoint: {checkpoint_path}")
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}: validation loss did not improve by {EARLY_STOPPING_MIN_DELTA} for {EARLY_STOPPING_PATIENCE} epochs.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(WORK_DIR / "training_history.csv", index=False)
    return checkpoint_path, history_df


if RUN_TRAINING:
    checkpoint_path, history_df = train_model()
else:
    checkpoint_path = WORK_DIR / "best_resnet_attention.pt"
    history_df = pd.DataFrame()

checkpoint_path
"""
    ),
    md("## Beam Search with Length Normalization and Repetition Penalty"),
    code(
        r"""
def sequence_score(log_prob: float, length: int, alpha: float) -> float:
    return log_prob / (((5 + length) / 6) ** alpha)


@torch.no_grad()
def beam_search_caption(encoder, decoder, image, vocab: Vocabulary, max_length: int, beam_size: int):
    encoder.eval()
    decoder.eval()
    features = encoder(image)
    hidden, cell = decoder.init_hidden_state(features)
    beams = [([vocab.start_idx], 0.0, hidden, cell)]
    completed = []

    for _ in range(max_length):
        candidates = []
        for sequence, score, beam_hidden, beam_cell in beams:
            token = torch.tensor([sequence[-1]], device=device)
            logits, next_hidden, next_cell, _ = decoder.step(features, token, beam_hidden, beam_cell)

            # Penalize immediate repetition and repeated content words.
            for seen_idx in set(sequence[-4:]):
                logits[0, seen_idx] /= REPETITION_PENALTY

            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            values, ids = torch.topk(log_probs, beam_size)
            for value, idx in zip(values.tolist(), ids.tolist()):
                new_sequence = [*sequence, idx]
                new_score = score + float(value)
                if idx == vocab.end_idx:
                    completed.append((new_sequence, new_score))
                else:
                    candidates.append((new_sequence, new_score, next_hidden, next_cell))
        if not candidates:
            break
        beams = sorted(
            candidates,
            key=lambda item: sequence_score(item[1], len(item[0]), LENGTH_ALPHA),
            reverse=True,
        )[:beam_size]

    best = max(
        completed or [(seq, score) for seq, score, _, _ in beams],
        key=lambda item: sequence_score(item[1], len(item[0]), LENGTH_ALPHA),
    )[0]
    return " ".join(vocab.decode(best[1:]))


def load_best_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location=device)
    loaded_vocab = Vocabulary.from_dict(checkpoint["vocab"])
    loaded_encoder = ResNetSpatialEncoder(trainable=False).to(device)
    loaded_decoder = DecoderWithAttention(vocab_size=len(loaded_vocab)).to(device)
    loaded_encoder.load_state_dict(checkpoint["encoder_state_dict"])
    loaded_decoder.load_state_dict(checkpoint["decoder_state_dict"])
    loaded_encoder.eval()
    loaded_decoder.eval()
    return loaded_encoder, loaded_decoder, loaded_vocab, checkpoint


if checkpoint_path.exists():
    encoder, decoder, vocab, checkpoint = load_best_checkpoint(checkpoint_path)
    print("Loaded:", checkpoint_path)
"""
    ),
    md("## Evaluation and Prediction Export"),
    code(
        r"""
def bleu_weights(n: int):
    if n == 1:
        return (1.0, 0.0, 0.0, 0.0)
    if n == 2:
        return (0.5, 0.5, 0.0, 0.0)
    if n == 3:
        return (1 / 3, 1 / 3, 1 / 3, 0.0)
    return (0.25, 0.25, 0.25, 0.25)


def build_references(captions: pd.DataFrame, image_ids: set[str]):
    references = defaultdict(list)
    for row in captions.itertuples(index=False):
        if row.image_id in image_ids:
            references[row.image_id].append(clean_caption(row.caption))
    return references


def evaluate_and_save(limit: Optional[int] = None):
    test_ids = list(dict.fromkeys(test_df.image_id.tolist()))
    if limit is not None:
        test_ids = test_ids[:limit]
    references = build_references(captions_df, set(test_ids))
    transform = build_transforms(train=False)
    smoother = SmoothingFunction().method1
    totals = {f"BLEU-{n}": 0.0 for n in range(1, 5)}
    rows = []

    for image_id in tqdm(test_ids):
        image = Image.open(IMAGE_DIR / image_id).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        generated = beam_search_caption(encoder, decoder, image_tensor, vocab, MAX_LEN, BEAM_SIZE)
        hypothesis = clean_caption(generated)
        refs = references[image_id]
        for n in range(1, 5):
            totals[f"BLEU-{n}"] += sentence_bleu(
                refs,
                hypothesis,
                weights=bleu_weights(n),
                smoothing_function=smoother,
            )
        rows.append({
            "image_id": image_id,
            "generated_caption": generated,
            "reference_1": " ".join(refs[0]) if refs else "",
            "all_references": json.dumps([" ".join(ref) for ref in refs]),
        })

    scores = {metric: value / max(len(test_ids), 1) for metric, value in totals.items()}
    predictions_df = pd.DataFrame(rows)
    predictions_df.to_csv(WORK_DIR / "sample_predictions.csv", index=False)
    (WORK_DIR / "bleu_scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return scores, predictions_df


if RUN_EVALUATION and checkpoint_path.exists():
    bleu_scores, predictions_df = evaluate_and_save(limit=EVAL_LIMIT)
    print(json.dumps(bleu_scores, indent=2))
    display(predictions_df.head(10))
"""
    ),
    md("## Visual Inference Examples"),
    code(
        r"""
import matplotlib.pyplot as plt


def show_predictions(n: int = 6):
    sample_ids = list(dict.fromkeys(test_df.image_id.tolist()))[:n]
    transform = build_transforms(train=False)
    cols = 2
    rows = math.ceil(n / cols)
    plt.figure(figsize=(14, 5 * rows))
    for i, image_id in enumerate(sample_ids, start=1):
        image = Image.open(IMAGE_DIR / image_id).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        caption = beam_search_caption(encoder, decoder, image_tensor, vocab, MAX_LEN, BEAM_SIZE)
        refs = captions_df[captions_df.image_id == image_id].caption.head(2).tolist()
        plt.subplot(rows, cols, i)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Generated: {caption}\nRef: {refs[0] if refs else ''}", fontsize=10)
    plt.tight_layout()
    out_path = WORK_DIR / "inference_examples.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


if RUN_SAMPLE_INFERENCE and checkpoint_path.exists():
    show_predictions(n=6)
"""
    ),
    md("## Run Metadata"),
    code(
        r"""
run_metadata = {
    "model_kind": MODEL_KIND,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
    "max_len": MAX_LEN,
    "beam_size": BEAM_SIZE,
    "length_alpha": LENGTH_ALPHA,
    "repetition_penalty": REPETITION_PENALTY,
    "eval_limit": EVAL_LIMIT,
}
(WORK_DIR / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
run_metadata
"""
    ),
    md("## Optional: Publish Artifacts Directly to GitHub"),
    code(
        r"""
def publish_artifacts_to_github() -> None:
    if not PUBLISH_TO_GITHUB:
        print("GitHub publishing disabled. Set PUBLISH_TO_GITHUB = True to push artifacts.")
        return

    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise RuntimeError("Kaggle Secrets are only available inside Kaggle notebooks.") from exc

    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Missing Kaggle Secret: GITHUB_TOKEN")

    repo_url = f"https://x-access-token:{token}@github.com/{GITHUB_REPO}.git"
    clone_dir = Path("/kaggle/working/github_repo")
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    subprocess.run(["git", "clone", "--depth", "1", "--branch", GITHUB_BASE_BRANCH, repo_url, str(clone_dir)], check=True)
    subprocess.run(["git", "config", "user.name", GITHUB_COMMIT_NAME], cwd=clone_dir, check=True)
    subprocess.run(["git", "config", "user.email", GITHUB_COMMIT_EMAIL], cwd=clone_dir, check=True)

    if shutil.which("git-lfs") or shutil.which("git-lfs.exe"):
        subprocess.run(["git", "lfs", "install", "--local"], cwd=clone_dir, check=True)
        subprocess.run(["git", "lfs", "track", "artifacts/**/*.pt"], cwd=clone_dir, check=True)
    else:
        print("git-lfs was not found. Large .pt files may fail to push.")

    if GITHUB_TARGET_BRANCH != GITHUB_BASE_BRANCH:
        subprocess.run(["git", "checkout", "-B", GITHUB_TARGET_BRANCH], cwd=clone_dir, check=True)

    artifact_dir = clone_dir / "artifacts" / "resnet_attention"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "best_resnet_attention.pt",
        "bleu_scores.json",
        "sample_predictions.csv",
        "training_history.csv",
        "inference_examples.png",
        "run_metadata.json",
    ]
    copied = []
    for filename in files_to_copy:
        source = WORK_DIR / filename
        if source.exists():
            shutil.copy2(source, artifact_dir / filename)
            copied.append(filename)

    if not copied:
        raise RuntimeError(f"No expected artifact files found in {WORK_DIR}")

    subprocess.run(["git", "add", ".gitattributes", str(artifact_dir.relative_to(clone_dir))], cwd=clone_dir, check=True)
    status = subprocess.run(["git", "status", "--short"], cwd=clone_dir, check=True, capture_output=True, text=True).stdout
    if not status.strip():
        print("No GitHub artifact changes to commit.")
        return

    subprocess.run(["git", "commit", "-m", "data: add resnet attention kaggle artifacts"], cwd=clone_dir, check=True)
    subprocess.run(["git", "push", "origin", f"HEAD:{GITHUB_TARGET_BRANCH}"], cwd=clone_dir, check=True)
    print(f"Pushed {copied} to {GITHUB_REPO}:{GITHUB_TARGET_BRANCH}")


# Run after training/evaluation if GitHub publishing is enabled.
# publish_artifacts_to_github()
"""
    ),
    md(
        """
## Result Notes

After the run finishes, compare this notebook against the saved VGG16 attention results:

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---:|---:|---:|---:|
| VGG16 Attention | 0.5718 | 0.3901 | 0.2625 | 0.1761 |
| ResNet50 Attention | fill after run | fill after run | fill after run | fill after run |

Useful checks:

- Did ResNet50 improve object/action recognition?
- Did beam length normalization reduce short or repetitive captions?
- Did validation loss continue improving after epoch 5?
- Are qualitative samples better even if BLEU changes modestly?
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path("notebooks/flickr8k_resnet_attention_kaggle.ipynb")
output.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(output)
