# Flickr8k Image Captioning

A professional PyTorch project for image captioning on Flickr8k. The original assignment notebook has been converted into reusable modules for data preparation, model training, inference, and evaluation.

## Project Highlights

- Baseline encoder-decoder model using a frozen VGG16 image encoder and LSTM decoder.
- Optional spatial attention decoder with beam-search inference.
- Reproducible configuration through YAML.
- Clean package layout under `src/flickr_captioning`.
- CLI commands for training, evaluation, and prediction.

## Repository Layout

```text
.
├── configs/default.yaml
├── docs/assignment_summary.md
├── notebooks/
│   ├── flickr8k_image_captioning_project.ipynb
│   └── archive/
├── src/flickr_captioning/
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── models.py
│   ├── text.py
│   ├── train.py
│   └── utils.py
└── tests/
```

## Dataset

Download Flickr8k and place it under `data/raw/flickr8k`:

```text
data/raw/flickr8k/
├── Flickr8k_Dataset/
│   └── *.jpg
└── Flickr8k_text/
    ├── Flickr8k.token.txt
    ├── Flickr_8k.trainImages.txt
    ├── Flickr_8k.devImages.txt
    └── Flickr_8k.testImages.txt
```

You can override these paths in `configs/default.yaml`.

## Notebook Workflow

The original assignment notebook is preserved in `notebooks/archive/`.

Use `notebooks/flickr8k_image_captioning_project.ipynb` for the Kaggle-ready professional workflow. It keeps the core training, evaluation, checkpointing, and inference code inline so the notebook can run on Kaggle without installing this repository as a package.

The reusable implementation under `src/` remains the cleaner project version for local development, testing, and future refactoring.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Install the right PyTorch build for your hardware from the official PyTorch selector if needed.

## Train

```powershell
flickr-caption train --config configs/default.yaml --model baseline
```

For the attention model:

```powershell
flickr-caption train --config configs/default.yaml --model attention
```

## Predict

```powershell
flickr-caption predict --checkpoint models/best_baseline.pt --image path\to\image.jpg
```

Use beam search with attention checkpoints:

```powershell
flickr-caption predict --checkpoint models/best_attention.pt --image path\to\image.jpg --model attention --beam-size 3
```

## Evaluate

```powershell
flickr-caption evaluate --checkpoint models/best_baseline.pt --model baseline
```

The evaluator reports BLEU-1 through BLEU-4 on the configured test split.
