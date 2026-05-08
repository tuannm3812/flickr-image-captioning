# Assignment Conversion Summary

The source notebook, `Deep Learning - Assignment 3 - 2025.ipynb`, described a Flickr8k image-captioning workflow but contained markdown sections rather than executable implementation cells.

This repository converts that assignment outline into a reusable project:

- Environment setup became `pyproject.toml`, `configs/default.yaml`, and package entry points.
- Data ingestion, cleaning, vocabulary building, and split handling became `data.py` and `text.py`.
- PyTorch dataset and collate logic became reusable loader functions in `data.py`.
- Baseline VGG16 plus LSTM architecture became `models.py`.
- Attention, beam search, and greedy decoding became `models.py` and `inference.py`.
- Training, validation, checkpointing, and BLEU evaluation became `train.py` and `evaluation.py`.

The original notebook is preserved under `notebooks/archive/`. The refined Kaggle workflows are split by architecture:

- `notebooks/flickr8k_baseline_kaggle.ipynb`
- `notebooks/flickr8k_attention_kaggle.ipynb`
- `notebooks/flickr8k_resnet_attention_kaggle.ipynb`

They include inline code for training, saving checkpoints, generating BLEU scores, exporting prediction examples, and optionally publishing `/kaggle/working` outputs to a Kaggle Dataset through Kaggle Secrets.

The `src/` package remains the maintainable codebase version of the same workflow.

## Current Results

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---:|---:|---:|---:|
| Baseline CNN-LSTM | 0.4929 | 0.3009 | 0.1680 | 0.1060 |
| Attention CNN-LSTM | 0.5718 | 0.3901 | 0.2625 | 0.1761 |

The attention model improves all BLEU metrics, with the largest practical gain in phrase-level quality shown by BLEU-3 and BLEU-4.

The next experiment notebook upgrades the visual encoder to ResNet50 and adds mixed precision, scheduling, and improved beam-search controls so it can be compared directly against the VGG16 attention benchmark.

The Kaggle notebooks now use longer maximum training runs with validation-loss early stopping. Baseline and VGG16 attention run up to 20 epochs with patience 3; ResNet50 attention runs up to 25 epochs with patience 4.
