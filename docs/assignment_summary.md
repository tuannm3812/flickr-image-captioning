# Assignment Conversion Summary

The source notebook, `Deep Learning - Assignment 3 - 2025.ipynb`, described a Flickr8k image-captioning workflow but contained markdown sections rather than executable implementation cells.

This repository converts that assignment outline into a reusable project:

- Environment setup became `pyproject.toml`, `configs/default.yaml`, and package entry points.
- Data ingestion, cleaning, vocabulary building, and split handling became `data.py` and `text.py`.
- PyTorch dataset and collate logic became reusable loader functions in `data.py`.
- Baseline VGG16 plus LSTM architecture became `models.py`.
- Attention, beam search, and greedy decoding became `models.py` and `inference.py`.
- Training, validation, checkpointing, and BLEU evaluation became `train.py` and `evaluation.py`.

The original notebook is preserved under `notebooks/archive/`. The refined project notebook is `notebooks/flickr8k_image_captioning_project.ipynb`; it acts as a professional Kaggle experiment notebook with inline code for training, saving checkpoints, generating BLEU scores, and exporting prediction examples.

The `src/` package remains the maintainable codebase version of the same workflow.
