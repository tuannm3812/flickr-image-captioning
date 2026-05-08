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
├── artifacts/
│   └── baseline/
├── notebooks/
│   ├── flickr8k_baseline_kaggle.ipynb
│   ├── flickr8k_attention_kaggle.ipynb
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

Use the Kaggle-ready notebooks for training and artifact generation:

- `notebooks/flickr8k_baseline_kaggle.ipynb`
- `notebooks/flickr8k_attention_kaggle.ipynb`

Both notebooks keep the core training, evaluation, checkpointing, and inference code inline so they can run on Kaggle without installing this repository as a package. They also include an optional publishing cell that can version `/kaggle/working` outputs to a Kaggle Dataset using Kaggle Secrets.

The reusable implementation under `src/` remains the cleaner project version for local development, testing, and future refactoring.

### Push Kaggle Outputs to GitHub

To save notebook outputs directly from Kaggle back to this repository:

1. Create a GitHub fine-grained personal access token for this repo with **Contents: Read and write** access.
2. In Kaggle, open the notebook and go to **Add-ons > Secrets**.
3. Add a secret named `GITHUB_TOKEN` with the token value.
4. In the notebook configuration cell, set:

   ```python
   PUBLISH_TO_GITHUB = True
   GITHUB_TARGET_BRANCH = "kaggle-baseline-artifacts"  # or "main"
   ```

5. Run training/evaluation.
6. Run the final GitHub publishing cell:

   ```python
   publish_artifacts_to_github()
   ```

By default, the notebooks push to a separate artifact branch to avoid unexpected changes on `main`. Set `GITHUB_TARGET_BRANCH = "main"` only when you intentionally want Kaggle to commit directly to the main branch.

## Saved Artifacts

The baseline Kaggle run is saved under `artifacts/baseline/`. The checkpoint is tracked with Git LFS because it is close to GitHub's regular file-size limit.

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
