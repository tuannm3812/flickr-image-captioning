# Flickr8k Image Captioning

A PyTorch image-captioning project converted from an assignment notebook into a reproducible training, evaluation, and inference workflow.

The project compares two CNN-LSTM captioning systems on Flickr8k:

- **Baseline CNN-LSTM:** frozen VGG16 encoder plus LSTM decoder.
- **Attention CNN-LSTM:** VGG16 spatial features plus additive attention and beam-search decoding.

## Results

Kaggle runs are saved under `artifacts/`. The attention model improves every BLEU metric, with the strongest gains on BLEU-3 and BLEU-4, which suggests better phrase-level fluency rather than only better single-word matching.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Validation Loss | Artifacts |
|---|---:|---:|---:|---:|---:|---|
| Baseline CNN-LSTM | 0.4929 | 0.3009 | 0.1680 | 0.1060 | 2.8575 | `artifacts/baseline/` |
| Attention CNN-LSTM | 0.5718 | 0.3901 | 0.2625 | 0.1761 | 2.7660 | `artifacts/attention/` |

### Evaluation Insights

The attention model is the better system in this experiment. BLEU-1 increased from **0.4929** to **0.5718**, showing better object and keyword coverage. BLEU-4 increased from **0.1060** to **0.1761**, which is the more important improvement because it captures longer phrase overlap and more coherent caption structure.

Qualitatively, the attention model generates more specific captions, especially for people and action scenes. It still makes semantic mistakes, such as hallucinating scene context or misidentifying subjects, but its captions are generally less generic than the baseline. The remaining errors suggest that more training, a stronger visual backbone, and better decoding constraints would be useful next steps.

## Repository Layout

```text
.
|-- artifacts/
|   |-- baseline/
|   `-- attention/
|-- configs/
|   `-- default.yaml
|-- docs/
|   `-- assignment_summary.md
|-- notebooks/
|   |-- flickr8k_baseline_kaggle.ipynb
|   |-- flickr8k_attention_kaggle.ipynb
|   `-- archive/
|-- src/
|   `-- flickr_captioning/
`-- tests/
```

Checkpoints are tracked with Git LFS because they are close to GitHub's regular file-size limit.

## Dataset

For local runs, place Flickr8k under `data/raw/flickr8k`:

```text
data/raw/flickr8k/
|-- Flickr8k_Dataset/
|   `-- *.jpg
`-- Flickr8k_text/
    |-- Flickr8k.token.txt
    |-- Flickr_8k.trainImages.txt
    |-- Flickr_8k.devImages.txt
    `-- Flickr_8k.testImages.txt
```

Paths can be changed in `configs/default.yaml`.

## Kaggle Workflow

Use these notebooks for training and artifact generation on Kaggle:

- `notebooks/flickr8k_baseline_kaggle.ipynb`
- `notebooks/flickr8k_attention_kaggle.ipynb`

The notebooks keep the full training, prediction, checkpointing, and evaluation code inline so they can run on Kaggle without installing this repository.

Each notebook writes outputs to `/kaggle/working`:

- `best_baseline.pt` or `best_attention.pt`
- `bleu_scores.json`
- `sample_predictions.csv`
- `training_history.csv`
- `inference_examples.png`

### Push Kaggle Outputs to GitHub

To push notebook outputs directly back to this repository:

1. Create a GitHub fine-grained personal access token for this repo with **Contents: Read and write** access.
2. In Kaggle, open **Add-ons > Secrets**.
3. Add a secret named `GITHUB_TOKEN`.
4. In the notebook configuration cell, set:

```python
PUBLISH_TO_GITHUB = True
GITHUB_TARGET_BRANCH = "kaggle-baseline-artifacts"  # or "main"
```

5. After training/evaluation, run:

```python
publish_artifacts_to_github()
```

Using an artifact branch is recommended first. Set `GITHUB_TARGET_BRANCH = "main"` only when you intentionally want Kaggle to commit directly to the main branch.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Install the right PyTorch build for your hardware if your environment does not already include `torch` and `torchvision`.

## Local CLI

Train, predict, and evaluate are available through one CLI:

```powershell
# Train
flickr-caption train --config configs/default.yaml --model baseline
flickr-caption train --config configs/default.yaml --model attention

# Predict
flickr-caption predict --checkpoint artifacts/baseline/best_baseline.pt --image path\to\image.jpg
flickr-caption predict --checkpoint artifacts/attention/best_attention.pt --image path\to\image.jpg --beam-size 3

# Evaluate
flickr-caption evaluate --checkpoint artifacts/baseline/best_baseline.pt
flickr-caption evaluate --checkpoint artifacts/attention/best_attention.pt --beam-size 3
```

## Next Improvements

- Train attention for more epochs and compare overfitting behavior.
- Replace VGG16 with a stronger encoder such as ResNet or EfficientNet.
- Add attention heatmap visualization for interpretability.
- Add repetition penalties or length normalization tuning for beam search.
