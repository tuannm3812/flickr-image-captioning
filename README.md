# Flickr8k Image Captioning

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN--LSTM-ee4c2c)
![Dataset](https://img.shields.io/badge/dataset-Flickr8k-2ea44f)
![Kaggle](https://img.shields.io/badge/runtime-Kaggle-20beff)
![Git LFS](https://img.shields.io/badge/checkpoints-Git%20LFS-f64935)
![Best BLEU-4](https://img.shields.io/badge/best%20BLEU--4-0.1955-brightgreen)

A PyTorch image-captioning project converted from an assignment notebook into a reproducible training, evaluation, and inference workflow.

The project compares two CNN-LSTM captioning systems on Flickr8k:

- **Baseline CNN-LSTM:** frozen VGG16 encoder plus LSTM decoder.
- **Attention CNN-LSTM:** VGG16 spatial features plus additive attention and beam-search decoding.
- **ResNet50 Attention:** stronger spatial encoder plus attention, mixed precision, scheduling, and improved beam search.

## Sample Captions

The image below shows qualitative predictions from the strongest current model, ResNet50 attention. These samples are exported directly from the Kaggle inference workflow.

![ResNet50 attention image-captioning examples](artifacts/resnet_attention/inference_examples.png)

## Results

Kaggle runs are saved under `artifacts/`. Attention improves every BLEU metric over the baseline, and the ResNet50 attention model is the strongest run so far.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Validation Loss | Artifacts |
|---|---:|---:|---:|---:|---:|---|
| Baseline CNN-LSTM | 0.4929 | 0.3009 | 0.1680 | 0.1060 | 2.8575 | `artifacts/baseline/` |
| Attention CNN-LSTM | 0.5718 | 0.3901 | 0.2625 | 0.1761 | 2.7660 | `artifacts/attention/` |
| ResNet50 Attention | 0.6103 | 0.4318 | 0.2885 | 0.1955 | 2.7556 | `artifacts/resnet_attention/` |

### Evaluation Insights

The ResNet50 attention model is the best system in this experiment. Compared with the VGG16 attention model, BLEU-1 increased from **0.5718** to **0.6103**, and BLEU-4 increased from **0.1761** to **0.1955**. That is about an **11% relative BLEU-4 gain**, which points to better phrase-level caption quality rather than only better object keywords.

Qualitatively, ResNet50 attention improves several subject/action captions, but it still makes plausible semantic mistakes. For example, it can correctly identify a dog or people sitting together while hallucinating the surrounding context. The remaining errors suggest that the next meaningful step is not only a stronger CNN backbone, but also better decoding constraints, larger-scale pretraining, or a transformer-based captioning model.

## Repository Layout

```text
.
|-- artifacts/
|   |-- baseline/
|   |-- attention/
|   `-- resnet_attention/
|-- configs/
|   `-- default.yaml
|-- docs/
|   `-- assignment_summary.md
|-- notebooks/
|   |-- flickr8k_baseline_kaggle.ipynb
|   |-- flickr8k_attention_kaggle.ipynb
|   |-- flickr8k_resnet_attention_kaggle.ipynb
|   `-- archive/
|-- src/
|   `-- flickr_captioning/
|-- scripts/
|   `-- build_improvement_notebook.py
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
- `notebooks/flickr8k_resnet_attention_kaggle.ipynb`

The notebooks keep the full training, prediction, checkpointing, and evaluation code inline so they can run on Kaggle without installing this repository.

Training defaults are intentionally set higher now, with early stopping enabled:

| Notebook | Max Epochs | Early Stopping |
|---|---:|---|
| Baseline CNN-LSTM | 20 | patience 3, min delta 0.001 |
| VGG16 Attention | 20 | patience 3, min delta 0.001 |
| ResNet50 Attention | 25 | patience 4, min delta 0.001 |

Each notebook writes outputs to `/kaggle/working`:

- `best_baseline.pt` or `best_attention.pt`
- `bleu_scores.json`
- `sample_predictions.csv`
- `training_history.csv`
- `inference_examples.png`

### Next Experiment Notebook

`notebooks/flickr8k_resnet_attention_kaggle.ipynb` is the next-improvement run. It keeps the attention decoder but upgrades the visual encoder to ResNet50 and adds:

- mixed precision training on CUDA,
- `ReduceLROnPlateau` scheduling,
- beam-search length normalization,
- a repetition penalty,
- `run_metadata.json` export.

Use it to compare against the current VGG16 attention benchmark:

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---:|---:|---:|---:|
| VGG16 Attention | 0.5718 | 0.3901 | 0.2625 | 0.1761 |
| ResNet50 Attention | 0.6103 | 0.4318 | 0.2885 | 0.1955 |

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
flickr-caption predict --checkpoint models/best_baseline.pt --image path\to\image.jpg
flickr-caption predict --checkpoint models/best_attention.pt --image path\to\image.jpg --beam-size 3

# Evaluate
flickr-caption evaluate --checkpoint models/best_baseline.pt
flickr-caption evaluate --checkpoint models/best_attention.pt --beam-size 3
```

The saved `artifacts/` checkpoints come from the self-contained Kaggle notebooks and are primarily preserved as run outputs. Use the notebooks for reproducing those exact artifact workflows.

## Next Improvements

- Train attention for more epochs and compare overfitting behavior.
- Replace VGG16 with a stronger encoder such as ResNet or EfficientNet.
- Add attention heatmap visualization for interpretability.
- Add repetition penalties or length normalization tuning for beam search.
