# Attention Kaggle Run Artifacts

Artifacts from the attention CNN-LSTM Kaggle run.

## Metrics

| Metric | Score |
|---|---:|
| BLEU-1 | 0.5718 |
| BLEU-2 | 0.3901 |
| BLEU-3 | 0.2625 |
| BLEU-4 | 0.1761 |

## Training History

| Epoch | Train Loss | Validation Loss | Seconds |
|---:|---:|---:|---:|
| 1 | 3.7627 | 3.1644 | 432.7 |
| 2 | 3.0101 | 2.9313 | 412.4 |
| 3 | 2.7598 | 2.8296 | 418.0 |
| 4 | 2.5949 | 2.7793 | 415.0 |
| 5 | 2.4681 | 2.7660 | 424.0 |

## Files

- `best_attention.pt`: saved attention checkpoint.
- `bleu_scores.json`: BLEU-1 through BLEU-4 metrics.
- `sample_predictions.csv`: generated captions and references.
- `training_history.csv`: train and validation loss per epoch.
- `inference_examples.png`: visual inference grid exported from the notebook.
