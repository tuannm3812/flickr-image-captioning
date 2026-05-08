# Baseline Kaggle Run Artifacts

Artifacts from the baseline CNN-LSTM Kaggle run.

## Metrics

| Metric | Score |
|---|---:|
| BLEU-1 | 0.4929 |
| BLEU-2 | 0.3009 |
| BLEU-3 | 0.1680 |
| BLEU-4 | 0.1060 |

## Training History

| Epoch | Train Loss | Validation Loss | Seconds |
|---:|---:|---:|---:|
| 1 | 3.8348 | 3.3120 | 184.8 |
| 2 | 3.1431 | 3.0558 | 196.0 |
| 3 | 2.8872 | 2.9506 | 202.5 |
| 4 | 2.7208 | 2.8894 | 200.7 |
| 5 | 2.5985 | 2.8575 | 202.1 |

## Files

- `best_baseline.pt`: saved baseline checkpoint.
- `bleu_scores.json`: BLEU-1 through BLEU-4 metrics.
- `sample_predictions.csv`: generated captions and references.
- `training_history.csv`: train and validation loss per epoch.
- `inference_examples.png`: visual inference grid exported from the notebook.
