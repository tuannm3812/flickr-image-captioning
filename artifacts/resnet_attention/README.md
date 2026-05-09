# ResNet50 Attention Kaggle Run Artifacts

Artifacts from the ResNet50 attention Kaggle run.

## Metrics

| Metric | Score |
|---|---:|
| BLEU-1 | 0.6103 |
| BLEU-2 | 0.4318 |
| BLEU-3 | 0.2885 |
| BLEU-4 | 0.1955 |

## Training History

The run was configured for 25 maximum epochs with early stopping. Training stopped after epoch 15 because validation loss did not improve by at least 0.001 for 4 consecutive epochs.

| Best Epoch | Train Loss | Validation Loss | Learning Rate |
|---:|---:|---:|---:|
| 11 | 2.1545 | 2.7556 | 0.0004 |

## Review

This is the strongest run so far by BLEU score. Compared with the VGG16 attention model, ResNet50 attention improves:

- BLEU-1 by 0.0385, about 6.7%.
- BLEU-2 by 0.0417, about 10.7%.
- BLEU-3 by 0.0260, about 9.9%.
- BLEU-4 by 0.0193, about 11.0%.

The qualitative samples show better subject/action coverage in several cases, especially people sitting together and the black dog playing in grass. Some hallucinations remain: the model can still infer plausible but incorrect scene context, such as snow for a resting white dog or a simplified caption for the boy at the beach. Overall, the ResNet50 encoder improves aggregate caption quality, but the project would still benefit from stronger decoding constraints, more training data, or a modern encoder-decoder architecture.

## Files

- `best_resnet_attention.pt`: saved ResNet50 attention checkpoint.
- `bleu_scores.json`: BLEU-1 through BLEU-4 metrics.
- `sample_predictions.csv`: generated captions and references.
- `training_history.csv`: train and validation loss per epoch.
- `run_metadata.json`: notebook configuration for the run.
- `inference_examples.png`: visual inference grid exported from the notebook.
