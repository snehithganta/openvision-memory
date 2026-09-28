# Metric definitions and interpretation

The committed result files are compact snapshots from separate dataset runs.
They should be compared only with their protocols in each JSON file; dataset,
split, image resolution, and per-class training/calibration budgets differ.

| Metric | Meaning |
| --- | --- |
| `known_closed_set_accuracy` | Top-1 prototype-class accuracy on known test examples, before applying the unknown rejection decision. |
| `known_acceptance_coverage` | Fraction of known examples whose nearest-prototype distance is at or below the rejection threshold. |
| `known_accuracy_among_accepted` | Fraction of accepted known examples assigned the correct known class. |
| `unknown_detection_f1` | F1 for detecting unknown examples as the positive class at the saved threshold. |
| `unknown_detection_auroc` | AUROC from the continuous nearest-prototype distance; larger distance means more unfamiliar. |
| `hdbscan_ari` | Adjusted Rand Index of the exploratory HDBSCAN assignments on the recorded sample, including noise as a cluster label. |

The confusion-matrix values are recorded in the order
`[true_known, false_unknown, missed_unknown, true_unknown]`, equivalently
`[TN, FP, FN, TP]` when unknown is the positive class. F1 and coverage depend on
the selected threshold. AUROC does not use a single threshold.

## Threshold selection

Calibration uses only known-class validation examples and sets the threshold to
their nearest-prototype distance quantile. The ImageNet-100 ablation illustrates
the trade-off: increasing the quantile accepts more known examples, but can miss
more unknown classes. Pick the operating point on validation data according to
the application's cost of false rejection versus missed unknowns. Do not tune on
the final test set, and do not interpret distance scores as calibrated
confidence probabilities.

## Limitations visible in the results

- These experiments use dataset-specific class splits and validation protocols;
  the metric table is not a controlled ranking of datasets.
- CIFAR-10 has lower unknown F1 at the selected 95th-percentile threshold than
  STL-10 in these runs. ImageNet-100 also shows a coverage/F1 trade-off in the
  threshold sensitivity file.
- HDBSCAN is included as an exploratory view. Low ARI, especially on ImageNet-100,
  means the measured clustering did not recover class labels reliably.
- The ImageNet-100 run uses a fixed seeded 80/20 class split and reports a single
  run; the repository does not claim confidence intervals or broad statistical
  significance.
- Runtime and GPU-memory measurements describe the recorded Colab Tesla T4 run.
  They do not predict CPU-only laptop latency.
