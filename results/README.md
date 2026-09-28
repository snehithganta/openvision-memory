# Experiment results

These compact JSON files capture the evaluated metrics and protocols from the Colab runs. The full experiment code is in the linked Colab notebook in the project README. Embedding matrices and prototype artifacts are too large for this source repository and remain in Google Drive at the paths recorded in the notebook and metrics.

- `cifar10.json`: CIFAR-10 official test split; 30 prototype-fit and 20 calibration images per known class.
- `stl10.json`: STL-10 official labeled test split; 300 prototype-fit and 100 calibration images per known class.
- `stl10_matched_budget.json`: STL-10 test results matched to the CIFAR-10 fit/calibration counts.
- `imagenet100.json`: ImageNet-100 validation split; seeded 80-known/20-unknown class split.
- `imagenet100_threshold_sensitivity.json`: ImageNet-100 rejection-threshold ablation on cached validation embeddings.

All reported unknown-detection metrics use a binary known-versus-unknown target. The HDBSCAN ARI values come from balanced class subsets, as described in each run's config/output.
