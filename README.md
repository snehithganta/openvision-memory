# Open-Vision Memory Check

Open-Vision Memory is a reusable open-set visual recognition baseline with a user-guided object memory. Phase 01 contains the model core; CIFAR-10 and STL-10 experiments are implemented in the Colab notebook; ImageNet-100 is the current scalability experiment.

## Core flow

`image -> DINOv3 embedding -> L2 normalization -> cosine prototype classifier -> reject as unknown`

Embeddings can also be clustered with HDBSCAN for exploration. The package keeps backbone extraction, recognition, clustering, persistence, and evaluation independent so later datasets can reuse the same architecture.

## Setup

```bash
pip install -r requirements.txt
```

The full experiment notebook is [Open-Vision Memory - Phases 01-05 on Colab](https://colab.research.google.com/drive/1tySqIEAp4l-ACuXn4pkj9rt04MyiOe-n). It contains the dataset runs and writes their embedding/prototype artifacts to Google Drive. Compact metric outputs are versioned in [`results/`](results/); large `.npz` embedding archives remain in Drive.

The default Hugging Face model ID is `facebook/dinov3-vits16-pretrain-lvd1689m`. DINOv3 weights may require accepting the model terms and authenticating with Hugging Face. In Colab, authenticate with `huggingface-cli login` (or set `HF_TOKEN`) before loading weights. For fully offline use, pass a local model directory as `model_name`.

## Run the interactive app

```bash
pip install -e ".[app,clustering]"
streamlit run app.py
```

The default DINOv3 checkpoint is gated. After accepting its Hugging Face terms, set `HF_TOKEN` in the environment used to launch Streamlit. Do not commit the token. In the sidebar, load a saved prototype NPZ from a completed experiment, or teach categories by uploading labeled examples. User-taught embeddings are saved to `.open_vision_memory/object_memory.npz` on the machine running the app.

The app uses the loaded experiment threshold as its starting point when the artifact contains one. For a newly taught memory, choose the rejection distance using a separate held-out validation set before treating it as a calibrated operating point. User-taught examples directly update their class prototype.

## Minimal inference setup

```python
from open_vision_memory import DinoV3FeatureExtractor, OpenSetRecognizer

extractor = DinoV3FeatureExtractor(device="cuda")
train_embeddings = extractor.encode_images(train_images)
recognizer = OpenSetRecognizer().fit(train_embeddings, train_labels)
recognizer.calibrate(validation_embeddings, validation_labels, known_labels=train_labels)
predictions, unknown_scores = recognizer.predict(test_embeddings)
```

`train_images` can be a sequence of PIL images or NumPy arrays. Supply RGB images. The processor handles resizing and normalization expected by the selected model. Labels are integer-like or strings. For evaluation, encode `UNKNOWN_LABEL` (`-1`) for out-of-scope examples.

Threshold calibration uses the configured quantile of each known validation sample's nearest-prototype cosine distance. It is an initial baseline, not a claim of calibrated probabilities; choose the quantile using validation data only and report it with each experiment.

## Saved artifacts

`save_embeddings` writes embeddings, labels, and JSON metadata (including model/configuration details) in a compressed NPZ. `OpenSetRecognizer.save` writes prototypes and the rejection configuration. These artifacts are sufficient to reproduce recognition without recomputing the training embeddings; feature extraction still requires the selected backbone weights.

The backbone reference and saved ImageNet-100 prototype artifact instructions are in [`models/`](models/). Use `models/imagenet100_config.json` to identify the exact feature model, known/unknown class split, and threshold used by the app-ready prototype artifact.

## Experimental snapshot

The completed runs use DINOv3 ViT-S/16. CIFAR-10 and STL-10 use 80% of classes as known and 20% as unknown; thresholds were calibrated at the 95th percentile from known-class validation examples. The STL-10 matched-budget row uses the same per-class prototype and calibration counts as CIFAR-10. ImageNet-100 uses a seeded split of 80 known and 20 unknown classes, with 50 calibration images per known class.

| Dataset / protocol | Known closed-set accuracy | Known acceptance | Accuracy when accepted | Unknown F1 | Unknown AUROC | HDBSCAN ARI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10, official test split | 0.8941 | 0.9405 | 0.9059 | 0.5940 | 0.9040 | 0.0950 |
| STL-10, 300 fit + 100 calibration per known class | 0.9642 | 0.9350 | 0.9776 | 0.8728 | 0.9871 | 0.2504 |
| STL-10, matched budget: 30 fit + 20 calibration per known class | 0.9427 | 0.9536 | 0.9594 | 0.8337 | 0.9766 | 0.2504 |
| ImageNet-100, 80 known + 20 unknown classes | 0.8755 | 0.9475 | 0.8984 | 0.6203 | 0.8982 | 0.0059 |

These are dataset-specific evaluation results, not a direct cross-dataset ranking: CIFAR-10 and STL-10 use their official test splits, while ImageNet-100 uses its validation split. The ImageNet-100 HDBSCAN probe produced four clusters (including noise) and a low ARI, so clustering did not recover the class structure in this setup. Its full 126,689-image training and 5,000-image validation embedding pass took 1,121.4 seconds on a Tesla T4 (117.4 images/second combined); PyTorch reported 226.1 MB peak allocated GPU memory. The app warns that its selectable rejection distance is a working threshold, not a calibrated probability.

### ImageNet-100 threshold sensitivity

The validation embeddings were reused to compare rejection thresholds calibrated at three quantiles; no feature extraction was repeated. Closed-set known accuracy and AUROC are unchanged because the former ignores rejection and the latter uses the continuous distance score. The quantile changes the known-class coverage versus unknown-detection F1 trade-off.

| Calibration quantile | Threshold | Known acceptance | Accuracy when accepted | Unknown F1 | Unknown AUROC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.90 | 0.6641 | 0.8998 | 0.9114 | 0.6752 | 0.8982 |
| 0.95 | 0.7190 | 0.9475 | 0.8984 | 0.6203 | 0.8982 |
| 0.99 | 0.8045 | 0.9858 | 0.8838 | 0.3020 | 0.8982 |

For this split, the 90th-percentile operating point gives the highest unknown F1 among the three tested, while the 95th-percentile point accepts more known examples. Select the operating point on validation data to match the intended cost of false rejections and missed unknowns. The ablation was saved as `threshold_sensitivity.json` alongside the ImageNet-100 artifacts.

Machine-readable results are checked into [`results/`](results/). ImageNet-100 full artifacts were saved in Google Drive at `/content/drive/MyDrive/open-vision-memory/phase04/imagenet100`: train and validation embeddings, prototype memory, HDBSCAN assignments, `config.json`, `metrics.json`, and `threshold_sensitivity.json`.

## Current scope

- DINOv3 ViT-S/16 feature extraction through Transformers.
- L2-normalized embeddings and cosine nearest-prototype recognition.
- Validation-based unknown rejection.
- Optional HDBSCAN clustering.
- Reusable open-set metrics and embedding/model persistence.

CIFAR-10, STL-10, and ImageNet-100 split design, thresholds, artifacts, and evaluation metrics are recorded above, in `results/`, and in the Google Colab Drive folder. The local Streamlit app in `app.py` supports prototype loading, image recognition and rejection, user-guided persistent category examples, and optional clustering of rejected examples. Install `requirements.txt`, authenticate to Hugging Face after accepting the DINOv3 terms, and launch with `streamlit run app.py`.

The proposed semantic fusion branch (MobileCLIP2 or SigLIP 2), object localization (YOLO26n), and fusion ablations remain future extensions. They are not part of the completed baseline measurements above.
