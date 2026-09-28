# System architecture

Open-Vision Memory separates visual representation, recognition, rejection,
evaluation, and user memory so datasets and interfaces can reuse the same core.

```text
RGB image
   │
   ▼
DINOv3 ViT-S/16 image processor and pretrained encoder
   │  class-token embedding (384 dimensions)
   ▼
L2-normalized feature vector
   ├── training examples → normalized class mean prototypes
   ├── validation examples → nearest-prototype distance threshold
   └── query images → cosine nearest prototype → known label or unknown
                                   │
                                   └── rejected query embeddings → optional HDBSCAN
```

## Recognition and rejection

For an image embedding `x` and normalized class prototype `p`, the distance is
`1 - dot(x, p)`. The nearest prototype supplies the candidate class. A query is
accepted when its nearest distance is at or below the selected threshold;
otherwise it receives the reserved label `UNKNOWN_LABEL` (`-1`). This score is a
distance, not a probability.

The prototype for a class is the normalized mean of its normalized training
embeddings. The default rejection threshold is the 95th percentile of nearest
prototype distances on known-class validation examples. The quantile is
configurable. Thresholds depend on the backbone, dataset, class split, and
validation protocol, so artifacts should be used with their matching model and
experiment configuration.

## User-guided memory

The Streamlit app stores normalized image embeddings and category strings in
`.open_vision_memory/object_memory.npz` on the machine running the app. The
current prototype for a learned category is the normalized mean of its examples.
This is a local convenience memory, not an online-trained neural network. The
user should add varied examples and choose a threshold using held-out data when
reliable rejection is important.

## Modules

- `features.py`: batched RGB preprocessing and DINOv3 embedding extraction.
- `memory.py`: normalized class prototypes, validation calibration, prediction,
  and NPZ persistence.
- `evaluation.py`: known/unknown classification and open-set metrics.
- `clustering.py`: optional HDBSCAN exploratory clustering.
- `artifacts.py`: compressed embedding and metadata persistence.
- `app.py`: image upload, recognition, user-guided memory, and reject clustering.
- `scripts/predict.py`: command-line inference from an existing prototype NPZ.

The pretrained DINOv3 checkpoint is hosted by Hugging Face and may require
accepting its terms and authenticating. The checkpoint itself is not distributed
in this repository.
