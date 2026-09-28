# From Colab results to local inference

The dataset experiments and embedding extraction were run on a Colab Tesla T4.
The Streamlit app can run on a local computer, but DINOv3 feature extraction on a
CPU-only laptop will be slower. The app does not repeat the CIFAR-10, STL-10, or
ImageNet-100 benchmark runs.

## 1. Save the ImageNet-100 prototype artifact from Drive

Open the linked [Phases 01–05 Colab notebook](https://colab.research.google.com/drive/1tySqIEAp4l-ACuXn4pkj9rt04MyiOe-n),
mount the Google Drive used for the experiment, and run a cell containing:

```python
from google.colab import files

artifact = "/content/drive/MyDrive/open-vision-memory/phase04/imagenet100/prototype_memory.npz"
files.download(artifact)
```

The browser downloads `prototype_memory.npz`. Put it in this repository's
`models/` folder. The artifact contains 80 known-class prototype vectors,
their dataset label IDs, and the saved rejection threshold. Large embedding
archives are not needed for inference.

## 2. Install and authenticate

Use Python 3.10 or newer. In a virtual environment, install the project:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Accept the DINOv3 model terms on its Hugging Face page, then authenticate in the
same environment that will run the app:

```bash
hf auth login
```

Alternatively, set `HF_TOKEN` in the shell environment. Never put a real token
in a notebook committed to GitHub, source code, or a public `.env` file. If model
loading fails, confirm that the account accepted the terms and that the token is
available to the process running Streamlit.

## 3. Launch the app

```bash
streamlit run app.py
```

In the sidebar, upload `models/prototype_memory.npz`, then upload images under
**Recognize**. Start with device `auto` and a small batch size. The Hugging Face
checkpoint is downloaded and cached at first use. On CPU-only hardware, even a
small batch can take noticeably longer than the T4 benchmark; use Colab or a
GPU-backed machine for repeated or larger batches.

## CLI inference

```bash
python scripts/predict.py --prototype models/prototype_memory.npz --device auto image.jpg
```

The CLI prints JSON with the predicted label ID/name, nearest-prototype cosine
distance, and rejection threshold. A label map can be supplied as a JSON object
mapping string IDs to display names with `--label-map labels.json`.

## Artifact compatibility

Keep the prototype artifact paired with the feature model and preprocessing used
to create it. A prototype from a different backbone, image processor, or class
split is not interchangeable. The artifact does not contain the pretrained
backbone weights or the full training embeddings.
