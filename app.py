"""Streamlit interface for the Open-Vision user-guided object memory."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from open_vision_memory import DinoV3FeatureExtractor
from open_vision_memory.clustering import cluster_embeddings


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MEMORY_PATH = APP_DIR / ".open_vision_memory" / "object_memory.npz"
if os.environ.get("OPEN_VISION_MEMORY_PATH"):
    DEFAULT_MEMORY_PATH = Path(os.environ["OPEN_VISION_MEMORY_PATH"]).expanduser()
DEFAULT_PROTOTYPE_PATH = os.environ.get("OPEN_VISION_PROTOTYPE_PATH")
DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def load_examples(path: Path) -> tuple[list[np.ndarray], list[str]]:
    if not path.is_file():
        return [], []
    with np.load(path, allow_pickle=False) as data:
        if "embeddings" not in data or "labels" not in data:
            return [], []
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        labels = [str(label) for label in data["labels"].tolist()]
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError(f"Saved object memory is malformed: {path}")
    return [row.copy() for row in embeddings], labels


def save_examples(path: Path, examples: list[np.ndarray], labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = len(examples[0]) if examples else 0
    matrix = np.stack(examples).astype(np.float32) if examples else np.empty((0, width), np.float32)
    np.savez_compressed(path, embeddings=matrix, labels=np.asarray(labels, dtype=str))


def read_prototype_bytes(content: bytes) -> tuple[np.ndarray, list[str], float | None]:
    data = np.load(io.BytesIO(content), allow_pickle=False)
    try:
        labels_key = "labels" if "labels" in data else "known_ids"
        prototypes_key = "prototypes" if "prototypes" in data else "prototype_vectors"
        if labels_key not in data or prototypes_key not in data:
            raise ValueError("The NPZ must contain prototype labels and vectors.")
        labels = [str(value) for value in data[labels_key].tolist()]
        prototypes = np.asarray(data[prototypes_key], dtype=np.float32)
        if prototypes.ndim != 2 or prototypes.shape[0] != len(labels):
            raise ValueError("Prototype labels and vectors have incompatible shapes.")
        threshold = None
        for key in ("threshold", "rejection_threshold"):
            if key in data:
                value = float(np.asarray(data[key]).reshape(-1)[0])
                if np.isfinite(value):
                    threshold = value
                break
        return np.stack([unit_vector(row) for row in prototypes]), labels, threshold
    finally:
        data.close()


def read_prototype_artifact(file) -> tuple[np.ndarray, list[str], float | None]:
    return read_prototype_bytes(file.getvalue())


@st.cache_resource
def get_extractor(model_name: str, device: str, batch_size: int) -> DinoV3FeatureExtractor:
    selected_device = None if device == "auto" else device
    return DinoV3FeatureExtractor(model_name, device=selected_device, batch_size=batch_size)


def combine_memory(
    base_prototypes: np.ndarray | None,
    base_labels: list[str],
    examples: list[np.ndarray],
    example_labels: list[str],
) -> tuple[np.ndarray, list[str]]:
    rows: dict[str, list[np.ndarray]] = {}
    if base_prototypes is not None:
        for label, prototype in zip(base_labels, base_prototypes, strict=True):
            rows.setdefault(label, []).append(unit_vector(prototype))
    for label, embedding in zip(example_labels, examples, strict=True):
        rows.setdefault(label, []).append(unit_vector(embedding))
    labels = sorted(rows)
    if not labels:
        return np.empty((0, 0), dtype=np.float32), []
    prototypes = np.stack([unit_vector(np.mean(rows[label], axis=0)) for label in labels])
    return prototypes, labels


st.set_page_config(page_title="Open-Vision Memory", page_icon="👁️", layout="wide")
st.title("Open-Vision Memory")
st.caption("Recognize familiar objects, reject unfamiliar ones, and teach the system new categories.")

if "memory_examples" not in st.session_state:
    try:
        st.session_state.memory_examples, st.session_state.memory_labels = load_examples(DEFAULT_MEMORY_PATH)
    except Exception as error:
        st.session_state.memory_examples, st.session_state.memory_labels = [], []
        st.error(f"Could not read the saved object memory: {error}")
if "base_prototypes" not in st.session_state:
    st.session_state.base_prototypes = None
    st.session_state.base_labels = []
    st.session_state.base_threshold = None
    if DEFAULT_PROTOTYPE_PATH and Path(DEFAULT_PROTOTYPE_PATH).is_file():
        try:
            artifact_bytes = Path(DEFAULT_PROTOTYPE_PATH).read_bytes()
            (st.session_state.base_prototypes,
             st.session_state.base_labels,
             st.session_state.base_threshold) = read_prototype_bytes(artifact_bytes)
        except Exception as error:
            st.error(f"Could not load the configured prototype memory: {error}")
if "predictions" not in st.session_state:
    st.session_state.predictions = []
    st.session_state.unknown_embeddings = []
    st.session_state.unknown_names = []
    st.session_state.query_embeddings = []
    st.session_state.query_names = []
if "learned_message" in st.session_state:
    st.success(st.session_state.pop("learned_message"))

with st.sidebar:
    st.header("Model and settings")
    model_name = st.text_input("Hugging Face model", value=DEFAULT_MODEL)
    device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    batch_size = st.slider("Batch size", min_value=1, max_value=32, value=8)
    st.caption("DINOv3 access may require accepted Hugging Face model terms and an HF_TOKEN environment variable.")
    if DEFAULT_PROTOTYPE_PATH and Path(DEFAULT_PROTOTYPE_PATH).is_file():
        st.caption("Loaded the prototype memory configured for this deployment.")
    st.divider()
    st.subheader("Known-class memory")
    prototype_file = st.file_uploader("Load a saved prototype_memory.npz", type=["npz"], key="prototype_upload")
    if prototype_file is not None and st.button("Load prototype memory", width="stretch"):
        try:
            protos, names, saved_threshold = read_prototype_artifact(prototype_file)
            st.session_state.base_prototypes = protos
            st.session_state.base_labels = names
            st.session_state.base_threshold = saved_threshold
            st.session_state.predictions = []
            st.session_state.unknown_embeddings = []
            st.session_state.unknown_names = []
            st.session_state.query_embeddings = []
            st.session_state.query_names = []
            st.success(f"Loaded {len(names)} prototype classes.")
        except Exception as error:
            st.error(f"Could not load prototypes: {error}")

base_prototypes = st.session_state.base_prototypes
base_labels = st.session_state.base_labels
examples = st.session_state.memory_examples
example_labels = st.session_state.memory_labels
prototype_matrix, memory_labels = combine_memory(base_prototypes, base_labels, examples, example_labels)
saved_threshold = st.session_state.base_threshold
default_threshold = float(saved_threshold if saved_threshold is not None else 0.65)
threshold = st.sidebar.number_input(
    "Unknown rejection distance",
    min_value=0.0,
    max_value=2.0,
    value=min(max(default_threshold, 0.0), 2.0),
    step=0.01,
    format="%.5f",
)

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Known classes", len(memory_labels))
metric_b.metric("User-taught examples", len(examples))
metric_c.metric("Requested device", device.upper() if device != "auto" else "Auto")

recognize_tab, teach_tab, memory_tab = st.tabs(["Recognize", "Teach an object", "Memory and clustering"])

with recognize_tab:
    st.subheader("Open-set recognition")
    query_files = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="recognition_upload")
    run_recognition = st.button("Recognize images", type="primary", disabled=not query_files or len(memory_labels) == 0)
    if len(memory_labels) == 0:
        st.info("Load a prototype NPZ in the sidebar or teach at least one category first.")
    if run_recognition and query_files:
        st.session_state.query_embeddings = []
        st.session_state.query_names = []
        st.session_state.predictions = []
        st.session_state.unknown_embeddings = []
        st.session_state.unknown_names = []
        try:
            extractor = get_extractor(model_name, device, batch_size)
            images = [Image.open(io.BytesIO(file.getvalue())).convert("RGB") for file in query_files]
            embeddings = extractor.encode_images(images)
            st.session_state.query_embeddings = [row.copy() for row in embeddings]
            st.session_state.query_names = [file.name for file in query_files]
        except Exception as error:
            st.error(f"Recognition could not run: {error}")
    if st.session_state.query_embeddings and len(memory_labels):
        query_embeddings = np.stack(st.session_state.query_embeddings)
        distances = np.clip(1.0 - query_embeddings @ prototype_matrix.T, 0.0, 2.0)
        indices = distances.argmin(axis=1)
        scores = distances[np.arange(len(query_embeddings)), indices]
        is_unknown = scores > threshold
        st.session_state.predictions = [
            {
                "Image": name,
                "Prediction": "Unknown" if unknown else memory_labels[index],
                "Distance": round(float(score), 4),
                "Decision": "Rejected" if unknown else "Accepted",
            }
            for name, index, score, unknown in zip(st.session_state.query_names, indices, scores, is_unknown, strict=True)
        ]
        st.session_state.unknown_embeddings = [row.copy() for row, unknown in zip(query_embeddings, is_unknown, strict=True) if unknown]
        st.session_state.unknown_names = [name for name, unknown in zip(st.session_state.query_names, is_unknown, strict=True) if unknown]
    if st.session_state.predictions:
        st.dataframe(st.session_state.predictions, width="stretch", hide_index=True)
        known_count = sum(item["Decision"] == "Accepted" for item in st.session_state.predictions)
        st.caption(f"Accepted {known_count} of {len(st.session_state.predictions)} images with distance threshold {threshold:.3f}.")

with teach_tab:
    st.subheader("User-guided object learning")
    st.write("Give a name to example images. Their normalized DINOv3 features are added to the persistent object memory.")
    teach_files = st.file_uploader("Upload example images for one category", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="teach_upload")
    category_name = st.text_input("Category name", placeholder="e.g. my red water bottle")
    if st.button("Learn this category", disabled=not teach_files or not category_name.strip(), type="primary"):
        try:
            extractor = get_extractor(model_name, device, batch_size)
            images = [Image.open(io.BytesIO(file.getvalue())).convert("RGB") for file in teach_files]
            new_embeddings = extractor.encode_images(images)
            st.session_state.memory_examples.extend(row.copy() for row in new_embeddings)
            st.session_state.memory_labels.extend([category_name.strip()] * len(new_embeddings))
            save_examples(DEFAULT_MEMORY_PATH, st.session_state.memory_examples, st.session_state.memory_labels)
            st.session_state.predictions = []
            st.session_state.learned_message = f"Added {len(new_embeddings)} example(s) for “{category_name.strip()}”. Memory saved locally."
            st.rerun()
        except Exception as error:
            st.error(f"Learning could not be saved: {error}")

with memory_tab:
    st.subheader("Saved object memory")
    if memory_labels:
        class_rows = []
        for label in memory_labels:
            count = sum(item == label for item in example_labels)
            class_rows.append({"Category": label, "User examples": count, "Loaded prototype": label in base_labels})
        st.dataframe(class_rows, width="stretch", hide_index=True)
        user_memory_bytes = DEFAULT_MEMORY_PATH.read_bytes() if DEFAULT_MEMORY_PATH.exists() else b""
        st.download_button("Download user memory", data=user_memory_bytes, file_name="object_memory.npz", mime="application/octet-stream", disabled=not user_memory_bytes)
    else:
        st.info("No categories have been loaded or learned yet.")
    st.divider()
    st.subheader("Explore rejected images")
    unknown_vectors = st.session_state.unknown_embeddings
    if len(unknown_vectors) >= 2:
        min_size = st.slider("Minimum cluster size", min_value=2, max_value=min(20, len(unknown_vectors)), value=min(3, len(unknown_vectors)), key="cluster_size")
        if st.button("Cluster rejected images with HDBSCAN"):
            try:
                cluster_ids, strengths = cluster_embeddings(np.stack(unknown_vectors), min_cluster_size=min_size)
                st.dataframe([{"Rejected image": name, "Cluster": int(cluster), "Membership strength": round(float(strength), 3)} for name, cluster, strength in zip(st.session_state.unknown_names, cluster_ids, strengths, strict=True)], width="stretch", hide_index=True)
            except Exception as error:
                st.error(f"Clustering could not run. Install the clustering extra: {error}")
    else:
        st.caption("Recognize at least two unfamiliar images to explore their clusters.")

with st.expander("About this baseline"):
    st.write("This prototype system uses cosine distance to the nearest class prototype. A query is rejected when its distance exceeds the selected threshold. The threshold from a loaded artifact is used as the initial setting when available. User-taught images update prototypes, but this app does not claim calibrated probabilities; validate a threshold on held-out images before reporting metrics.")
    st.write("The app stores user-taught embeddings and labels in a local NPZ file. Model weights are cached by Streamlit for the current server process.")
