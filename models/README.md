# Model files

## Visual backbone

- Hugging Face model ID: `facebook/dinov3-vits16-pretrain-lvd1689m`
- Feature: 384-dimensional CLS token, L2 normalized
- The app downloads model files from Hugging Face on first use and caches them outside this repository. The checkpoint is gated; accept its terms and authenticate with Hugging Face before inference. The backbone weights are deliberately not copied into GitHub.

## Open-set prototype memory

The ImageNet-100 prototype artifact was produced by the Colab run and is stored in Google Drive at:

`/content/drive/MyDrive/open-vision-memory/phase04/imagenet100/prototype_memory.npz`

Download it from the final export cell in the Colab notebook, then load it in the Streamlit sidebar under **Load a saved prototype_memory.npz**. This small NPZ stores known class IDs, normalized prototypes, and the 95th-percentile rejection threshold. Its run settings are summarized in `imagenet100_config.json` and the measured results are in `../results/imagenet100.json`.

The large training and validation embedding matrices remain in Google Drive and are not required for app inference once the prototype memory is available.
