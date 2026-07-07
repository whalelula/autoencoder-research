# SAME Training

This project vendors the open SAME implementation pieces from
`Stability-AI/stable-audio-tools` and adapts them to the existing
`ae-train` flow. The goal is to train an official SAME-S-shaped autoencoder
with the same validation, checkpoint, and sample export logic used elsewhere
in this repository.

## What Is Vendored

Source reference:

- Repository: `https://github.com/Stability-AI/stable-audio-tools`
- Commit used for vendoring: `3241adba4fc2a85cf5b29d9eb68d42f40a28e820`
- Upstream files referenced:
  - `stable_audio_tools/models/autoencoders.py`
  - `stable_audio_tools/models/pretransforms.py`
  - `stable_audio_tools/models/bottleneck.py`
  - `stable_audio_tools/models/transformer.py`

Local files:

- `src/ae_research/vendor/stable_audio_tools/transformer.py`
- `src/ae_research/vendor/stable_audio_tools/same.py`
- `src/ae_research/models/same_autoencoder.py`

The local `same_s` config follows the public `stabilityai/SAME-S`
`model_config.json`: patched pretransform, one encoder TRB, SoftNorm
bottleneck, one decoder TRB, and unpatching back to waveform.

## Training

Install the project after installing the PyTorch and torchaudio build that
matches your CUDA environment:

```powershell
pip install -e .
```

Run SAME training:

```powershell
ae-train --config configs/same_s.yaml --device cuda
```

The SAME-S config uses the official audio shape:

```yaml
data:
  root: data/processed_same_s
  manifest_dir: data/processed_same_s/manifests
  sample_rate: 44100
  duration_seconds: 0.5572789115646258
  channels: 2
```

So `ae-train` will prepare a separate 44.1 kHz stereo chunk directory for
SAME-S instead of reusing the 24 kHz mono MERT chunks.
Outputs are written to:

```text
runs/same_s/
```

including:

- `resolved_config.yaml`
- `history.csv`
- `loss_curves.png`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `samples/`

## Compare Against This Project's Autoencoder

Train the existing MERT autoencoder with its own config, for example:

```powershell
ae-train --config configs/base.yaml --device cuda
```

Then compare reconstructions by inspecting the exported paired samples:

```text
runs/mert95m/samples/
runs/same_s/samples/
```

Both runs use the same training loop and MR-STFT reconstruction objective, so
the logged fields are comparable at the training-loop level. For SAME-S, the
`kl` field is backed by the SoftNorm bottleneck loss.

- `total`
- `mrstft`
- `kl`
- `mrstft_sc`
- `mrstft_lm`
- `mrstft_if`
- `mrstft_gd`
- `mrstft_complex`
- `si_sdr`

For aggregate metrics, use the existing evaluation command on each checkpoint
or exported audio directory according to your current evaluation workflow.

## SAME Config Notes

The default `configs/same_s.yaml` mirrors the public official SAME-S model
configuration:

```yaml
model:
  type: same
  variant: same_s
  pretransform:
    type: patched
    config:
      patch_size: 256
      channels: 2
  encoder:
    type: same
    config:
      in_channels: 512
      channels: 128
      c_mults: [6]
      strides: [16]
      latent_dim: 256
      transformer_depths: [6]
      chunk_size: 32
      chunk_midpoint_shift: true
  bottleneck:
    type: softnorm
  latent_dim: 256
  downsampling_ratio: 4096
```

The total downsampling ratio is `patch_size 256 * TRB stride 16 = 4096`.
For the official `24576`-sample input length, the latent sequence has 6 frames.

If GPU memory is tight, make a local smoke config by reducing:

```yaml
model:
  encoder:
    config:
      channels: 32
      c_mults: [2]
      transformer_depths: [1]
  decoder:
    config:
      channels: 32
      c_mults: [2]
      transformer_depths: [1]
training:
  batch_size: 1
```

Keep `chunk_size` divisible by `stride` when `sliding_window: null`.

## Resume

Resume works the same way as the existing trainer:

```yaml
training:
  resume_from: runs/same_s/checkpoints/last.pt
```

Then rerun:

```powershell
ae-train --config configs/same_s.yaml --device cuda
```
