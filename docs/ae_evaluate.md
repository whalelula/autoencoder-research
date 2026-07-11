# ae-evaluate unified evaluation CLI

`ae-evaluate` is the single evaluation entry point for project checkpoints and
external baseline autoencoders. The old standalone commands
`ae-evaluate-sa3-same` and `ae-evaluate-stable-audio-vae` have been folded into
this command.

## Model selection

Choose the evaluator with `--model`:


| Model value                                                            | Evaluator                                                 | Required inputs            |
| ---------------------------------------------------------------------- | --------------------------------------------------------- | -------------------------- |
| `checkpoint`, `dam_mert330`, `dam_mert95`, `mert330`, `mert95`, `ours` | Project checkpoint evaluator                              | `--config`, `--checkpoint` |
| `same`                                                                 | SA3 SAME-S and SAME-L baseline evaluator                  | `--data-root`              |
| `same-s`                                                               | SA3 SAME-S only                                           | `--data-root`              |
| `same-l`                                                               | SA3 SAME-L only                                           | `--data-root`              |
| `sao`                                                                  | Stable Audio Open 1.0 VAE/pretransform baseline evaluator | `--data-root`              |


`--model` may be repeated only for SAME variants, for example
`--model same-s --model same-l`. For checkpoint models, the model value is a
label/alias; the actual architecture and evaluation settings still come from
the YAML config and checkpoint.

For backwards compatibility, this still works and implies `--model checkpoint`:

```powershell
ae-evaluate --config configs/base.yaml --checkpoint runs/mert95m/checkpoints/best.pt
```



## Project checkpoint examples

Evaluate a DAM/MERT checkpoint:

```powershell
ae-evaluate `
  --model dam_mert330 `
  --config configs/dam_mert330m_1k_5s.yaml `
  --checkpoint runs/dam_mert330m_1k_5s/checkpoints/best.pt `
  --device cuda
```

Useful overrides:

```powershell
ae-evaluate `
  --model dam_mert330 `
  --config configs/dam_mert330m_1k_5s.yaml `
  --checkpoint runs/dam_mert330m_1k_5s/checkpoints/best.pt `
  --output-dir outputs/evaluation/dam_mert330m_1k_5s_v2 `
  --batch-size 4 `
  --max-batches 2
```

Checkpoint evaluation reads dataset paths, mel/STFT settings, and default output
settings from `config["data"]`, `config["loss"]`, and `config["evaluation"]`.
The CLI can override `--output-dir`, `--batch-size`, `--max-batches`,
`--sample-rate`, `--duration-seconds`, `--channels`, `--num-workers`,
`--no-pin-memory`, `--no-export-audio`, `--max-audio-samples`, and
`--sample-seed`.

## SA3 SAME examples

Evaluate both SAME-S and SAME-L:

```powershell
ae-evaluate `
  --model same `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --device cuda
```

Evaluate only SAME-L:

```powershell
ae-evaluate `
  --model same-l `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --output-dir outputs/evaluation/sa3_same_l_listen5 `
  --sample-count 5 `
  --device cuda
```

Use chunked SA3 encode/decode when memory is tight:

```powershell
ae-evaluate `
  --model same-l `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --chunked `
  --chunk-size 128 `
  --overlap 32 `
  --device cuda
```

SAME evaluation defaults:

- `--output-dir outputs/evaluation/sa3_same`
- `--batch-size 4`
- `--sample-rate 24000`
- `--duration-seconds 5`
- `--channels 1`



## Stable Audio Open VAE examples

Evaluate the Stable Audio Open 1.0 VAE/pretransform:

```powershell
ae-evaluate `
  --model sao `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --device cuda
```

Use an existing sampled manifest so the listening samples match a SAME run:

```powershell
ae-evaluate `
  --model sao `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --manifest-dir outputs/evaluation/sa3_same_l_listen5/sample_manifest `
  --output-dir outputs/evaluation/stable_audio_open_1_0_vae_latent_listen5 `
  --max-audio-samples 5 `
  --device cuda
```

Stable Audio Open VAE defaults:

- `--pretrained-name stabilityai/stable-audio-open-1.0`
- `--system-name stable-audio-open-1.0-vae-latent`
- `--output-dir outputs/evaluation/stable_audio_open_1_0_vae_latent`
- `--batch-size 1`
- `--sample-rate 24000`
- `--duration-seconds 5`
- `--channels 1`

Use `--half` to run the Stable Audio model in half precision.

## Metrics and audio export

All evaluators write `metrics.json` under the selected output directory. When
audio export is enabled, reference and reconstruction WAV files are written next
to the metrics:

- Checkpoint and SAO: `reference/` and `reconstruction/`
- SAME: `reference/`, `same-s/`, and/or `same-l/`

`--max-audio-samples` controls only the final listening WAV export. Metrics and
rFAD still run on the full evaluated set unless `--max-batches` or a sampled
manifest is used. Exported listening samples are selected randomly from
`test.jsonl` with `--sample-seed`, so using the same manifest,
`--max-audio-samples`, and `--sample-seed` across `same`, `sao`, and checkpoint
runs exports the same track IDs for human listening.

The shared metric set includes full-band `SI-SDR`, `MEL`, `MR-STFT`, MR-STFT
components, and bandwise spectral errors:


| Band | Frequency range | Metric keys             |
| ---- | --------------- | ----------------------- |
| low  | 0-500 Hz        | `STFT/low`, `MEL/low`   |
| mid  | 500 Hz-4 kHz    | `STFT/mid`, `MEL/mid`   |
| high | 4-12 kHz        | `STFT/high`, `MEL/high` |
| air  | 12-20 kHz       | `STFT/air`, `MEL/air`   |


Bandwise `STFT/*` and `MEL/*` are log-magnitude L1 errors inside each frequency
band. If the dataset sample rate cannot represent a band, the corresponding
value is `null`; for example, 24 kHz evaluation has a 12 kHz Nyquist frequency,
so the 12-20 kHz air band is unavailable.

To compute rFAD during evaluation, install the eval extra and pass `--run-rfad`:

```powershell
pip install -e ".[eval]"

ae-evaluate `
  --model same `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --run-rfad `
  --fad-model vggish `
  --device cuda
```

`--run-rfad` requires audio export. It can be combined with
`--max-audio-samples`; in that case rFAD uses temporary full-set audio exports
while the final listening WAV directories keep only the requested sampled tracks.