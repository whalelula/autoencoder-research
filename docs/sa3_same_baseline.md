# SA3 SAME-S/SAME-L baseline evaluation

本文档说明如何在已经预处理好的 `MTG-Jamendo-1000-24k-mono-5s` test 集上，评测 Stability AI `stable-audio-3` 的 `SAME-S` 和 `SAME-L` autoencoder baseline，并导出主观听感 sample。

## 1. 准备环境

先安装本项目、PyTorch/torchaudio 和 `stable-audio-3`：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install git+https://github.com/Stability-AI/stable-audio-3.git
```

如果 `stable-audio-3` 的依赖和训练环境冲突，可以新建一个只用于 baseline 评测的环境：

```powershell
python -m venv .venv-sa3
.\.venv-sa3\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e .
pip install git+https://github.com/Stability-AI/stable-audio-3.git
```

首次运行会下载 SA3 SAME 权重，请确保 Hugging Face/GitHub 访问正常，并准备足够磁盘空间。

## 2. 数据目录要求

评测命令不依赖 `configs/base.yaml`。只需要传入已经预处理好的 5 秒音频数据目录：

```text
data/MTG-Jamendo-1000-24k-mono-5s/
  manifests/
    test.jsonl
  ...
```

默认假设数据是：

- `24000 Hz`
- `mono`
- `5.0 s`
- manifest 路径为 `DATA_ROOT/manifests/test.jsonl`

如果 manifest 不在默认位置，可以额外传入 `--manifest-dir`。

## 3. 计算 SAME-S 和 SAME-L 指标

默认同时评测 `same-s` 和 `same-l`：

```powershell
ae-evaluate --model same --data-root data/MTG-Jamendo-1000-24k-mono-5s --device cuda
```

只评测其中一个模型：

```powershell
ae-evaluate --model same-s --data-root data/MTG-Jamendo-1000-24k-mono-5s --device cuda
ae-evaluate --model same-l --data-root data/MTG-Jamendo-1000-24k-mono-5s --device cuda
```

显存不足时可启用 SA3 的 chunked encode/decode：

```powershell
ae-evaluate --model same --data-root data/MTG-Jamendo-1000-24k-mono-5s --device cuda --chunked --chunk-size 128 --overlap 32
```

调试时只跑少量 batch：

```powershell
ae-evaluate --model same --data-root data/MTG-Jamendo-1000-24k-mono-5s --device cuda --max-batches 2
```

常用可选参数：

- `--output-dir outputs/evaluation/sa3_same`
- `--batch-size 4`
- `--num-workers 4`
- `--max-audio-samples 32`
- `--no-export-audio`

## 4. 输出内容

默认输出到：

```text
outputs/evaluation/sa3_same/
  metrics.json
  mushra_command.txt
  reference/
  same-s/
  same-l/
```

`metrics.json` 包含每个 SAME baseline 的：

- `SI-SDR`
- `MEL`
- `MR-STFT`
- `MR-STFT/sc`
- `MR-STFT/lm`
- `MR-STFT/if`
- `MR-STFT/gd`
- `MR-STFT/complex`
- `STFT/low`, `MEL/low` (`0-500 Hz`)
- `STFT/mid`, `MEL/mid` (`500 Hz-4 kHz`)
- `STFT/high`, `MEL/high` (`4-12 kHz`)
- `STFT/air`, `MEL/air` (`12-20 kHz`; 24 kHz evaluation 中该频段为 `null`)
- `rFAD`，仅在传入 `--run-rfad` 时计算

导出的 WAV 会转换回 `24 kHz mono 5s`，方便和本项目 autoencoder 的 reconstruction 直接对比。

如果只想导出少量主观听感 sample，但仍然在完整 test set 上算指标：

```powershell
ae-evaluate --model same --data-root data/MTG-Jamendo-1000-24k-mono-5s --device cuda --max-audio-samples 32
```

`--max-audio-samples` 不能和 `--run-rfad` 同时使用，因为 rFAD 应该基于完整导出的 reference/system WAV 计算。

## 5. 计算 rFAD

先安装 eval extra 和 FADtk：

```powershell
pip install -e ".[eval]"
```

然后运行：

```powershell
ae-evaluate --model same --data-root data/MTG-Jamendo-1000-24k-mono-5s --device cuda --run-rfad --fad-model vggish
```

也可以在评测结束后手动运行：

```powershell
fadtk vggish outputs/evaluation/sa3_same/reference outputs/evaluation/sa3_same/same-s
fadtk vggish outputs/evaluation/sa3_same/reference outputs/evaluation/sa3_same/same-l
```

## 6. 准备主观听感对比

`mushra_command.txt` 中会写入只包含 SA3 baseline 的 MUSHRA 准备命令。若要和你训练的 autoencoder 一起比较，把自己的 reconstruction 目录作为额外 system 加进去：

```powershell
ae-mushra prepare `
  --reference-dir outputs/evaluation/sa3_same/reference `
  --system same-s=outputs/evaluation/sa3_same/same-s `
  --system same-l=outputs/evaluation/sa3_same/same-l `
  --system ours=outputs/evaluation/reconstruction `
  --output-dir outputs/mushra_sa3_vs_ours `
  --sample-rate 24000 `
  --max-trials 32
```

生成后，将 `outputs/mushra_sa3_vs_ours/manifest.json` 和 `stimuli/` 发给听评者。组织者保留 `key.json`，收集评分后汇总：

```powershell
ae-mushra summarize --scores outputs/mushra_sa3_vs_ours/scores.csv
```

## 7. 注意事项

- SA3 SAME 原生工作在 44.1 kHz stereo。本项目评测输入默认是 24 kHz mono，SA3 内部会按其模型配置处理，输出再转换回 24 kHz mono 后计算指标。
- 指标口径仍然和本项目 checkpoint evaluator 保持一致，因此可直接和 `ae-evaluate` 结果对比。
- `SAME-L` 更耗显存和时间；显存紧张时优先使用 `--chunked` 或先评测 `--model same-s`。
- 如果数据不是 `24 kHz mono 5s`，可以显式传入 `--sample-rate`、`--duration-seconds`、`--channels`，但与当前对比实验建议保持默认值一致。
