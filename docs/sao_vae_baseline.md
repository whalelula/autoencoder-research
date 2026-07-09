# Stable Audio Open 1.0 VAE latent baseline evaluation

本文档说明如何在已经预处理好的 `MTG-Jamendo-1000-24k-mono-5s` test 集上，评测 Stable Audio Open 1.0（即 Stable Audio 1）的 VAE/pretransform latent autoencoder baseline，并导出和 `sa3_same_l_listen5` 完全一致的主观听感 sample。

当前项目统一评测命令为 `ae-evaluate --model sao`。默认使用：

- `--pretrained-name stabilityai/stable-audio-open-1.0`
- `--system-name stable-audio-open-1.0-vae-latent`
- `--output-dir outputs/evaluation/stable_audio_open_1_0_vae_latent`

## 1. 准备环境

建议在已有 `tokenizer-eval` 或其他包含 PyTorch/torchaudio 的环境中安装本项目和 `stable-audio-tools`：

```powershell
conda activate tokenizer-eval
pip install -e .
pip install git+https://github.com/Stability-AI/stable-audio-tools.git
```

如果 Stable Audio Open 1.0 权重托管在 gated/private Hugging Face repo，需要先登录：

```powershell
conda activate tokenizer-eval
hf auth login
```

`huggingface-cli login` 在新版 `huggingface_hub` 中已经废弃；如果看到 “Use `hf` instead”，请使用上面的 `hf auth login`。

首次运行会下载 `model_config.json` 和 `model.safetensors`/`model.ckpt` 到 Hugging Face cache。请确保 Hugging Face 访问权限正常，并准备足够磁盘空间。

## 2. 权重来源

默认从 Hugging Face repo 读取：

```text
stabilityai/stable-audio-open-1.0
```

如果该 repo 对当前账号不可见，命令会报 `Repository Not Found` 或权限错误。可以先做一次权限自检：

```powershell
hf auth whoami
hf download stabilityai/stable-audio-open-1.0 model_config.json
```

如果 `hf download` 仍然返回 `401 Unauthorized`，说明当前 token 无效、没有读权限，或该 repo 名称对当前账号不可见。确认账号已接受模型页面的访问条款，且 token 至少有 read 权限。

也可以使用本地权重目录，目录里需要包含：

```text
path/to/stable-audio-open-1.0/
  model_config.json
  model.safetensors
```

或：

```text
path/to/stable-audio-open-1.0/
  model_config.json
  model.ckpt
```

然后运行时显式传入：

```powershell
--pretrained-name path/to/stable-audio-open-1.0
```

## 3. 数据目录要求

评测命令不依赖 `configs/base.yaml`。只需要传入已经预处理好的 5 秒音频数据目录：

```text
data/MTG-Jamendo-1000-24k-mono-5s/
  manifests/
    test.jsonl
  audio/
    test/
      ...
```

默认假设数据是：

- `24000 Hz`
- `mono`
- `5.0 s`
- manifest 路径为 `DATA_ROOT/manifests/test.jsonl`

如果 manifest 不在默认位置，可以额外传入 `--manifest-dir`。

## 4. 跑完整 test set

使用默认 Stable Audio Open 1.0 repo：

```powershell
$env:PYTHONPATH='src'
conda run -n tokenizer-eval python -m ae_research.cli.evaluate `
  --model sao `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --device cuda `
  --batch-size 1 `
  --num-workers 0
```

如果已经 `pip install -e .` 并刷新了 entry point，也可以直接运行：

```powershell
ae-evaluate `
  --model sao `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --device cuda `
  --batch-size 1 `
  --num-workers 0
```

显存紧张时可以启用半精度：

```powershell
ae-evaluate `
  --model sao `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --device cuda `
  --batch-size 1 `
  --num-workers 0 `
  --half
```

调试时只跑少量 batch：

```powershell
ae-evaluate `
  --model sao `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --device cuda `
  --max-batches 2
```

常用可选参数：

- `--pretrained-name stabilityai/stable-audio-open-1.0`
- `--output-dir outputs/evaluation/stable_audio_open_1_0_vae_latent`
- `--system-name stable-audio-open-1.0-vae-latent`
- `--batch-size 1`
- `--num-workers 0`
- `--max-audio-samples 32`
- `--no-export-audio`

## 5. 复用 sa3_same_l_listen5 的 5 条 sample

为了和 `outputs/evaluation/sa3_same_l_listen5` 保持主观听感 sample 完全一致，直接复用它的 sampled manifest：

```powershell
$env:PYTHONPATH='src'
conda run -n tokenizer-eval python -m ae_research.cli.evaluate `
  --model sao `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --manifest-dir outputs/evaluation/sa3_same_l_listen5/sample_manifest `
  --output-dir outputs/evaluation/stable_audio_open_1_0_vae_latent_listen5 `
  --device cuda `
  --batch-size 1 `
  --num-workers 0
```

如果使用本地权重目录：

```powershell
$env:PYTHONPATH='src'
conda run -n tokenizer-eval python -m ae_research.cli.evaluate `
  --model sao `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --manifest-dir outputs/evaluation/sa3_same_l_listen5/sample_manifest `
  --output-dir outputs/evaluation/stable_audio_open_1_0_vae_latent_listen5 `
  --pretrained-name path/to/stable-audio-open-1.0 `
  --device cuda `
  --batch-size 1 `
  --num-workers 0
```

当前固定的 5 条 sample manifest 也已经复制到：

```text
outputs/evaluation/stable_audio_open_1_0_vae_latent_listen5/sample_manifest/test.jsonl
```

## 6. 输出内容

完整评测默认输出到：

```text
outputs/evaluation/stable_audio_open_1_0_vae_latent/
  metrics.json
  mushra_command.txt
  reference/
  stable-audio-open-1.0-vae-latent/
```

listen5 评测输出到：

```text
outputs/evaluation/stable_audio_open_1_0_vae_latent_listen5/
  metrics.json
  mushra_command.txt
  sample_manifest/
    test.jsonl
  reference/
  stable-audio-open-1.0-vae-latent/
```

`metrics.json` 包含：

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
- `latent_shapes`，记录 autoencoder latent 的形状

导出的 WAV 会转换回 `24 kHz mono 5s`，方便和 SA3 SAME-L、本项目 checkpoint reconstruction 直接对比。

## 7. 计算 rFAD

先安装 eval extra 和 FADtk：

```powershell
pip install -e ".[eval]"
```

然后运行：

```powershell
ae-evaluate `
  --model sao `
  --data-root data/MTG-Jamendo-1000-24k-mono-5s `
  --device cuda `
  --run-rfad `
  --fad-model vggish
```

也可以在评测结束后手动运行：

```powershell
fadtk vggish outputs/evaluation/stable_audio_open_1_0_vae_latent/reference outputs/evaluation/stable_audio_open_1_0_vae_latent/stable-audio-open-1.0-vae-latent
```

`--run-rfad` 不能和 `--max-audio-samples` 同时使用，因为 rFAD 应该基于完整导出的 reference/system WAV 计算。

## 8. 准备主观听感对比

`mushra_command.txt` 会写入只包含 Stable Audio Open 1.0 VAE baseline 的 MUSHRA 准备命令。若要和 SA3 SAME-L 或本项目 autoencoder 一起比较，可以手动加入额外 system：

```powershell
ae-mushra prepare `
  --reference-dir outputs/evaluation/stable_audio_open_1_0_vae_latent_listen5/reference `
  --system stable-audio-open-1.0-vae-latent=outputs/evaluation/stable_audio_open_1_0_vae_latent_listen5/stable-audio-open-1.0-vae-latent `
  --system same-l=outputs/evaluation/sa3_same_l_listen5/same-l `
  --system ours=outputs/evaluation/reconstruction `
  --output-dir outputs/mushra_sao1_vae_vs_sa3_vs_ours `
  --sample-rate 24000 `
  --max-trials 5
```

生成后，将 `outputs/mushra_sao1_vae_vs_sa3_vs_ours/manifest.json` 和 `stimuli/` 发给听评者。组织者保留 `key.json`，收集评分后汇总：

```powershell
ae-mushra summarize --scores outputs/mushra_sao1_vae_vs_sa3_vs_ours/scores.csv
```

## 9. 注意事项

- Stable Audio Open 1.0 VAE/pretransform 原生采样率和声道数由 `model_config.json` 决定；评测脚本会把输入从 `24 kHz mono` 转到模型格式，decode 后再转回 `24 kHz mono 5s` 计算指标。
- 如果 `stabilityai/stable-audio-open-1.0` 报 404，不一定是代码问题，可能是 repo 名称不可见、private/gated 权限不足，或权重尚未缓存。
- 如果导入 `pywt` 时报 `numpy.dtype size changed`，说明 `PyWavelets` wheel 和当前 `numpy` ABI 不匹配。可在当前环境中运行 `python -m pip install --upgrade --force-reinstall "PyWavelets>=1.8.0"`，然后用 `python -c "import pywt, numpy; print(pywt.__version__, numpy.__version__)"` 验证。
- 本地权重目录方式适合私有权重或离线机器；只要目录中有 `model_config.json` 和 `model.safetensors`/`model.ckpt` 即可。
- `batch-size` 建议先用 `1`。确认显存足够后再增大。
- 指标口径和本项目 checkpoint evaluator、SA3 SAME evaluator 保持一致，可直接横向比较。
