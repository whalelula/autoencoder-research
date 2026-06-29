# Semantic Audio Autoencoder

这是一个面向音乐生成研究的最小完整基线：冻结 MERT 作为 semantic encoder，用一层可训练
linear projection 和镜像 MERT feature encoder 的 7 层转置卷积重建波形。当前阶段只训练和评估
autoencoder；`ae_research.diffusion` 仅定义后续 latent diffusion 的接口边界。

研究假设是：与从零学习的 audio codec latent 相比，预训练 SSL semantic latent 是否能让下游
diffusion 更快收敛，并提高最终生成质量。

## 设计边界

- MERT-v1-95M 和 MERT-v1-330M 都使用 24 kHz 输入，分别输出 768 和 1024 维、约 75 Hz
的特征。本项目可切换两者，encoder 参数始终冻结。
- decoder 从所选 checkpoint 的 MERT config 自动读取 `hidden_size`、`conv_dim`、
`conv_kernel` 和 `conv_stride`，构造 `Linear + 7 x ConvTranspose1d`。因此 95M/330M
切换时无需手工同步 decoder 参数。
- SAME 式 KL 施加在可训练 linear projection 的输出，而不是冻结 MERT 的输出；否则该 loss
对任何可训练参数都没有梯度。
- 默认以 24 kHz 训练。配置允许其他数据采样率，但模型会在 MERT/decoder 前后重采样；高于
24 kHz 不会恢复 MERT 从未观察到的高频信息，因此正式实验建议保持 24 kHz。
- MUSHRA 是人工主观测试，不是可由模型自动计算的单一指标。本项目负责生成 reference、
hidden reference、3.5/7 kHz anchors、盲化清单并汇总评分。

参考实现与定义：

- [MERT model card](https://huggingface.co/m-a-p/MERT-v1-330M)
- [SAME paper](https://arxiv.org/abs/2605.18613)
- [MTG-Jamendo official dataset](https://mtg.github.io/mtg-jamendo-dataset/)
- [Microsoft FADtk](https://github.com/microsoft/fadtk)
- [MuQ-Eval](https://github.com/dgtql/MuQ-Eval)



## 1. 安装

建议 Python 3.10 或 3.11。先按机器 CUDA 版本安装固定配套的 PyTorch/torchaudio，再安装本项目；
`pyproject.toml` 故意不让 pip 自动选择 CUDA 大包。

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# CUDA 12.6；若驱动环境不同，请换成 PyTorch 官方对应 index。
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# CPU-only
# pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

Linux/bash：

```bash
python -m venv .venv
source .venv/bin/activate

# CUDA 12.6；若驱动环境不同，请换成 PyTorch 官方对应 index。
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# CPU-only
# pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

大陆网络可先使用 PyPI/Hugging Face 镜像。PyPI 镜像不会加速 GitHub URL。

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "D:\hf-cache"  # 放在空间充足、可持久化的盘
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

AutoDL/Linux：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
```

如果 editable install 的依赖解析失败，可先逐项安装
`PyYAML requests tqdm soundfile tensorboard matplotlib "transformers>=4.40,<4.47"`，再运行
`pip install -e . --no-deps`。

## 2. 抽样并下载 MTG-Jamendo

先在 [Jamendo developer portal](https://developer.jamendo.com/) 获取 client id。脚本会下载官方
metadata，按 tag/时长筛选后随机抽样，再通过 Jamendo API 下载所选单曲。大文件先写入 `.part`，
支持 Range 续传和重试；下载成功后才原子改名。

PowerShell：

```powershell
$env:JAMENDO_CLIENT_ID = "your_client_id"
ae-prepare --output-root data --num-tracks 1000 --seed 42 --workers 8

# 只抽样包含指定 tag 的曲目；多个 --tag 默认是“任一匹配”
ae-prepare --output-root data --num-tracks 500 `
  --tag "genre---electronic" --tag "instrument---piano"
```

Linux/bash：

```bash
export JAMENDO_CLIENT_ID="your_client_id"
ae-prepare --output-root data --num-tracks 1000 --seed 42 --workers 8
```

若 metadata 的 GitHub 直链在当前网络不可达，可手动下载官方仓库后显式传入文件：

```powershell
git clone --depth 1 https://github.com/MTG/mtg-jamendo-dataset.git
ae-prepare --metadata .\mtg-jamendo-dataset\data\raw_30s.tsv `
  --output-root data --num-tracks 1000
```

已有官方 archive 解压后的音频时，不需要 API：

```powershell
ae-prepare --output-root data --num-tracks 1000 --audio-root E:\mtg-jamendo\raw_30s\audio
```

输出为 `data/manifests/{train,val,test}.jsonl`，严格按 track 数量做可复现的 7:1:2 切分。
默认 track-level 切分；可加 `--group-by-artist` 避免 artist 泄漏（比例会取最接近 7:1:2）。
MTG-Jamendo 仅限非商业研究使用，且每首音频仍受各自 Creative Commons license 约束。

## 3. 训练与查看

```powershell
ae-train --config configs/base.yaml
tensorboard --logdir runs
```

切换 330M：

```yaml
model:
  mert_name: m-a-p/MERT-v1-330M
```

训练会记录 total、MR-STFT、KL、SI-SDR 以及 MR-STFT 的 SC/LM/IF/GD/complex 分项；按配置定期
验证、保存可恢复 checkpoint，并每若干轮导出 reference/reconstruction WAV。训练结束自动输出
`history.csv` 和 `loss_curves.png`。

## 4. Test evaluation

```powershell
ae-evaluate --config configs/base.yaml --checkpoint runs/mert95m/checkpoints/best.pt
```

内置计算 SI-SDR、log-mel L1 和 SAME MR-STFT，并可导出配对音频。rFAD 使用 FADtk：

```powershell
pip install -e ".[eval]"
fadtk vggish outputs/evaluation/reference outputs/evaluation/reconstruction
```

这里的 reference/reconstruction FAD 即 rFAD。下游 diffusion 完成后：

```powershell
# gFAD
fadtk vggish /path/to/test_reference /path/to/generated

# FAD-CLAP；论文中应同时报告具体 embedding 名称
fadtk clap-laion-music /path/to/test_reference /path/to/generated
```

MuQ-Eval 当前官方仓库不是可直接依赖的稳定 PyPI 包，建议独立环境按其 README 安装后对
`/path/to/generated` 批量评分，避免它的依赖约束污染训练环境。

准备和汇总 MUSHRA：

```powershell
ae-mushra prepare --reference-dir outputs/evaluation/reference `
  --system reconstruction=outputs/evaluation/reconstruction `
  --output-dir outputs/mushra --sample-rate 24000

# 让听者按 outputs/mushra/scores_template.csv 填写 0-100 分后
ae-mushra summarize --scores outputs/mushra/scores.csv
```



## 5. 测试

```powershell
pytest
```

未安装 PyTorch 时，纯数据逻辑测试仍可运行，模型/loss 测试会自动跳过。