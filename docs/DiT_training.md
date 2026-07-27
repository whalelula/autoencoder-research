# DiT 训练与评估

本文说明如何使用仓库中的 text-conditioned audio DiT，评估已训练 autoencoder 的下游生成能力。当前流程只负责接通数据清单、冻结的 T5Gemma 文本条件、DiT 训练、validation 试听和 test gFAD；不会自动下载 MTG-Jamendo 音频，也不会自动启动训练。

参考配置是 `[configs/dit_mert330m_1k_5s.yaml](../configs/dit_mert330m_1k_5s.yaml)`。其中 autoencoder 配置和 checkpoint 路径必须换成实际待评估模型。

## 1. 环境安装

PyTorch 和 torchaudio 的版本及 CUDA 构建必须匹配，因此它们不放入项目的通用依赖。先按机器环境手动安装，再安装项目。

CUDA 12.6 示例：

```bash
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[dit,eval,dev]"
```

仅 CPU 示例：

```bash
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dit,eval,dev]"
```

中国大陆网络可为普通 Python 包使用镜像；PyTorch wheel 仍建议使用上面的官方 index：

```bash
python -m pip install -e ".[dit,eval,dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

若可选依赖解析失败，可先分别安装 `sentencepiece`、`fadtk`、`pytest`，再执行 `python -m pip install -e . --no-deps`。安装后可用以下命令确认入口已经注册：

```bash
ae-prepare-dit-tags --help
ae-dit-train --help
ae-dit-evaluate --help
```



## 2. 数据和标签准备

输入使用项目现有的 MTG-Jamendo 抽样清单。清单记录需要包含音频路径以及 genre、instrument、mood 标签；预处理会规范化标签，按 `genre, instrument, mood` 的顺序用英文逗号拼接，并保存为 `text` 字段。

当前本地抽样清单中成功可用的是 767 条，而不是完整 1000 条：train/validation/test 分别为 536/76/155；其余 233 条属于既有下载或预处理失败记录。DiT 不会补下载这些音频。

先只生成带文本的 DiT 清单（不加载 Transformers，也不访问网络）：

```bash
ae-prepare-dit-tags --config configs/dit_mert330m_1k_5s.yaml --manifests-only
```

随后用冻结的最小 T5Gemma 版本 `google/t5gemma-s-s-ul2` 预计算 token-level text embeddings：

```bash
ae-prepare-dit-tags --config configs/dit_mert330m_1k_5s.yaml --device cuda
```

T5Gemma 是受访问许可约束的 Hugging Face 模型。首次使用前需要在模型页面接受许可并登录：

```bash
huggingface-cli login
```

若模型已放入本机或共享缓存，可严格禁止联网：

```bash
ae-prepare-dit-tags --config configs/dit_mert330m_1k_5s.yaml --device cuda --local-files-only
```

建议把 Hugging Face 缓存放在持久磁盘。PowerShell 示例：

```powershell
$env:HF_HOME = "D:\hf-cache"
$env:HF_HUB_CACHE = "D:\hf-cache\hub"
```

Linux 示例：

```bash
export HF_HOME=/data/hf-cache
export HF_HUB_CACHE=/data/hf-cache/hub
```

如需 Hugging Face 镜像，可按所在环境设置 `HF_ENDPOINT`；但镜像不能绕过模型许可。`--manifests-only` 不会生成 embeddings，因此它不能单独满足训练输入。`ae-dit-train` 只读取已存在的音频、DiT manifests 和 embedding cache，缺失时会直接报错，不会隐式下载或编码。

## 3. 条件与模型结构

音频先经过冻结的 autoencoder encoder 得到 clean latent `x1`。DiT 接收带噪 latent，结构为：

```text
latent
  -> residual 1x1 convolution
  -> linear projection to DiT dimension
  -> N x Transformer block (default N=11)
  -> linear projection to autoencoder latent dimension
  -> residual 1x1 convolution
  -> predicted clean latent x_pred
```

每个 Transformer block 使用 self-attention、feed-forward 和 text cross-attention。T5Gemma 输出的 token embeddings先通过线性投影从 `text_dim` 映射到 DiT 的 `model_dim`，再作为 cross-attention 的 K/V；它们不直接进入 AdaLN。

扩散 timestep 与生成 duration 先分别编码，再合成为 global condition。连续标量使用 Fourier embedding，是因为一组不同频率的正弦/余弦能把单个数值展开为平滑的多尺度表示，使网络更容易区分相近 timestep，同时学习低频趋势和高频变化。global condition 经 MLP 为每个 block 生成 AdaLN 参数。

每个 block 的 AdaLN 输出 `6 × model_dim`，对应 self-attention 分支和 feed-forward 分支各三组逐通道参数：`shift`、`scale`、`gate`。cross-attention 采用独立的归一化和门控，不把 text embedding 加到 local latent 上。当前 Vanilla 版本没有 Stable Audio 3 的 local-additive conditioning；将来若加入逐时间对齐条件，应先投影到 `model_dim` 后与 latent token 相加，它与全局 AdaLN、文本 cross-attention 是三条不同的条件路径。

## 4. Flow matching 与 loss

训练从 clean latent `x1` 和高斯噪声 `x0` 构造：

```text
xt = (1 - t) * x1 + t * x0
vt = (xt - x1) / clamp_min(t, t_eps)
v_pred = (xt - x_pred) / clamp_min(t, t_eps)
L_x = mean((v_pred - vt)^2)
```

模型直接预测 `x_pred`，但 loss 在等价的 velocity 空间计算。这样保留 x-prediction 的输出语义，同时使用 flow-matching velocity 监督。配置中的 `t_eps` 防止除零。

当前辅助项名为 `repa_internal_guidance`：中间层 latent 表示与 clean latent 的投影目标做对齐，并乘以 `dit.repa.loss_weight`。总损失为：

```text
L_total = L_x + lambda_repa * L_repa
```

这里没有实现一个额外、独立的 internal-guidance loss，也没有 CFG。原因是 RAE/RAEv2 语境中的 representation alignment 可在采样时提供 internal/auto guidance，但它和“再添加一项 guidance training loss”并非天然等价。为避免未经验证地重复约束，本版只保留一个明确的 REPA 对齐项；日志中的历史名称 `repa_internal_guidance` 指的就是该项。

## 5. Timestep、优化器与学习率

`t` 来自 `sigmoid(z), z ~ N(0, 1)`。样本在 0.075 处截断并重新缩放到 `[0, 1]`，去除极低噪声区域，把训练预算集中在中高噪声区间。

优化器采用 Muon + AdamW hybrid：

- attention Q/K/V/out 和 feed-forward 矩阵参数使用 Muon，默认 learning rate `1e-5`、momentum `0.95`；
- bias、归一化、卷积、embedding、输入/输出投影等其余参数使用 AdamW，默认 learning rate `1e-6`、betas `(0.9, 0.95)`、weight decay `0.01`。

学习率沿用 autoencoder 的 warmup-cosine 方案。前 `warmup_steps` 线性升温，之后余弦衰减到各参数组初始 learning rate 的 `min_lr_ratio`。两个 optimizer 参数组共享同一个倍率，因此保留 10:1 的基础 learning-rate 比例。scheduler 每次 optimizer update 后推进一次，gradient accumulation 的 micro-batch 不单独推进。

## 6. 训练

先检查配置中的路径：

- `data.dit_manifest_dir`：带 `text` 与 embedding 引用的 train/validation/test JSONL；
- `conditioning.text_encoder.cache_dir`：预计算的 token embeddings；
- `autoencoder.config_path` 与 `autoencoder.checkpoint_path`：待评估的冻结 autoencoder；
- `dit.text_dim`：必须与缓存 embedding 的最后一维一致；
- `data.sample_rate`、`channels`：必须与 autoencoder 一致。

启动训练：

```bash
ae-dit-train --config configs/dit_mert330m_1k_5s.yaml --device cuda
```

断点续训通过配置指定：

```yaml
training:
  resume_from: runs/dit_mert330m_1k_5s/checkpoints/latest.pt
```

TensorBoard：

```bash
tensorboard --logdir runs/dit_mert330m_1k_5s/tensorboard_dit
```

训练会汇报 `total`、`x_prediction`、`repa_internal_guidance` 以及 Muon/AdamW 的当前 learning rate，同时写入 TensorBoard 和 `dit_history.csv`。`training.loss_curve_every_steps` 控制每隔多少个 optimizer step 原子更新 `loss_curves.png`；未配置时默认使用 `log_every_steps`。

每次 validation 使用固定随机种子选择同一组 5 条 prompt，并使用 DiT 的 padding collator 对不同长度的 text tokens 组 batch。生成结果经冻结 decoder 保存到 `listening/step_*`，其中包含 reference、generated WAV 和记录 prompt/seed/track id 的 metadata，便于跨 step 人工对比。checkpoint 会保存这 5 条索引和随机种子，resume 后保持一致。

本训练流程明确不包含 inpainting、minibatch optimal transport coupling、variable-length training、distillation、adversarial post-training 或 CFG。

## 7. Test evaluation 与 gFAD

单独运行完整 test 生成并计算 gFAD：

```bash
ae-dit-evaluate \
  --config configs/dit_mert330m_1k_5s.yaml \
  --checkpoint runs/dit_mert330m_1k_5s/checkpoints/best.pt \
  --device cuda \
  --run-gfad
```

Windows PowerShell 可写成一行，或用反引号换行。只生成音频、不运行 FADtk：

```bash
ae-dit-evaluate --config configs/dit_mert330m_1k_5s.yaml --checkpoint runs/dit_mert330m_1k_5s/checkpoints/best.pt --device cuda --skip-gfad
```

当 `evaluation.run_after_training: true` 时，trainer 在完成训练后会直接调用同一个 evaluator，并使用 best checkpoint。若当前环境未安装 FADtk，可把 `evaluation.gfad.enabled` 或 `run_after_training` 设为 `false`，之后再独立评估。

评估目录包含：

- `reference/`：test 原音频；
- `generated/`：相同 prompt 对应的生成音频；
- `prompts.jsonl`：文件名、track id 与 text；
- `gfad_vggish.txt`：FADtk 原始输出；
- `summary.json`：生成数量、目录和最终 gFAD。

gFAD 比较真实 test 音频集合与生成集合，是无配对的分布距离；数值只应在相同数据划分、采样率、时长和 FAD feature model 下横向比较。`evaluation.max_batches` 仅用于 smoke test，正式结果应保持为 `null`。

## 8. 常见问题

- 提示 text embedding 缺失：先运行不带 `--manifests-only` 的 tag preparation；若处于离线环境，确认模型已缓存后加 `--local-files-only`。
- T5Gemma 返回 401/403：先接受 Hugging Face 模型许可并登录；镜像地址不能替代授权。
- text dimension 不匹配：让 `dit.text_dim` 与缓存 metadata 一致，改模型后重新生成整个 cache，避免混用。
- autoencoder sample rate/channel 不匹配：DiT 不会静默重采样到另一个 codec 规格，应改用匹配的音频清单或 checkpoint。
- 找不到 `fadtk`：安装 `.[eval]`，或先使用 `--skip-gfad`。
- 显存不足：降低 `training.batch_size`、提高 gradient accumulation，或使用 `bf16`；不要通过缩短 text cache 的 tensor 维度规避配置检查。

