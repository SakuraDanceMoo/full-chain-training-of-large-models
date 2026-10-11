# 阶段三·第一部分：英法歌曲、关闭 AMP

状态：**已实现并可复现**。这是阶段三的前半部基线：只使用 `eng_fra_songs.txt`，不启用 AMP，也不启用 activation checkpointing。它与第二部分的原始语料、tokenizer、输出目录和 checkpoint 完全隔离。

## 工程内容

- `config.json`：0.1B profile、CUDA、seed、batch/梯度累积和开关。
- `scripts/train.py`：复用阶段二 GPT，使用本目录数据管线和输出路径。
- `scripts/data_pipeline.py`：阶段一 BPE tokenizer + 可校验 token stream。
- `data/raw/eng_fra_songs.txt`：唯一训练语料，SHA-256 为 `4d927cb4d2bb2ebbdba1b9914175c05a308a37c5bd0d9ea5b39a03aabafd6ea3`。
- `logs/`：配置、训练 CSV、独立 PPL 和实验记录。
- `curves/training_loss.png`：真实训练曲线。
- `models/best_checkpoint.json` + 9 个分片：验证 PPL 最佳模型。

## 一键复现

从仓库根目录执行：

```powershell
# 已有项目环境时使用父目录的 CUDA/Torch 配置；普通环境也可直接运行后两行
. ..\Use-ProjectEnv.ps1
..\.venv\Scripts\python.exe 03_pretraining/part1_eng_fra_no_amp/scripts/train.py --profile 0.1b --steps 20 --eval-interval 10 --eval-iters 2 --device cuda --no-amp --no-activation-checkpointing --block-size 128 --batch-size 1 --grad-accum 4
..\.venv\Scripts\python.exe 03_pretraining/part1_eng_fra_no_amp/scripts/evaluate.py --checkpoint 03_pretraining/part1_eng_fra_no_amp/models/best_checkpoint.json --corpus 03_pretraining/part1_eng_fra_no_amp/data/raw --iters 4 --batch-size 1 --device cuda --output 03_pretraining/part1_eng_fra_no_amp/logs/evaluation_ppl.json
..\.venv\Scripts\python.exe 03_pretraining/part1_eng_fra_no_amp/scripts/check_repro.py
```

若从未生成过 tokenizer 或 token cache，训练脚本会在本目录 `models/` 与 `data/processed/` 中自动生成，不读取第二部分文件。

## 实际结果

模型为 12 层、12 heads、768 hidden，参数量 85,547,520；共 23,978 tokens，20 个 optimizer steps，`amp_enabled=false`，`activation_checkpointing=false`。验证 PPL 从 648.66（step 0）降至 225.91（step 20）；独立评测 PPL 为 200.90。

详见 [`logs/experiment.md`](logs/experiment.md)、[`logs/training_metrics.csv`](logs/training_metrics.csv) 和 [`curves/training_loss.png`](curves/training_loss.png)。
