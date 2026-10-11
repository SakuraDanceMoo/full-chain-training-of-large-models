# 阶段三：0.1B 端到端预训练（拆分为两部分）

阶段三现在由两套相互隔离、可独立复现的预训练工程组成。每套工程都有自己的 `config.json`、`scripts/`、`data/raw/`、`data/processed/`、`logs/`、`curves/` 和 `models/`；两套原始语料、tokenizer、token cache 和 checkpoint 不共享。

| 部分 | 训练语料 | AMP | activation checkpointing | 状态 |
| --- | --- | --- | --- | --- |
| [第一部分：英法歌曲、关闭 AMP](part1_eng_fra_no_amp/README.md) | 仅 `eng_fra_songs.txt` | 关闭 | 关闭 | 已实现，20 steps |
| [第二部分：单卡 CUDA](part2_mixed_amp_checkpointing/README.md) | 维基 + 斯瓦希里语 + Python 代码，不含英法歌曲 | 开启 | 开启 | 已实现，50 steps |

## 复现入口

从仓库根目录分别执行对应 README 中的命令。两部分不能通过复制或合并 `data/raw` 来复现；第二部分的 `check_repro.py` 会拒绝英法歌曲文件。

```powershell
..\.venv\Scripts\python.exe 03_pretraining/part1_eng_fra_no_amp/scripts/check_repro.py
..\.venv\Scripts\python.exe 03_pretraining/part2_mixed_amp_checkpointing/scripts/check_repro.py
```

## 共享实现与历史文件

两个子工程各自复制了训练脚本和数据管线，模型结构复用 `02_gpt/scripts/minigpt.py`，BPE tokenizer 复用 `01_tokenizer/scripts/tokenizer.py`。阶段三根目录下的早期聚合脚本和日志保留作历史审计；新的复现实验以两个子目录为准，尤其不能使用早期把英法歌曲并入 mixed corpus 的命令。

## 验收

根仓库审计：

```powershell
python C:/Users/Administrator/.codex/skills/push-to-github/scripts/audit_repository.py --root .
```

两部分的真实结果、seed、设备、PPL、曲线和分片 checkpoint 分别记录在各自的 `logs/experiment.md`、`logs/training_config.json`、`logs/training_metrics.csv`、`logs/evaluation_ppl.json`、`curves/training_loss.png` 和 `models/best_checkpoint.json`。
