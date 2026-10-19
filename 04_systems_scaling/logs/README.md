# 实测日志

入口：[2026-10-19_stage4_systems_scaling.md](2026-10-19_stage4_systems_scaling.md)。

- `attention_correctness.json`：108 项对照及因果性检查。
- `benchmark_metrics.csv`：各 sequence/backend/mode 的 25 样本时延、吞吐、显存。
- `backend_unavailable.json`：SDPA FLASH 不可用与官方参考 N=64 不支持的错误。
- `attention_speed_report.md`：从 CSV 生成的单卡吞吐和加速比。
- `attention_trace.json`、`profiler_table.txt`、`profiler_events.json`：真实 torch.profiler。
- `profiling.html`、`profiling_screenshot.png`、`profiling_capture.json`：真实截图及来源。
- `scaling_training.jsonl`：训练 step loss/lr；`scaling_runs.csv`：75 次初始/最终验证 loss、参数、数据、seed、时间和显存。
- `scaling_config.json`、`scaling_fit.json`：实际配置、系数、bootstrap、留出误差。
- `environment.json`、`dependency_versions.json`：软硬件版本。
- `figure_*`、`*_audit.json`：图源、最终 PDF 字体和碰撞检查。
- `reproducibility_check.json`：归档证据验收，不等同于重新训练。

没有多卡或 ZeRO 实验。Profiler 插桩时间不能与独立 benchmark 混算。
