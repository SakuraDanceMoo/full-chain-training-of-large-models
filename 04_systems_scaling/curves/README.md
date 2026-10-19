# 实测曲线

每张图一个面板，输出 PNG（300 dpi）、可编辑文字 PDF/SVG；三个格式为同一图的交付格式。

- `attention_launch`：包含主机提交空隙的 CUDA 时延。
- `attention_cuda_graph`：CUDA Graph replay 时延，每次 replay 包含 10 调用。
- `scaling_N` / `scaling_D`：一维拟合、整档留出与远距离外推；点为 3-seed mean±sample SD。
- `scaling_holdout`：16 个拟合 cell 与 9 个整档留出 cell 的预测对照。
- `scaling_law`：联合模型不同 D 的参数外推，虚线为未实测 N。

生成入口 `scripts/plot_results.py`，输入为真实 CSV 和拟合 JSON。证据边界与审查见 `../logs/figure_qa.md`。
