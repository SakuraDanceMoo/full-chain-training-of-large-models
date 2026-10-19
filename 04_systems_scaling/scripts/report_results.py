"""Build the speed report directly from measured CSV samples."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

STAGE = Path(__file__).resolve().parents[1]


def main():
    groups = defaultdict(list)
    with (STAGE/'logs/benchmark_metrics.csv').open(encoding='utf-8') as file:
        for row in csv.DictReader(file):
            groups[row['execution'],int(row['sequence']),row['method']].append(row)
    text = ['# 单卡注意力吞吐与加速比报告', '',
        '测量于 2026-10-19，Asia/Hong_Kong。RTX 4070 Ti 12GB；PyTorch 2.8.0+cu128、Triton 3.4.0。', '',
        'B=2、H=12、head_dim=64、FP16、causal=True、dropout=0。仅 forward；不含 backward、优化器或训练端到端速度。',
        '每项预热 20 次，25 个样本，每个样本 10 次调用；CUDA events 计时。CUDA Graph 每次 replay 包含 10 次调用，用于减小主机提交空隙。',
        '表中吞吐按 B×N/时间计算，不重复计数 heads；额外显存为单次调用相对调用前 live allocation 的峰值，不是进程总显存。', '',
        '当前 Windows PyTorch 未编译 SDPA FLASH；强制调用失败见 backend_unavailable.json。',
        'FlashAttention 对照使用随仓库归档的官方 Triton v3.2.0 FlashAttention-v2 教程，未将 math fallback 标成 FlashAttention，也未声称运行 flash-attn 包。', '',
        '## 原始提交与 CUDA Graph 对照', '',
        '| 模式 | N | 方法 | 中位延迟 ms | P10–P90 ms | tokens/s | 相对 eager | 额外显存 MiB |',
        '| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |']
    for mode in ['launch','cuda_graph']:
        for n in [64,128,256,512,1024,2048]:
            baseline = statistics.median(float(r['latency_ms']) for r in groups[mode,n,'eager'])
            for method in ['eager','sdpa_math','flashattention2_ref','triton_32','triton_64']:
                rows = groups.get((mode,n,method))
                if not rows:
                    continue
                values = [float(r['latency_ms']) for r in rows]
                median = statistics.median(values)
                lo,hi = np.quantile(values,[.1,.9])
                text.append(f'| {mode} | {n} | {method} | {median:.6f} | {lo:.6f}–{hi:.6f} | {2*n*1000/median:,.0f} | {baseline/median:.2f}× | {int(rows[0]["peak_extra_bytes"])/2**20:.2f} |')
    med = lambda method: statistics.median(float(r['latency_ms']) for r in groups['cuda_graph',2048,method])
    text += ['', '## 解释与 profiling', '',
        f'N=2048、CUDA Graph 下，BM64 相对 eager 为 {med("eager")/med("triton_64"):.2f}×，相对 SDPA math 为 {med("sdpa_math")/med("triton_64"):.2f}×；官方参考延迟/本实现延迟为 {med("flashattention2_ref")/med("triton_64"):.3f}。本实现接近官方参考，但没有超过它。', '',
        'BM64 相对 BM32 的提升来自单一 tile 配置对照，未进行 exhaustive autotuning。融合 QK/softmax/PV 与在线归一化避免了 N×N scores 的显式存储。',
        'SDPA math 的 FP16 输入可能使用 FP32 中间计算，与手写 eager 精度/中间存储策略不同，因此分别报告这两类基线。吞吐差异不能全部归因于某一项优化。', '',
        '![实际 profiler 截图](profiling_screenshot.png)', '',
        '截图由 Edge headless 打开真实 torch.profiler 表后截取；这不是 Nsight 截图。',
        '原始 [Chrome trace](attention_trace.json) 可导入 chrome://tracing 或本地 Perfetto；[表格](profiler_table.txt) 与 [事件](profiler_events.json) 支持复核。',
        'N=512，每个后端采样 5 次，JIT 已预热。_attention 和 _attn_fwd 为自写与官方参考的实际 CUDA 事件。',
        'Profiler 有插桩开销，只用于定位算子与内核，不能替代上表独立计时。', '',
        '多卡、DDP/ZeRO 与多卡加速比按用户要求取消。']
    (STAGE/'logs/attention_speed_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('Speed report written from measured samples.')


if __name__ == '__main__':
    main()
