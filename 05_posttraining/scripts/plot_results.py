"""Summarize measured runs, paired uncertainty and evidence-based rejection rules."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_pipeline import STAGE,write_json


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    config=json.loads((STAGE/'config.json').read_text(encoding='utf-8'))
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((STAGE/'logs/runs').glob('*/result.json'))
          if p.parent.name!='warmup']
    keys=[('sft','lora'),('dpo','lora'),('rlvr','lora'),('sft','qlora'),('sft','adapter')]
    groups=[[r for r in rows if (r['method'],r['peft'])==key] for key in keys]
    if any(len(g)!=len(config['seeds']) for g in groups):
        raise ValueError('incomplete registered experiment matrix')
    baseline=json.loads((STAGE/'logs/anchor_test_predictions.json').read_text(encoding='utf-8'))
    ref={r['seed']:r for r in groups[0]}
    rng=np.random.default_rng(744)
    summary=[]
    for key,group in zip(keys,groups):
        method,kind=key
        acc=[r['test_accuracy']*100 for r in group]
        differences=[]
        # Paired case-and-seed bootstrap; exploratory uncertainty with only three seeds.
        for r in sorted(group,key=lambda r:r['seed']):
            read=lambda run:json.loads((STAGE/f'logs/runs/{run["run_id"]}/test_predictions.json').read_text(encoding='utf-8'))['predictions']
            pred=read(r); peer=read(ref[r['seed']])
            assert [p['id'] for p in pred]==[p['id'] for p in peer]
            differences.append([float(b['correct'])-float(a['correct']) for a,b in zip(pred,peer)])
        diff=np.asarray(differences)
        boot=[]
        for _ in range(2000):
            seeds=rng.integers(0,diff.shape[0],diff.shape[0])
            cases=rng.integers(0,diff.shape[1],diff.shape[1])
            boot.append(diff[seeds][:,cases].mean()*100)
        lo,hi=np.percentile(boot,[2.5,97.5])
        drop=float(diff.mean()*100)
        verdict='reject' if drop>config['drop_threshold_pp'] else 'retain'
        summary.append({'method':method,'peft':kind,'seeds':len(group),
                        'accuracy_mean_pct':statistics.mean(acc),'accuracy_std_pp':statistics.stdev(acc),
                        'format_mean_pct':statistics.mean(r['test_format_rate']*100 for r in group),
                        'nll_mean':statistics.mean(r['test_response_nll'] for r in group),
                        'peak_allocated_mib_mean':statistics.mean(r['peak_allocated_mib'] for r in group),
                        'peak_reserved_mib_mean':statistics.mean(r['peak_reserved_mib'] for r in group),
                        'train_seconds_mean':statistics.mean(r['train_seconds'] for r in group),
                        'train_seconds_std':statistics.stdev(r['train_seconds'] for r in group),
                        'trainable_parameters':group[0]['trainable_parameters'],
                        'drop_vs_sft_lora_pp':drop,'drop_bootstrap_ci_low_pp':float(lo),
                        'drop_bootstrap_ci_high_pp':float(hi),'decision':verdict})
    flat=[]
    for r in rows:
        entry={k:v for k,v in r.items() if k!='quantization'}
        entry['delta_vs_anchor_pp']=(r['test_accuracy']-baseline['accuracy'])*100
        entry['drop_vs_same_seed_sft_lora_pp']=(ref[r['seed']]['test_accuracy']-r['test_accuracy'])*100
        flat.append(entry)
    write_csv(STAGE/'logs/evaluation_metrics.csv',flat)
    write_csv(STAGE/'logs/summary_metrics.csv',summary)
    write_json(STAGE/'logs/summary.json',{'anchor_accuracy':baseline['accuracy'],'summary':summary,
                                       'drop_rule':f'observed mean drop > {config["drop_threshold_pp"]} pp relative to SFT-LoRA; uncertainty reported, not a significance claim',
                                       'bootstrap':'2000 paired case-and-seed resamples, seed 744; exploratory with 3 seeds'})
    curves=STAGE/'curves';curves.mkdir(exist_ok=True)
    labels=['SFT/LoRA','DPO/LoRA','RLVR/LoRA','SFT/QLoRA','SFT/Adapter']
    colors=['#386cb0','#7fc97f','#fdc086','#beaed4','#f0027f']
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.bar(labels,[s['accuracy_mean_pct'] for s in summary],yerr=[s['accuracy_std_pp'] for s in summary],color=colors,capsize=4)
    ax.axhline(baseline['accuracy']*100,ls='--',color='black',label='Common SFT anchor')
    ax.set(ylabel='Test exact-match accuracy (%)',ylim=(0,105),title='Fixed base and eval: 3 seeds; bars = mean +/- sample SD')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(curves/'alignment_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    peft=[summary[i] for i in [0,3,4]]
    for ax,metric,label in zip(axes,['peak_allocated_mib_mean','train_seconds_mean'],['Peak allocated CUDA memory (MiB)','Training time (s), excluding eval/setup']):
        ax.bar(['LoRA','QLoRA','Adapter'],[s[metric] for s in peft],color=[colors[i] for i in [0,3,4]])
        ax.set(ylabel=label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same SFT data, rank/width parameter-matched, 96 updates, 3 seeds')
    fig.tight_layout();fig.savefig(curves/'peft_resources.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,method in zip(axes,['sft','dpo','rlvr']):
        for seed in config['seeds']:
            path=STAGE/f'logs/runs/{method}_lora_s{seed}/training.jsonl'
            history=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
            smooth=np.convolve([r['loss'] for r in history],np.ones(8)/8,'valid')
            ax.plot(np.arange(8,len(history)+1),smooth,label=str(seed))
        ax.set(title=method.upper(),xlabel='Optimizer update',ylabel='Own objective, 8-step mean')
        ax.grid(alpha=.2)
    axes[0].legend(title='Seed');fig.tight_layout();fig.savefig(curves/'training_loss.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for i,(key,group) in enumerate(zip(keys,groups)):
        vals=[]
        for r in group:
            hist=[json.loads(l) for l in (STAGE/f'logs/runs/{r["run_id"]}/training.jsonl').read_text(encoding='utf-8').splitlines()]
            vals.append([(p['step'],p['val_accuracy']) for p in hist if 'val_accuracy' in p])
        a=np.array(vals)
        ax.plot(a[0,:,0],a[:,:,1].mean(0)*100,'o-',label=labels[i],color=colors[i])
    ax.set(xlabel='Optimizer update',ylabel='Validation exact match (%)',title='Fixed validation set, no checkpoint/HP selection')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(curves/'validation_accuracy.png',dpi=180);plt.close(fig)
    lines=['# 阶段五后训练与 PEFT 对照实验报告','',
           '## 数据与统一协议','',
           '复用阶段三的 85,547,520 参数 GPT 与 512 词表 BPE。先进行一份共享 SFT 预热，再合并其 LoRA 增量作为全部实验的固定基座。任务是 00–31 的整数比较，回答 0 或 1，按有序数对去重并分层划分训练、验证、测试集。数据、划分、基座分片及 tokenizer 的 SHA-256 见 manifest 与 protocol。','',
           f'固定基座测试准确率为 {baseline["accuracy"]*100:.2f}%。评测对完整词表进行贪心生成，最多 2 tokens，解码后精确匹配；不限制答案候选、不用 teacher forcing 准确率替代生成准确率。只训练回复和 EOS，提示词与 padding 不计 SFT loss。最终固定预算 checkpoint 用于测试，验证集仅监控，不选 checkpoint，不根据测试调参。','',
           '## 超参数','',
           '| 项目 | 设置 |','| --- | --- |',
           f'| Seed | {config["seeds"]} |',
           f'| 共同预热 | LoRA，{config["warmup_steps"]} 步，lr={config["warmup_lr"]}，seed={config["warmup_seed"]} |',
           f'| 各实验预算 | {config["steps"]} 次更新；SFT/DPO batch={config["batch_size"]}；RLVR {config["rlvr_prompt_batch"]} prompts × {config["rlvr_group_size"]} rollouts |',
           f'| 学习率 | SFT={config["sft_lr"]}；DPO={config["dpo_lr"]}；RLVR={config["rlvr_lr"]} |',
           f'| 对齐参数 | DPO beta={config["dpo_beta"]}；RLVR KL={config["rlvr_kl_beta"]}，clip={config["rlvr_clip"]} |',
           f'| PEFT | qkv/proj LoRA rank={config["rank"]}，alpha/r=2；逐 block 残差 Adapter width={config["adapter_width"]}；可训练参数均为 442,368 |',
           '| 精度与优化器 | BF16 基座和算子，FP32 PEFT 参数及 AdamW 状态，lr 恒定，weight decay=0，clip norm=1，activation checkpointing |',
           '| QLoRA | bitsandbytes 0.48.2，NF4，双重量化，UINT8 packed；embedding/共享输出头保留 BF16；标准 AdamW，非 paged optimizer |','',
           'SFT、DPO 与 RLVR 都从同一固定基座上新建零输出增量。DPO 使用基座的冻结参考 log-prob 缓存，目标是正确回答优于相反回答。RLVR 在当前策略上从完整词表采样，以精确验分的 0/1 奖励构造组内标准化 advantage，使用 clipped policy objective 和冻结基座 KL；每批只更新一次，不使用标签交叉熵。','',
           '## 统一结果表','',
           '| 方法 | 测试准确率 %，均值±SD | 掉点 pp，相对 SFT-LoRA | 掉点探索性 95% CI | 显存 MiB | 训练秒 | 规则判定 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for s,label in zip(summary,labels):
        lines.append(f'| {label} | {s["accuracy_mean_pct"]:.2f} ± {s["accuracy_std_pp"]:.2f} | {s["drop_vs_sft_lora_pp"]:+.2f} | [{s["drop_bootstrap_ci_low_pp"]:+.2f}, {s["drop_bootstrap_ci_high_pp"]:+.2f}] | {s["peak_allocated_mib_mean"]:.1f} | {s["train_seconds_mean"]:.2f} | {"下岗" if s["decision"]=="reject" else "保留"} |')
    lines+=['','完整 15 次运行逐行结果见 [evaluation_metrics.csv](evaluation_metrics.csv)，汇总见 [summary_metrics.csv](summary_metrics.csv)，逐题输出在 `runs/*/test_predictions.json`。','',
            '显存为独立进程内训练阶段的 PyTorch peak allocated，包含基座、梯度和优化器；RLVR 还包含冻结参考模型。训练秒包含 backward、优化器、RLVR rollout 和同步，排除模型加载、参考缓存与验证/测试；setup_seconds 另记。reserved 显存也在 CSV 中，不等于全卡或进程的 nvidia-smi 占用。固定更新次数没有让三种对齐方法的 FLOPs 或 token 数相等。','',
            '![后训练准确率](../curves/alignment_comparison.png)','',
            '![PEFT资源对比](../curves/peft_resources.png)','',
            '![各自训练目标](../curves/training_loss.png)','',
            '三种 loss 的定义不同，不能横向比较其绝对数值；图中每条线来自实际训练日志。RLVR 每批只更新一次，组内 advantage 均值为零，ratio=1 时策略损失的前向值接近零，仍有非零梯度；需要结合 grad_norm、delta_l2 与生成准确率判断变化。','',
            '## 谁掉点谁下岗','',
            f'预先设定掉点门槛为 {config["drop_threshold_pp"]} 个百分点。按三 seed 平均测试准确率，较 SFT-LoRA 下降超过门槛的方案下岗，其余保留。保留表示通过本次任务门槛，不表示统计等价、通用性能更好或其它任务同样适用。CI 为 2000 次成对题目与 seed 重采样，仅作探索性不确定性描述，3 个 seed 及单一共享预热限制了外推。','']
    rejected=[label for s,label in zip(summary,labels) if s['decision']=='reject']
    lines.append(('本次下岗方案为 '+ '、'.join(rejected)+'。') if rejected else '本次没有方案超过预先设定的掉点门槛，因此没有强行淘汰任何方法。')
    lines+=['','## 限制与下一步','',
            '本实验使用穷举生成的窄域数学判断任务，没有人类偏好标注，不能据此宣称完成通用人类价值对齐。共享 SFT 预热使三种方法起点一致，但它们比较的是预热后的继续训练效果，并非各自从原始阶段三 checkpoint 直接训练的差异。预热成本独立记录，不计入单个增量实验。','',
            '下一步应在保持评测集隔离的前提下扩大任务与数据，并为每个随机种子独立预热；增加等 token/计算预算、更多种子和明确置信门槛，验证显存与掉点权衡是否仍成立。','']
    (STAGE/'logs/posttraining_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
