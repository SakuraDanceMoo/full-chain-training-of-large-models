# 图表证据与导出检查

2026-10-19，Python/matplotlib。用于学习实验仓库，未指定期刊投稿；最终尺寸 9×4.8 inch，PNG 300 dpi，PDF/SVG 可编辑文字。不需要 TIFF；源审查的 TIFF 缺项和非 600 dpi 两项 WARN 在此记录，没有 FAIL。

证据映射：attention 两图展示本机给定形状的 forward 时延，来自完整 25 样本 CSV；N/D 两图展示固定协议的局部幂律拟合；holdout 图检验未参与拟合的整档大小；joint 图的虚线仅是拟合形式延伸，不是已验证的大模型性能。

6 张图均为单面板，跨面板对齐不适用，保存 `curves/*.alignment.json`。最终 PDF 逐张运行文本可读性与碰撞审查，均退出 0；源代码审查 19 PASS、2 WARN、0 FAIL。机器可读记录保存在本目录。

逐图视觉复核坐标、对数轴、legend、误差条、标题及脚注。Legend 放在 axes 外避免网格线/数据相交，底部预留脚注空间。时延误差条为 P10–P90；scaling 为 3-seed 样本 SD，均非泛化误差界。部分重合曲线和很小 SD 在当前尺度下难以分开，原 CSV 保留全部值。

没有裁剪不利数据点；SDPA FLASH 和官方参考 N=64 缺测未填零或补值。远距离外推标为未验证，E 触及下界保留解释。Profiler 另为真实浏览器截图，未标为 Nsight。
