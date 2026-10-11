# Sleep Medicine 投稿限制与报告边界

## 外部有效性

研究包含 SHHS1、SHHS2 follow-up、Sleep-EDF ST 和 ISRUC I，但没有 MASS 或 MESA 独立队列。
SHHS2 与 SHHS1 存在纵向来源关系，不能当作完全独立人口。ISRUC 是独立队列，但原始
montage、通道数量和映射差异明显，整体结果不能代表所有临床 PSG 配置。

源域到外部队列的下降是本研究的主要发现，也是重要限制：Sleep-EDF-20 内部
Accuracy/Macro-F1/kappa 为 0.8419/0.7885/0.7811，但 frozen source-only 结果分别降至
修正输入后的 Sleep-EDF ST 的 0.7441/0.6905/0.6389、SHHS1 的 0.6185/0.5176/0.4563、SHHS2 的
0.5314/0.4443/0.3664 和 ISRUC I 的 0.5459/0.5154/0.4160。尤其 Sleep-EDF ST 也出现
明显下降，说明问题不只是完全不同数据库造成的。因此 0.8419 不能写成 external validation
accuracy，也不能据此声称强零样本泛化。

## 输入完整性与修正结果

投稿前审计发现 ST 的 7/44 份旧缓存四通道全零，源自当前环境中 EDF annotation
失败回退与仍打开的 PSG 句柄之间的相互影响，并非真实生理信号缺失。已保留旧文件，
在新目录重建全部 44 份并使用原冻结模型重算；上述数值是修正后的结果。此纠错不可
写成模型创新或预处理方法收益，也不能删除问题记录后报更好成绩。37 份不受影响的波形
逐元素相同；CPU/CUDA FP32 近乎平局导致 1 个 epoch 的预测差异已独立记录并提供
同 GPU 新旧输入对照。旧 ST 适配和上下文结果尚未重新验证，本稿不使用这些结果。

完整性核对支持此次特定错误已修复，不证明所有数据都没有其他采集或标注问题。
外部标签此前已被查看，修正复核不能称为全新的盲测。ST 临床端点仅针对保留的 benchmark
窗口：睡眠效率分母为有效保留 epoch 而非真实 time-in-bed，未知时间间隙不补标注。
修正后 ST Macro-F1 仍比源域低约 0.0980，N1 precision 仅 0.3555，外部有效性限制仍存在。

## 参考标准和临床解释

当前只有数据集参考标注，没有第二人工评分者或 adjudicated human subset。因此所有 accuracy、
F1、kappa 和 sleep-structure MAE 都是 model-vs-reference annotation comparison，不能写成
human-vs-human agreement、expert equivalence 或 clinical readiness。适配后 TST MAE 约 0.539 h、
REM latency MAE 约 60 min、N1 duration MAE 约 63 min，仍足以影响睡眠表型解释。

## 少数阶段和适配稳定性

N1 sensitivity 在 SHHS1 适配后提高，但 precision 仍约 0.1496，提示 N1 过预测风险。ISRUC
10-subject 结果平均 Macro-F1 0.5832，sample SD 0.0133（旧population SD 0.0109），但每个选择只有 7 个 held-out subjects；
因此只能作为敏感性证据。三个 SHHS1 split 的 split-level p 值也不具备大量独立受试者的统计
功效。

## 模型和协议边界

质量门控在清洁 Sleep-EDF 上相对 early fusion 的增益不显著，支持的是特定通道退化下的选择性
鲁棒性。Viterbi 是 validation-controlled post-processing，不是新的网络架构；过强连续性约束
可能抹平真实短暂 N1/REM 事件。SHHS 监督预训练迁移到 Sleep-EDF 的完整 20-fold 结果为负，
不能把外部预训练写成必然提升。

适配后的结果也不能解释为 source model 原生外部泛化能力；它们是在 subject-disjoint
目标校准受试者上使用标签后的 domain repair 结果。目标标注预算、队列差异和通道映射
都会影响收益，当前证据支持的是“可量化的少样本修复”，不是跨医院直接部署保证。

## 预处理敏感性不支持通用改进

固定2×2探查中，EMG包络改善ST Macro-F1/N1，但降低ISRUC Macro-F1/N3；SHHS抽样
总体增益区间包含零，联合处理在SHHS2牺牲N1 F1。所有修改模式未通过预设资源/安全规则。
因此正文保留修正后的原始输入冻结基线，包络结果仅补充报告，不将其作为更换主线的依据。
SHHS每波仅32条预设记录，外部标签早已查看，多个终点为探索性；不能声称新盲测、
通用协调、精确复原源EMG、或证明非劣。包络同时改变信号幅值/频谱/门控统计，效应
不能单独归因于门控或某个神经网络组件。零相位全记录处理不适合直接称实时部署。

## 尚未覆盖的部署因素

没有 prospective deployment、实时延迟评估、不同评分者一致性、设备/医院级外部验证或治疗
决策影响分析。原始数据和许可受数据平台限制，不能在仓库中直接提供。结果也不支持替代人工
评分、临床决策或优于同协议 in-domain SOTA 的声明。

## 推荐的诚实结论

最稳妥的结论是：subject-disjoint target-domain annotation 可一致降低跨队列睡眠分期误差，
但收益依赖队列和通道配置；残余 N1、睡眠结构和独立临床验证缺口仍限制临床转化。后续最有
价值的工作是锁定一个非 SHHS 派生独立队列，并加入第二评分者或 adjudicated reference。
