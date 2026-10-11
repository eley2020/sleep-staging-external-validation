# Sleep Medicine 完整方法学整理

## 1. 研究设计与研究问题

本研究为回顾性、多队列、subject-disjoint 外部验证研究，核心问题是：仅在 Sleep-EDF
训练的多通道睡眠分期模型，在不同 PSG 队列中发生多大性能下降，以及需要多少目标队列
标注受试者才能恢复主要分期性能。主要终点为目标测试受试者上的 Macro-F1；Accuracy、
weighted-F1、Cohen's kappa、分期特异性指标、校准和睡眠结构误差为次要终点。主模型是
`four-channel quality-gated AttnSleep extension`，不是全新网络架构。

研究分析分为三层：

1. Sleep-EDF-20 的 source-domain subject-wise 训练和质量门控对照；
2. SHHS1、SHHS2 follow-up、Sleep-EDF ST 和 ISRUC Cohort I 的 frozen source-only 外部验证；
3. SHHS1 和 ISRUC 的 subject-disjoint few-shot/finetuning 预算分析。

各项主分析记录其数据划分、checkpoint规则、指标定义和信息使用范围。事后敏感性实验
在本轮评分前锁定配置，但此前已查看外部标签和结果，不称全新盲测或正式预注册。

## 2. 数据集、队列和受试者单位

Sleep-EDF 作为 source domain。主实验使用 Sleep-EDF-20 的四通道 PSG：EEG Fpz-Cz、EEG
Pz-Oz、水平 EOG 和下颏 EMG。外部验证包括 SHHS1、SHHS2 longitudinal follow-up、
Sleep-EDF Sleep-Telemetry (ST) 和 ISRUC Cohort I。SHHS2 与 SHHS1 存在纵向关联，因此在
正文中称为 follow-up validation，不称为完全独立人口；ISRUC 作为独立队列报告，但其
montage/channel mapping 差异较大。当前研究不包含 MASS 或 MESA 原始 PSG，不能声称这些
队列上的泛化。

原始 PSG、标注和数据许可由各数据平台管理，不纳入代码仓库。数据访问和引用要求见
`docs/data_access_and_licenses.md`。纳入条件是存在可读取的 PSG、对应睡眠分期标注和
可映射的目标通道；缺少必要通道、无法建立 epoch 对齐或全部标签未知的记录排除。每个
受试者是 bootstrap 和数据划分的基本单位；同一受试者的多个记录不能跨训练、校准和测试
集合。

锁定的 frozen source-only 规模如下：SHHS1 5,601 subjects/records、5,164,326 epochs；
SHHS2 2,650 subjects/records、2,691,739 epochs；Sleep-EDF ST 22 subjects、44 records、
42,471 epochs；ISRUC I 100 subjects/records、90,187 epochs。上述规模是清洗和有效标签
排除后的分析规模。

## 3. 信号预处理和标签定义

每个睡眠 epoch 长度为 30 s，目标采样率为 100 Hz，因此每个输入 epoch 有 3,000 个样本。
信号按照物理单位读取并执行 deterministic normalization；Sleep-EDF 使用
`physical-volts` normalization。EMG 原始采样率为 1 Hz 时，采用确定性的 repetition
上采样到 3,000 点，而不是随机插值。记录边缘窗口为 `w_edge_mins=30`。标签按 30 s
epoch 对齐，未知或无法对齐的标签排除。五分类标签为 Wake (W)、N1、N2、N3 和 REM。

目标队列的通道映射在预处理阶段固定，不根据测试标签修改。ISRUC 保留原始 montage 的
映射敏感性，并把 montage 异质性作为外部有效性限制。所有标准化、裁剪和通道质量统计
均只能使用当前训练/校准流程允许的信息。

## 4. 模型架构和可追溯性

主干继承 Eldele 等发表于 IEEE TNSRE (2021) 的 AttnSleep：MRCNN、多尺度卷积特征、
adaptive feature recalibration/SE、causal-convolution-enhanced temporal context encoder
以及五分类头。多尺度卷积、残差、SE、Transformer attention、LayerNorm、causal convolution、
GELU、BatchNorm、pooling 和 dropout 均为已发表的基础组件。

本项目新增或改变的部分只有：

- 将 MRCNN 输入由单 EEG 改为四通道 PSG；
- 增加轻量级 per-epoch channel-quality gate，输入为幅值和导数统计（RMS、差分 RMS、
  平均绝对幅值等），用于通道特征重加权；
- 使用 channel dropout=0.15 进行训练期通道扰动；
- 在外部验证中采用明确分离的冻结推理、目标域适配和序列后处理协议。

质量门控是输入融合模块，不改变 AttnSleep 的 MRCNN/AFR/TCE 归属。匹配的四通道
early-fusion 对照使用相同训练划分、优化器和预算，仅移除质量门控，以隔离“多模态输入”
与“质量门控”的贡献。质量门控的清洁 Sleep-EDF 增益很小且不显著，因此正文将其定位为
选择性的通道退化鲁棒性机制，而不是普遍的准确率提升。

## 5. Source-domain 训练和 checkpoint 规则

Sleep-EDF-20 使用 subject-wise 外层划分。每一折的测试受试者在训练和模型选择期间完全
不可见；通常从训练受试者中固定留出 2 名内部 validation subjects。训练结束后仅依据
内部 validation 规则确定 checkpoint，再读取测试受试者一次完成最终评分。优化器、学习率、
batch size、训练轮数、随机种子和早停规则在运行配置中固定并随结果归档。主线 source-domain
结果为 Accuracy 0.8419、Macro-F1 0.7885、Cohen's kappa 0.7811；这是 Sleep-EDF-20
subject-wise 内部估计，不是 external validation accuracy。

与 early fusion 的完全匹配 20-fold 对照为 Accuracy 0.8414 vs 0.8419、Macro-F1 0.7856
vs 0.7885、kappa 0.7800 vs 0.7811；fold-paired p 值分别为 0.812、0.571 和 0.674。
这组结果不支持将质量门控写成 clean-set 显著优于 early fusion。

## 6. Frozen zero-shot 外部验证

外部验证使用 source checkpoint 的 frozen source-only 推理，不使用目标标签选择 checkpoint、
温度、阈值、解码器、归一化参数或任何后处理。目标信号仅经过预先固定的通道映射和输入
预处理。模型输出按 epoch 聚合为五分类预测；记录边界重置序列状态。

冻结外部结果必须与 source-domain 结果分开报告。锁定的 pooled Accuracy/Macro-F1/kappa
为修正输入后的 Sleep-EDF ST 0.7441/0.6905/0.6389、SHHS1 0.6185/0.5176/0.4563、SHHS2
0.5314/0.4443/0.3664 和 ISRUC I 0.5459/0.5154/0.4160。该 source-to-external
degradation 是研究终点之一，不应被改写为模型的普遍外部鲁棒性。

点估计按所有有效 epoch 的 pooled confusion matrix 计算，但不把 epoch 当作独立重复样本。
95% CI 使用 5,000 次 subject-level bootstrap：每次重采样受试者，并将受试者的全部记录
和 epoch 一起纳入；每个受试者贡献一个 5x5 confusion matrix。这样既保持 pooled point
estimate，又避免 epoch-level independence 假设。

投稿前完整性审计发现 ST 的 7/44 份缓存在 annotation EDF 打开失败回退时被静默写成
四通道全零（5/22 受试者，6,106/42,471 epochs）。修复将 PSG 信号读取和关闭与
annotation 读取分离，并拒绝非有限或整通道全零信号。全部 44 份写入新目录，不删除问题
受试者；标签、shape、物理单位、采样率、通道、裁剪均不变。重建数组与独立原始 EDF
重建逐元素一致，37 份原本有效的信号与旧缓存完全一致；使用原 20 个冻结 checkpoint
在两块 TITAN V 上重新评分，不重训、不用目标标签调参。

原 ST 推理使用 CPU，新推理固定 CUDA FP32。原协议和首次停止日志保留，另行锁定
CPU/CUDA 数值一致性修订：全部旧输入 logits 满足 atol=rtol=1e-5；仅 ST7181J0 的
epoch 948 在 W/N1 近乎平局时翻转（原 logits 差 4.768e-7）。另在相同 CUDA 条件下
评分旧输入，配对 5,000 次受试者 bootstrap，隔离纯输入修复效应；该差值不是方法收益。
ST 两夜始终按受试者一起重采样。修正主表、数值修订及审计输出均在
`outputs/st_baseline_repair_20261011/`。旧 ST 适配、校准和上下文支线暂不纳入稿件；
若恢复，须在正确输入上按对应协议重新验证，不能由本次基线评分代替。

## 7. Few-shot 目标域适配

SHHS1 使用三个固定 subject-disjoint splits。每个 split 具有 calibration、validation 和
held-out test subjects；目标标签只用于 calibration/validation 允许的适配和 checkpoint
选择，held-out subjects 在适配完成后一次性评分。预算为 1、5、10、25 和 50 个 calibration
subjects，所有预算共享预先固定的 split 和 frozen 对照。主要报告 held-out Macro-F1 及其
跨 split 均值和标准差；不把三个 split 当成大量独立受试者进行显著性扩展。

ISRUC 使用原始 montage 的 subject-disjoint fine-tuning，并保留 1、5、10-subject budget
敏感性分析。10-subject 结果的平均 Macro-F1 为 0.5832，sample SD 0.0133（旧汇总population SD 0.0109），但每次选择仅有 7 名
held-out subjects，因此仅作为敏感性证据，不作为确定性临床性能估计。

## 8. 睡眠结构端点

记录级端点从参考标注和模型 hypnogram 分别计算：total sleep time (TST, h)、sleep
efficiency (%)、wake after sleep onset (WASO, min)、REM latency (min) 以及各阶段时长。
报告 model-vs-reference 的 bias、MAE、median absolute error 和 95% limits of agreement，
并以受试者为 bootstrap 单位估计不确定性。这些是与可用参考标注的误差，不是第二人工评分者
之间的 inter-rater agreement。

## 9. 序列连续性后处理

Viterbi 解码作为预先指定的 secondary post-processing protocol，而非神经网络架构改进。
它接收 epoch-level soft log probabilities，transition 参数只能由 source training subjects
或指定 target calibration subjects 估计，并在内部 validation subjects 上选择。held-out
target labels 不得选择 transition penalty、最短持续时间或 smoothing strength。序列在每条
记录边界重置。正文同时报告 raw 和 decoded 结果、每阶段 F1/recall、transition count、
平均 run length 和睡眠结构误差；不把 decoded 增益归因于网络参数变化。

## 10. 统计分析

主要终点是subject-disjoint held-out集合的pooled五阶段Macro-F1，受试者为重采样单位，
并非简单平均每人Macro-F1。source-domain数值为20折受试者指标均值，与外部pooled估计
的权重不同。Accuracy、weighted-F1 和 kappa 为总体补充
指标；W/N1/N2/N3/REM 分别报告 precision、recall、F1、support。source-domain ablation
使用同一外层 fold 的 paired differences；SHHS1 budget 使用预先固定 split 的 paired
frozen-versus-adapted differences。所有置信区间优先以受试者为重采样单位；只有在明确说明
时才使用 fold 或 record 作为重采样单位。多重比较使用 Holm 校正时，结果解释为探索性而
不是大规模确认性检验。所有点估计保留至少四位小数，正文可按期刊格式四舍五入。

## 11. 信息使用、泄漏控制和可复现性

每项结果必须标注 frozen、unlabeled-target 或 few-shot。主测试标签不用于checkpoint、
超参数、阈值、校准、通道映射或解码器选择；历史上已进行多轮外部诊断和复核，不能
将所有评分描述为只查看一次。任何未来利用未标注目标
信号统计的实验都应明确称为 transductive，并列出所用统计量。

原统计由 `scripts/pre_submission_statistics.py` 生成，但该历史入口仍指向旧ST预测，
不作为当前投稿再生成入口。当前有效统一统计位于
`outputs/st_baseline_repair_20261011/pre_submission_statistics_corrected.json`，并已完成独立
审计。公开包提供基于审定aggregate数据的统计表/图复现入口；原始级复现另需官方数据、
源checkpoint及相应锁定清单。核心JSON、CSV、脚本路径和种子保留，旧协议不静默改写。

## 12. 固定预处理敏感性（Supplement S8）

原20-fold source ensemble与100Hz/30秒/四通道合同保持不变。四模式为original、仅抗
混叠、仅EMG包络及两者联合。抗混叠仅用于native>100Hz的波形，完整时间轴polyphase
resample_poly/Kaiser5；≤100Hz保留原重复/插值。EMG固定为完整native记录四阶Butterworth
HP16Hz、绝对值整流、四阶LP0.4Hz、polyphase到1Hz并重复100次，不做幅值拟合；
这是近似表示协调而非精确恢复源域设备处理。sosfiltfilt为离线零相位，不称实时因果。

ST全44夜/22人、ISRUC全100人；SHHS每波按此前原始文件清单等距预选32条，不据标签
选样。SHHS1两条缺EEG(sec)不补抽，故有效30条；SHHS2有效32条。两块TITAN V上
FP32/batch64、mean logits/noTTA评分，全部原模式历史类别预测一致；ST两组identity负对照
逐logit一致。5000次paired subject bootstrap同时纳入ST两夜，不将epoch独立抽样。

提前固定的资源规则为至少两个非随访队列Macro-F1增量≥0.005且含ST或ISRUC；各队列
Macro-F1/Accuracy≥−0.005、N1 recall≥−0.02、N1/N3 F1≥−0.005。该规则不是非劣效
检验或预注册确认性假设。全部修改模式报告，没有通过者，不改变主稿冻结模型。
没有bootstrap置信区间的多重检验校正，因此描述为事后探索性敏感性。仅SHHS抽样，
不能代表修改后全库性能；原20折来自一个source训练种子，不是多独立训练种子。

原始数据不随仓库分发；复现者须按各数据集官方流程获得数据，并核对当前许可、引用和衍生
预测共享要求。
