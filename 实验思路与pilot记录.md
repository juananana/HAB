# 代理脑实验思路与 Pilot 实验记录

本文档基于 `代理脑论文1(1).docx` / `paper.md` 中的 Hierarchical Agentic Brain 思路整理实验方案，并记录本次已经完成的初步实验。原始 Word 论文思路未修改。

## 1. 实验部分应该围绕什么问题展开

代理脑论文的核心不是单纯做一个 EEG 分类器，而是要证明“神经信号 -> 神经图 -> 认知单元图 -> 世界模型预测 -> 不确定性决策”这一层级框架是有用的。因此实验建议围绕 5 个问题设计：

1. HAB 是否比传统 EEG 解码和普通神经网络解码更好？
2. 认知单元层是否提供了有用的中间表征，而不只是多加一层网络？
3. 世界模型是否真的能预测下一时刻状态，并改善时序建模或跨 session 泛化？
4. 小世界结构约束是否让认知图更稳定、更可解释，而不明显损害分类性能？
5. 不确定性感知决策是否能在离线模拟中降低错误执行风险？

对应到论文中，实验结果不要只报 accuracy。至少要同时覆盖：分类性能、校准性能、选择性执行风险、世界模型预测误差、认知图结构指标和可解释性可视化。

## 2. 推荐的正式实验路线

### 2.1 主数据集

建议把 BNCI2014_001 / BCI Competition IV 2a 作为正式主数据集。理由是它是运动想象 BCI 常用基准，包含 9 名被试、2 个 session、4 类运动想象任务，适合做 within-subject cross-session、leave-one-subject-out、消融和统计检验。

正式设置建议：

| 模块 | 建议设置 |
| --- | --- |
| 任务 | 四分类运动想象：left hand, right hand, feet, tongue |
| 协议 1 | within-subject cross-session：session 1 训练/验证，session 2 测试 |
| 协议 2 | leave-one-subject-out：留一名被试测试，其余被试训练/验证 |
| 重复运行 | 每个神经网络方法至少 5 个随机种子 |
| 统计报告 | 每被试结果 + 均值/标准差 + 配对检验 |

### 2.2 外部鲁棒性数据集

PhysioNet EEGMMIDB 可以作为外部鲁棒性实验，而不是主结论来源。它被试数多，但任务和 trial 组织方式与 BNCI 不完全一致，适合作为“换采集协议后趋势是否还存在”的验证。

SEED 情绪数据集可以作为可选扩展。只有在运动想象主实验闭环完成后，再考虑用它验证“认知单元框架能不能跨任务迁移到情绪建模”。不要在没有实际实验前声称多模态或跨任务优势。

## 3. 方法与对照组设计

正式论文建议至少包含以下对照：

| 类型 | 方法 | 作用 |
| --- | --- | --- |
| 传统 EEG | CSP + LDA | 运动想象经典 baseline |
| 传统 EEG | FBCSP + LDA | 更强的多频带 baseline |
| 神经网络 | EEGNet | 轻量级 EEG 深度学习 baseline |
| 神经网络 | ShallowConvNet | 频带功率导向 baseline |
| 神经网络 | DeepConvNet | 更深的端到端 baseline |
| 图模型 | Dynamic Neural Graph Only | 验证“神经图”本身的贡献 |
| 本文方法 | HAB | 完整代理脑模型 |

HAB 消融建议包含：

| 消融版本 | 目的 |
| --- | --- |
| w/o Neural Graph | 验证动态图编码是否有用 |
| w/o Cognitive Units | 验证认知单元抽象是否有用 |
| w/o Small-World Constraint | 验证小世界约束是否改善结构与稳定性 |
| w/o World Model | 验证下一状态预测是否有用 |
| w/o Calibration | 验证不确定性校准是否有用 |
| w/o Risk Policy | 验证选择性执行是否降低风险 |

## 4. 指标设计

主分类指标：

| 指标 | 含义 |
| --- | --- |
| Accuracy | 总体分类准确率 |
| Macro-F1 | 类别均衡性能，防止只看多数类 |
| Cohen's kappa | BCI 论文常用一致性指标 |

不确定性与风险指标：

| 指标 | 含义 |
| --- | --- |
| NLL | 概率预测质量 |
| ECE | 校准误差 |
| Brier score | 概率预测平方误差 |
| Coverage | 模型选择执行的比例 |
| Selective risk | 只执行样本上的错误率 |
| Expected cost | 错误执行成本 + 请求确认成本 |

世界模型指标：

| 指标 | 含义 |
| --- | --- |
| Next-state MSE | 预测下一认知状态/神经图表征误差 |
| Reconstruction error | 用认知状态重构下一窗口信号的误差 |
| Next-step Acc. | 用预测的下一状态做下一步分类的准确率 |

结构与可解释性指标：

| 指标 | 含义 |
| --- | --- |
| Clustering coefficient | 认知图局部聚合程度 |
| Average path length | 全局信息整合效率 |
| Modularity | 模块结构 |
| Sparsity | 是否退化为全连接 |
| Small-worldness | 相对随机图的小世界程度 |
| Unit activation heatmap | 认知单元随时间/类别的激活差异 |

注意：如果没有注意、记忆、情绪等真实认知标签，论文中不要说某个 learned unit “就是注意/记忆/情绪”。更稳妥的说法是：认知单元是任务相关、结构化、可检查的潜在中间状态。

## 5. 本次已经完成的实验

本次完成了一个小规模 PhysioNet EEGBCI pilot 和一个 HAB 组件消融 pilot。

### 5.1 已有主 pilot

路径：`results/pilot_physionet/summary.md`

设置：PhysioNet EEGBCI，被试 1-5，runs 4/8/12，左/右手二分类，leave-subject-5-out。

核心结果：

| 方法 | Acc | Macro-F1 | Kappa | NLL | ECE | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| Bandpower-LDA | 0.533 | 0.348 | 0.000 | 7.440 | 0.467 | 0.933 |
| Raw-LogReg | 0.444 | 0.443 | -0.100 | 1.270 | 0.367 | 0.761 |
| EEGNet-Lite | 0.467 | 0.318 | 0.000 | 0.696 | 0.049 | 0.503 |
| HAB-Prototype | 0.533 | 0.348 | 0.000 | 0.691 | 0.016 | 0.498 |

可以写成：HAB 原型已经完成端到端训练与评估流程，在小规模 pilot 中达到与 Bandpower-LDA 相同的 accuracy，并在 NLL/ECE/Brier 上优于当前几个 baseline。

不能写成：HAB 已经显著优于所有 baseline。原因是当前只用了 5 个被试、二分类任务、单一测试被试，且 accuracy 与传统 Bandpower-LDA 持平。

### 5.2 本次新增消融 pilot

新增脚本：`code/run_pilot_ablations.py`

新增结果路径：`results/pilot_physionet_ablation/summary.md`

比较了 Full HAB、w/o World Model、w/o Small-World Loss、w/o Neural-Cognitive Align、Dense Neural Graph。结果显示所有变体的 Acc 都是 0.533，Macro-F1 都是 0.348，差异主要只体现在非常小的 NLL/ECE/Brier 变化。

关键观察：

| 观察 | 解释 |
| --- | --- |
| Full HAB 与消融变体几乎没有分类差异 | 当前 pilot 太小，模型也较轻，无法证明组件贡献 |
| w/o World Model 的 next-state MSE 明显变大 | 世界模型损失在 latent 预测上有作用 |
| last-state copy MSE 接近 0 | 认知 latent 可能过于平滑，世界模型评估需要更强 proxy |
| cognitive-graph 指标几乎完全一致 | 当前认知邻接矩阵接近均匀分布，小世界解释性还没有建立 |
| 选择性执行没有稳定降低风险 | 当前不确定性排序还不够可靠，需要温度校准或更强不确定性建模 |

因此，本次消融 pilot 的意义是“发现问题”，不是“证明方法优越”。这反而很有价值：正式实验前需要加强认知图结构学习、世界模型 proxy 和校准模块。

## 6. 下一步应该怎么做

优先级 1：把 PhysioNet pilot 扩成可靠版。

| 任务 | 目标 |
| --- | --- |
| 从 5 被试扩展到更多被试 | 降低偶然性 |
| 做 leave-one-subject-out 全被试循环 | 得到每被试均值/方差 |
| 每个方法跑 5 个 seed | 支持统计检验 |
| 加 CSP/FBCSP baseline | 避免 baseline 偏弱 |
| 加真正的 w/o Cognitive Units 结构消融 | 对应论文核心 claim |

优先级 2：实现正式 BNCI2014_001 实验。

| 任务 | 目标 |
| --- | --- |
| 数据加载与预处理固定 | 建立主实验可复现协议 |
| within-subject cross-session | 验证 session 泛化 |
| leave-one-subject-out | 验证跨被试泛化 |
| 画 accuracy/F1、calibration、coverage-risk 曲线 | 对应论文图表 |

优先级 3：增强 HAB 结构约束与可解释性。

| 任务 | 目标 |
| --- | --- |
| 对 cognitive adjacency 加稀疏化或 top-k | 避免均匀邻接 |
| 计算真实图指标和随机图参照 | 支持 small-worldness |
| 输出认知单元激活热图 | 支持解释性章节 |
| 增加温度缩放/MC dropout/ensemble | 改善不确定性排序 |

## 7. 当前可复现实验命令

运行主 pilot：

```bash
python3 code/run_pilot_experiment.py
```

运行新增消融 pilot：

```bash
python3 code/run_pilot_ablations.py
```

本次为了运行实验，已安装 Python 包 `mne`。另外，脚本已改为不向用户主目录写 MNE 配置，而是使用项目内数据目录和缓存目录。

## 8. 服务器全量实验运行记录

已在服务器上创建独立实验目录，避免与已有的边界水位/TimeMixer 相关项目混在一起。

| 项目 | 内容 |
| --- | --- |
| 服务器实验目录 | `/root/autodl-tmp/hab_agentic_brain_physionet_20260706` |
| 避开的已有目录 | `/root/timemixer_project`、`/root/timemixer_v2` |
| 全量配置 | `code/config_full_physionet.json` |
| 全量脚本 | `code/run_full_physionet_experiment.py` |
| 正式结果目录 | `results/full_physionet_loso` |
| 日志目录 | `logs/` |
| 后台 PID 文件 | `logs/full_physionet_loso.pid` |
| 进度监控日志 | `logs/full_progress_monitor.tsv` |

服务器 smoke test 已完成：使用 3 个被试、1 个 seed、4 个方法验证了数据加载、Bandpower-LDA、CSP-LDA、EEGNet-Lite、HAB-Full、风险指标和 summary 输出流程。该结果只用于验证脚本，不用于论文结论。

正式全量任务已后台启动，协议为 PhysioNet EEGBCI 109 被试 leave-one-subject-out，3 个随机 seed，主比较方法为 Bandpower-LDA、CSP-LDA、EEGNet-Lite、HAB-Full。由于服务器直接通过 MNE 单线程下载较慢，正式任务改为使用并行度 4 的 `wget` 预下载包装脚本 `scripts/predownload_then_run_full.sh`：先下载并缓存服务器缺失的 EDF 文件，再自动运行全量实验。该调整只改变数据下载方式，不改变数据集、split、预处理、模型或指标设置。实验开始后会逐 fold 写入 `metrics.csv`、`risk_metrics.csv`、`world_model_metrics.csv`、`graph_metrics.csv` 和 `summary.md`。

服务器监控命令：

```bash
cd /root/autodl-tmp/hab_agentic_brain_physionet_20260706
ps -p $(cat logs/full_physionet_loso.pid) -o pid,etime,cmd
tail -n 80 $(ls -t logs/predownload_then_full_*.log logs/full_physionet_loso_*.log 2>/dev/null | head -1)
find data/mne/MNE-eegbci-data/files/eegmmidb/1.0.0 -name '*.edf' | wc -l
tail -n 20 logs/full_progress_monitor.tsv
sed -n '1,220p' results/full_physionet_loso/summary.md
```

如果主比较完成后要继续全量消融，可将 `code/config_full_physionet.json` 中的 `run_ablations` 改为 `true`，并用新的结果目录运行，避免覆盖主比较结果。

### 2026-07-07 新服务器重启记录

原服务器资源不可用后，镜像迁移到新服务器。检查发现新服务器 `/root/autodl-tmp` 数据盘为空，说明旧服务器数据盘中的项目目录和 full 运行中间结果未随镜像保留。因此，旧服务器上已完成的大部分 fold 不能直接 resume。

为避免混淆结果来源，已在新服务器重新创建同名独立目录，并上传本地实验代码后从头启动 full runner：

```bash
cd /root/autodl-tmp/hab_agentic_brain_physionet_20260706
nohup python3 code/run_full_physionet_experiment.py \
  --config code/config_full_physionet.json \
  --result-dir results/full_physionet_loso \
  > logs/full_physionet_loso_restart_20260707_095029.log 2>&1 &
echo $! > logs/full_physionet_loso.pid
```

本次后台 PID 为 `1714`。日志显示已经开始从 PhysioNet 重新下载 EDF 文件。该重启不改变 full 实验配置，只是由于数据盘为空，需要重新下载数据并重新生成完整结果。

新服务器监控命令：

```bash
cd /root/autodl-tmp/hab_agentic_brain_physionet_20260706
ps -p $(cat logs/full_physionet_loso.pid) -o pid,etime,cmd
tail -n 80 logs/full_physionet_loso_restart_20260707_095029.log
find data/mne/MNE-eegbci-data/files/eegmmidb/1.0.0 -name '*.edf' | wc -l
sed -n '1,220p' results/full_physionet_loso/summary.md
```

### 2026-07-07 克隆昨晚实例后的有效续跑

重新克隆昨晚实例后，当前有效服务器为：

```bash
ssh -p 31062 root@connect.westb.seetacloud.com
```

该服务器保留了旧数据盘内容，`/root/autodl-tmp/hab_agentic_brain_physionet_20260706` 中存在 full 结果、缓存和 EDF 数据。续跑前检查结果：

| 项目 | 数值 |
| --- | --- |
| `metrics.csv` 行数 | `1355 / 1962` |
| 成功行数 | `1355` |
| 失败行数 | `0` |
| 完成进度 | `69.06%` |
| 当前 seed | `3407` |
| full 缓存加载 | `epochs=4743 subjects=109` |

已用不带 `--force` 的命令继续运行：

```bash
cd /root/autodl-tmp/hab_agentic_brain_physionet_20260706
nohup python3 code/run_full_physionet_experiment.py \
  --config code/config_full_physionet.json \
  --result-dir results/full_physionet_loso \
  > logs/full_physionet_loso_resume_20260707_095556.log 2>&1 &
echo $! > logs/full_physionet_loso.pid
```

续跑 PID 为 `1611`。启动后已确认 `metrics.csv` 增长到 `1366` 行，说明断点续跑生效。后续监控应以该服务器和该日志为准。

## 9. 全量前小规模验证记录

在正式全量实验继续运行前，已暂停服务器上的预下载流程，并使用已缓存的 S001-S005 数据完成 5 被试 leave-one-subject-out 验证。验证目的不是报告论文结论，而是检查实验代码、split、指标输出和方法设计是否存在明显问题。

验证设置：

| 项目 | 设置 |
| --- | --- |
| 数据 | PhysioNet EEGBCI S001-S005 |
| 协议 | 5-fold leave-one-subject-out |
| 随机种子 | 42 |
| 方法 | Bandpower-LDA, CSP-LDA, EEGNet-Lite, HAB-Full, HAB-NoWorldModel, HAB-NoSmallWorld, HAB-NoAlign, HAB-DenseGraph |
| 训练轮数 | max_epochs=8, patience=3，仅用于流程验证 |
| 结果目录 | `results/validation_physionet_loso5_topk` |

验证结论：

| 检查项 | 结果 |
| --- | --- |
| 运行完整性 | 40 行 method/fold 结果全部成功，无 failed 行 |
| split 泄漏 | 未发现 test subject 出现在 validation subjects 中 |
| 指标输出 | `metrics.csv`、`risk_metrics.csv`、`world_model_metrics.csv`、`graph_metrics.csv`、`summary.md` 均正常生成 |
| world-model loss | 已修复为只使用同一被试且原始索引连续的样本对，不再使用打乱 batch 内的假时序 |
| 认知图结构 | 已加入 `cognitive_top_k=3`，Full HAB 的 cognitive graph 不再退化为均匀全连接；DenseGraph 作为无稀疏认知图的对照保留 |
| 初步性能观察 | 5 被试验证中 HAB-Full 不稳定，HAB-NoWorldModel 和 HAB-DenseGraph 有必要纳入正式全量比较 |

验证暴露的论文写作边界：

1. 不能预设 HAB-Full 一定优于所有 baseline，必须以全量结果为准。
2. 不确定性风险分析必须如实报告；小规模验证中 HAB-Full 的 ECE 和 selective risk 并不稳定。
3. 小世界/认知图解释必须依赖修复后的稀疏认知图指标，不能使用均匀全连接图来支撑解释性结论。
4. 正式全量配置已改为保留关键消融：`HAB-NoWorldModel` 和 `HAB-DenseGraph`，避免只报告 Full HAB。
