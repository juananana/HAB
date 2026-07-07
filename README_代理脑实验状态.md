# 代理脑实验状态 README

本文档用于说明 `代理脑论文1(1).docx` 中实验设计的整体要求、我们当前实际完成了什么、完整实验思路是什么、结果对论文能支撑到什么程度，以及后续还需要补哪些实验。

注意：`代理脑论文1(1).docx` 是老师给出的论文思路文件，本文档只做实验台账与执行说明，不修改原 Word 文件。

## 1. 老师文档中的实验设计主线

`代理脑论文1(1).docx` 的实验部分不是只要求做一个 EEG 分类准确率表，而是围绕 Hierarchical Agentic Brain, HAB, 这个框架验证以下问题：

| 编号 | 论文实验问题 | 需要的证据 |
| --- | --- | --- |
| RQ1 | HAB 是否提升基础脑信号解码性能 | 与 CSP-LDA、FBCSP-LDA、EEGNet、ShallowConvNet、DeepConvNet、图模型等 baseline 比较 Acc、Macro-F1、Kappa |
| RQ2 | 神经到认知单元的抽象是否有用 | 与去掉 cognitive units 或直接 signal-to-label 的模型比较 |
| RQ3 | 认知世界模型是否改善状态预测和泛化 | next-state prediction、w/o World Model 消融、跨会话或跨被试泛化 |
| RQ4 | 小世界结构约束是否改善图结构稳定性和可解释性 | 稀疏图/稠密图对照、图结构指标、small-worldness 或相关 proxy |
| RQ5 | 不确定性感知决策是否降低高风险错误执行 | calibration、coverage-risk 曲线、selective risk、expected cost |

原始实验设计把数据集分成三层：

| 数据集 | 在论文中的角色 | 当前状态 |
| --- | --- | --- |
| BNCI2014_001 / BCI Competition IV 2a | 主数据集；用于被试内跨会话、跨受试者泛化、消融和可解释性分析 | 尚未完成，是后续最重要的主实验 |
| PhysioNet EEGMMIDB | 外部鲁棒性验证；用于检验不同采集系统、通道数量和大被试规模下的稳定性 | 已完成 full LOSO 外部验证 |
| SEED | 可选跨任务扩展；情绪三分类 | 尚未做，不应作为当前论文核心结论 |

## 2. 我们当前实际完成了什么

目前完成的是 PhysioNet EEGMMIDB 这条外部鲁棒性验证线，而不是 BNCI2014_001 主实验线。

### 2.1 已完成的小规模验证

目的：在全量实验前检查代码、数据划分、指标输出和 HAB 设计是否有明显问题，避免浪费服务器时间。

| 项目 | 内容 |
| --- | --- |
| 数据 | PhysioNet EEGBCI S001-S005 |
| 协议 | 5 被试 leave-one-subject-out |
| 方法 | Bandpower-LDA, CSP-LDA, EEGNet-Lite, HAB-Full, HAB-NoWorldModel, HAB-NoSmallWorld, HAB-NoAlign, HAB-DenseGraph |
| 作用 | 验证流程，不作为论文正式结论 |
| 发现 | HAB-Full 小样本不稳定，world-model 与图结构消融有必要保留 |

小规模验证后做过的重要修正：

| 修正 | 原因 |
| --- | --- |
| world-model loss 改为只使用同一被试且原始索引连续的样本对 | 避免用 shuffle batch 伪造时序关系 |
| cognitive graph 加入 `cognitive_top_k=3` | 避免认知图退化为均匀全连接 |
| HAB 辅助损失加入 warm-up/ramp | 避免分类目标和世界模型/结构目标早期互相拉扯 |
| temperature scaling 只在 validation split 上拟合 | 避免使用 test label 做校准，保证可信决策实验严谨 |

### 2.2 已完成的 PhysioNet full 外部验证

这是当前最完整、最可靠的一组结果。

| 项目 | 设置 |
| --- | --- |
| 数据集 | PhysioNet EEGBCI / EEGMMIDB |
| 被试 | 109 |
| 任务 | runs 4, 8, 12；左手/右手二分类 |
| 协议 | leave-one-subject-out, LOSO |
| 随机种子 | 42, 2024, 3407 |
| 方法 | Bandpower-LDA, CSP-LDA, EEGNet-Lite, HAB-Full, HAB-NoWorldModel, HAB-DenseGraph |
| 预处理 | 7-30 Hz bandpass, 0-4 s window, resample 80 Hz |
| 验证集 | 从训练被试中划分，测试被试不参与模型选择 |
| 结果行数 | 109 subjects x 3 seeds x 6 methods = 1962 |
| 完成状态 | 1962 / 1962 成功，0 failed |

服务器最终结果目录：

```bash
/root/autodl-tmp/hab_agentic_brain_physionet_20260706/results/full_physionet_loso
```

当前有效服务器：

```bash
ssh -p 31062 root@connect.westb.seetacloud.com
```

重要输出文件：

| 文件 | 内容 |
| --- | --- |
| `metrics.csv` | 每个 seed / test subject / method 的主分类指标 |
| `aggregate_metrics.csv` | 各方法聚合指标 |
| `risk_metrics.csv` | coverage-risk / selective decision 指标 |
| `world_model_metrics.csv` | HAB world-model next-state proxy |
| `graph_metrics.csv` | HAB 认知图结构 proxy |
| `summary.md` | 自动生成的实验摘要 |
| `config_used.json` | 实际使用的配置 |
| `run_meta.json` | 运行元信息 |

## 3. PhysioNet full 最终结果

完成状态：

| 项目 | 数值 |
| --- | --- |
| 生成时间 | 2026-07-07 11:30:02 |
| 成功主结果行数 | 1962 |
| 失败行数 | 0 |
| subjects | 109 |
| epochs | 4743 |
| methods | 6 |
| seeds | 3 |

主分类结果：

| 方法 | Acc | Macro-F1 | Kappa | NLL | ECE | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EEGNet-Lite | 0.806 | 0.798 | 0.613 | 0.470 | 0.122 | 0.276 |
| CSP-LDA | 0.608 | 0.553 | 0.215 | 0.682 | 0.140 | 0.479 |
| Bandpower-LDA | 0.570 | 0.519 | 0.140 | 0.693 | 0.104 | 0.493 |
| HAB-DenseGraph | 0.502 | 0.486 | 0.002 | 0.694 | 0.055 | 0.501 |
| HAB-NoWorldModel | 0.498 | 0.485 | -0.004 | 0.694 | 0.059 | 0.501 |
| HAB-Full | 0.494 | 0.484 | -0.011 | 0.694 | 0.060 | 0.501 |

统计比较中，HAB-Full 明显低于主要 baseline：

| 比较 | 指标 | HAB-Full - 对照 |
| --- | --- | ---: |
| HAB-Full vs EEGNet-Lite | Acc | -0.312 |
| HAB-Full vs EEGNet-Lite | Macro-F1 | -0.315 |
| HAB-Full vs CSP-LDA | Acc | -0.114 |
| HAB-Full vs CSP-LDA | Macro-F1 | -0.070 |
| HAB-Full vs Bandpower-LDA | Acc | -0.076 |
| HAB-Full vs Bandpower-LDA | Macro-F1 | -0.036 |

风险指标也不支持 HAB-Full 优于 baseline。以全覆盖 `coverage=1.0` 为例：

| 方法 | Selective Risk | Expected Cost |
| --- | ---: | ---: |
| EEGNet-Lite | 0.194 | 0.968 |
| CSP-LDA | 0.392 | 1.961 |
| Bandpower-LDA | 0.430 | 2.148 |
| HAB-DenseGraph | 0.498 | 2.491 |
| HAB-NoWorldModel | 0.502 | 2.508 |
| HAB-Full | 0.506 | 2.530 |

世界模型与图结构 proxy：

| 方法 | next-state MSE | copy baseline MSE | clustering proxy | edge entropy | mean max edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| HAB-Full | 0.170 | 0.203 | 0.041 | 1.099 | 0.334 |
| HAB-NoWorldModel | 0.176 | 0.209 | 0.041 | 1.099 | 0.334 |
| HAB-DenseGraph | 0.157 | 0.178 | 0.016 | 2.079 | 0.125 |

这些辅助指标说明：

1. HAB-Full 的 next-state proxy 比 copy baseline 更好，说明动态预测任务不是完全无效。
2. HAB-Full / HAB-NoWorldModel 与 DenseGraph 的图结构不同，说明稀疏认知图确实被实现并可度量。
3. 但这些结构性证据不能弥补分类性能不足，不能证明 HAB 当前版本带来更强解码能力。

## 4. 当前完整实验思路

当前建议把实验路线分成四个阶段。

### 阶段 A：代码与设计验证，已完成

目标：确认实验实现没有明显泄漏、指标文件能正常输出、HAB 关键模块能端到端训练。

已完成内容：

| 内容 | 状态 |
| --- | --- |
| PhysioNet 小规模 pilot | 已完成 |
| HAB 消融 pilot | 已完成 |
| split 泄漏检查 | 已完成 |
| world-model loss 修复 | 已完成 |
| cognitive graph top-k 修复 | 已完成 |
| validation-only temperature scaling | 已完成 |

### 阶段 B：PhysioNet 外部鲁棒性验证，已完成

目标：在大被试公开数据集上验证框架可运行、可复现，并观察 HAB 在外部数据集上的真实表现。

已完成内容：

| 内容 | 状态 |
| --- | --- |
| 109 被试 LOSO | 已完成 |
| 3 seeds 重复 | 已完成 |
| 6 methods 对比 | 已完成 |
| 主分类指标 | 已完成 |
| calibration 指标 | 已完成 |
| risk decision 指标 | 已完成 |
| world-model proxy | 已完成 |
| graph structure proxy | 已完成 |
| paired statistical test | 已完成基础版本 |

结论：PhysioNet 外部验证完成度高、真实性强，但结果不支持 HAB-Full 性能优越。

### 阶段 C：BNCI2014_001 主实验，未完成，建议优先做

这是 `代理脑论文1(1).docx` 里真正的主实验。当前如果要让论文更完整，下一步应该做 BNCI2014_001。

建议设置：

| 项目 | 建议 |
| --- | --- |
| 数据集 | BNCI2014_001 / BCI Competition IV 2a |
| 任务 | 左手、右手、双脚、舌部四分类 |
| 协议 1 | 被试内跨会话：session 1 train/val, session 2 test |
| 协议 2 | 跨被试 LOSO |
| baseline | CSP-LDA, FBCSP-LDA, EEGNet, ShallowConvNet, DeepConvNet |
| HAB 对照 | HAB-Full, w/o World Model, w/o Small-World, w/o Align, DenseGraph, Dynamic Neural Graph Only |
| seeds | 至少 3 个；若时间允许 5 个 |
| 指标 | Acc, Macro-F1, Kappa, NLL, ECE, Brier, selective risk, expected cost |

为什么必须做：因为论文原计划把 BNCI2014_001 作为核心数据集。仅有 PhysioNet 外部验证，不足以支撑“主实验完成”。

### 阶段 D：解释性、优化与补充实验，部分完成/未完成

| 实验 | 当前状态 | 说明 |
| --- | --- | --- |
| world-model 状态预测 | 部分完成 | 已有 next-state proxy，但缺 GRU Predictor 等更强时序 baseline |
| 小世界结构分析 | 部分完成 | 已有 edge entropy、mean max edge、clustering proxy；缺严格 small-worldness sigma 和随机图参照 |
| 风险感知决策 | 部分完成 | 已有 coverage-risk 离线模拟；缺更多 cost setting 和可视化 |
| 可解释性图 | 未完成 | 需要认知单元热力图、连接矩阵、subject-level 可视化 |
| 超参数分析 | 未完成 | K、top-k、lambda_wm、lambda_sw、lambda_align 系统搜索还没做 |
| SEED 跨任务扩展 | 未完成 | 可选，不建议在主实验之前做 |

## 5. 当前结果对论文的支撑边界

可以支撑的说法：

1. 我们建立了一个可复现的 HAB 实验系统，覆盖分类、世界模型、认知图、校准和风险决策。
2. PhysioNet full 外部验证完整完成，109 被试、3 seeds、1962 行结果全部成功。
3. HAB 结构模块可以被实现和度量，尤其是稀疏认知图与 DenseGraph 在图结构指标上有差异。
4. world-model proxy 不是完全无效，HAB-Full 的 next-state MSE 优于 copy baseline。
5. 风险和校准指标可以作为代理脑式 BCI 决策系统的补充评价维度。

不能支撑的说法：

1. 不能说 HAB-Full 分类性能优于 EEGNet-Lite、CSP-LDA 或 Bandpower-LDA。
2. 不能说世界模型显著提升分类准确率。
3. 不能说小世界结构已经带来更强泛化能力。
4. 不能把低 ECE 单独解释为模型更好，因为 HAB 系列接近随机二分类，校准好不等于判别能力强。
5. 不能说完整论文实验已经全部完成，因为 BNCI2014_001 主实验还没做。

建议论文结论调整为：

> 本文提出一个层级化代理脑实验框架，并在 PhysioNet EEGMMIDB 上完成了大规模外部验证。结果表明，当前 HAB 原型能够端到端实现神经图、认知单元图、世界模型和风险感知决策，但在当前实现和 PhysioNet 二分类设置下，分类性能尚未超过成熟 EEGNet/CSP baseline。结构与动态预测指标显示代理脑模块可被度量，但其判别能力仍需通过更强编码器、主数据集实验和进一步消融来改进。

## 6. 还需要补的关键实验

优先级从高到低：

| 优先级 | 实验 | 原因 |
| --- | --- | --- |
| 高 | BNCI2014_001 被试内跨会话 | 对应论文主数据集和主协议 |
| 高 | BNCI2014_001 跨被试 LOSO | 检验泛化能力 |
| 高 | 加 FBCSP-LDA、ShallowConvNet、DeepConvNet | 补足老师文档中的 baseline |
| 高 | Dynamic Neural Graph Only / w/o Cognitive Units | 检验认知单元抽象是否有用 |
| 中 | 严格 small-worldness + 随机图参照 | 支撑小世界结构 claim |
| 中 | GRU Predictor world-model baseline | 判断 HAB world model 是否真的比普通序列模型好 |
| 中 | 结果可视化 | 包括 per-subject bar、coverage-risk curve、reliability diagram、graph heatmap |
| 低 | SEED 情绪任务扩展 | 只有主实验闭环后再做 |

## 7. 当前代码与复现命令

本地主要代码：

| 文件 | 作用 |
| --- | --- |
| `code/run_full_physionet_experiment.py` | PhysioNet full LOSO 实验主脚本 |
| `code/config_full_physionet.json` | PhysioNet full 配置 |
| `code/run_pilot_experiment.py` | pilot 和共享模型代码 |
| `code/run_pilot_ablations.py` | pilot 消融脚本 |
| `scripts/predownload_then_run_full.sh` | PhysioNet EDF 预下载并启动 full 的辅助脚本 |

本地记录文档：

| 文件 | 作用 |
| --- | --- |
| `实验思路与pilot记录.md` | 实验思路、pilot、服务器运行记录 |
| `实验结果与设计检查.md` | full 结果、设计检查、论文结论边界 |
| `README_代理脑实验状态.md` | 当前实验状态总览，也就是本文档 |

服务器复现/检查命令：

```bash
cd /root/autodl-tmp/hab_agentic_brain_physionet_20260706
sed -n '1,220p' results/full_physionet_loso/summary.md
python3 - <<'PY'
import pandas as pd
base = 'results/full_physionet_loso'
print(pd.read_csv(f'{base}/aggregate_metrics.csv').to_string(index=False))
print('metrics rows:', len(pd.read_csv(f'{base}/metrics.csv')))
PY
```

如果要重新跑 PhysioNet full，不要覆盖已有正式结果，建议换新结果目录：

```bash
python3 code/run_full_physionet_experiment.py \
  --config code/config_full_physionet.json \
  --result-dir results/full_physionet_loso_rerun
```

## 8. 总结

目前我们完成的是一条扎实的外部验证线：PhysioNet EEGMMIDB 109 被试 full LOSO。它的实验规模、可复现性和记录完整度都足够好，可以作为论文中的外部鲁棒性或反例分析结果。

但是它没有证明 HAB-Full 性能优越。相反，它提示当前 HAB 原型的判别能力不足，尤其弱于 EEGNet-Lite。这不是坏事，但论文叙事必须从“模型显著更强”调整为“框架可实现、模块可度量、当前原型仍需改进”。

下一步最合理的是补 BNCI2014_001 主实验。如果 BNCI 上 HAB 仍然弱，则论文应定位为一个严谨的代理脑建模框架与负结果/诊断性研究；如果 BNCI 上某些 HAB 变体在校准、风险或结构稳定性上表现出优势，则可以把论文结论集中在可信 BCI 决策和结构化认知建模，而不是单纯分类准确率。
