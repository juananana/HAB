Hierarchical Agentic Brain: A Multi-Scale World Model for Cognitive State Modeling and Brain–Computer Interface
本文提出一种层级化代理脑模型，将脑机接口中的神经信号解码问题重新表述为一个多尺度动态状态估计、认知单元追踪与任务意图优化决策问题。
旨在提出一种面向脑机接口与认知状态建模的层级化代理脑框架，将传统脑机接口中的“脑信号到类别标签”的静态解码范式，扩展为“神经信号—认知单元—意图推理—代理决策”的动态建模过程。该框架认为，人脑并非单纯的信号发生器，而是一个具有层级结构、记忆机制、状态预测和自我监控能力的内部世界模型。因此，本文首先从 EEG、fMRI、fNIRS 等多模态脑信号中构建动态神经图，刻画脑区或通道之间的功能连接；随后进一步将低层神经活动抽象为由注意、记忆、情绪、任务规则、行为意图和不确定性等组成的认知单元图；在此基础上，引入小世界网络结构约束，使认知单元在保持局部聚类的同时具备全局高效整合能力；最后，通过认知世界模型预测认知状态的动态演化，并结合不确定性感知的代理决策机制，实现更加可信、可解释和自适应的脑机交互。整体而言，本文试图将脑机接口从被动的神经解码器提升为主动的代理脑模型，为认知状态识别、意图理解、情绪建模以及安全脑机交互提供统一的计算框架。
 
 
1. Introduction
Background: From Neural Decoding to Agentic BCI
介绍传统 BCI 和智能交互趋势。
Limitations of Existing Methods
指出现有方法存在：
缺乏层级建模； 
缺乏动态认知状态追踪； 
缺乏结构先验； 
缺乏可信决策机制。 
Motivation: Human Brain as a Hierarchical Small-World World Model
引出人脑的层级、小世界和世界模型特征。
Proposed Framework
提出 Hierarchical Agentic Brain。
Contributions
列出 4 个贡献：
提出一种面向 BCI 的层级化代理脑框架，将脑信号解码转化为神经—认知—意图—交互的动态建模问题。  
提出认知单元图建模方法，将低层神经活动抽象为注意、记忆、情绪、意图和不确定性等中间认知变量。  
引入小世界结构约束，使认知单元图同时具备局部功能聚合和全局信息整合能力。  
构建认知世界模型与不确定性感知决策机制，实现可预测、可解释、可信的脑机交互。
2. Related Work
2.1 Brain–Computer Interface and Neural Decoding
2.2 Graph Neural Networks for Brain Network Modeling
2.3 World Models and Agentic AI

3. Problem Formulation
本节将脑机接口中的传统神经解码任务重新表述为一个层级化代理脑建模问题。与现有方法直接将脑信号映射为任务标签或控制指令不同，本文认为脑机接口系统应同时建模低层神经状态、高层认知单元、认知状态转移、不确定性估计以及交互决策过程。因此，我们将脑机接口建模为一个由动态神经图、认知单元图和代理决策目标共同组成的多尺度优化问题。
3.1 代理脑的层次化建模
传统脑机接口方法通常将神经解码视为一个监督分类或回归问题。给定时间 t 的脑信号输入 Xt，模型直接预测对应的任务标签、认知状态或控制指令：
yt=fθXt,
其中，Xt 表示 EEG、fMRI、fNIRS 或其他脑信号观测，yt 表示模型预测的用户意图、情绪状态、运动想象类别或脑机控制指令，fθ⋅ 为参数为 θ 的神经解码模型。
然而，这种直接映射范式存在明显局限。首先，它将人脑简化为一个静态信号源，忽略了脑状态随任务、环境和反馈动态变化的过程。其次，它缺乏对高层认知状态的显式建模，难以解释神经信号背后的注意、记忆、情绪、意图和不确定性等认知因素。最后，传统解码模型通常被迫输出确定性结果，缺少对预测可靠性和交互风险的建模。
为此，本文将 BCI 问题从传统的神经解码扩展为代理脑建模问题。给定连续脑信号序列、外部任务环境和历史交互反馈，模型需要构建动态神经状态，进一步抽象为认知单元状态，并基于认知状态完成意图预测、不确定性估计和代理决策。整体过程可表示为：
X1:T,E1:T,A1:T→G1:Tn...G1:Tc→y1:T,u1:T,a1:T,
其中，X1:T 表示从 1 到 T 时刻的脑信号序列，E1:T 表示任务环境或上下文信息，A1:T 表示历史系统动作或反馈，G1:Tn 表示动态神经图序列，G1:Tc 表示认知单元图序列，y1:T 表示预测的用户意图或任务标签，u1:T 表示预测不确定性，a1:T 表示系统最终采取的交互动作。

因此，本文的核心目标不是简单学习从脑信号到标签的映射，而是学习一个代理脑模型，使其能够从神经信号中推断潜在认知状态，预测认知状态的动态演化，并在不确定性感知的条件下做出可靠的脑机交互决策。
这里面涉及到代理脑的层次化的设计，层次化越高，智能型越强

3.2 代理脑的世界模型刻画精度
3.2.1 图神经网络架构
为了刻画脑信号中不同脑区、通道或神经节点之间的动态交互关系，本文首先将低层脑信号建模为动态神经图。对于时间 t 的脑信号输入 Xt，其可以表示为：
Xt∈ℝC×L,
其中，C 表示脑信号通道数或脑区数量，L 表示时间窗口长度。经过神经编码器后，可以得到节点级神经表征：
Htn=fθnXt,
其中，Htn∈ℝC×d 表示时间 t 的神经节点特征矩阵，d 表示特征维度，fθn⋅ 表示神经状态编码函数。
基于节点表征，可以构建动态神经图：
Gtn=Vtn,ℰtn,Htn,
其中，Vtn 表示神经节点集合，ℰtn 表示神经节点之间的动态连接集合，Htn 表示节点特征。
动态神经连接可由邻接矩阵 Atn 表示。对于任意两个神经节点 i 和 j，其连接强度可以定义为：
eijt=simhit,hjt,
其中，hit 和 hjt 分别表示第 i 和第 j 个神经节点的特征，sim⋅,⋅ 表示相似度函数。
也可以采用可学习的注意力机制构建动态邻接矩阵：
Atn=softmaxQtKt⊤d,
其中：
Qt=HtnWQn,  Kt=HtnWKn.
WQn 和 WKn 为可学习参数矩阵。由此，动态神经图可进一步写为：
Gtn=Htn,Atn.
动态神经图的作用是将原始脑信号从孤立的时序特征转化为结构化脑状态，从而为后续认知单元抽象提供基础。
3.2.2 基于认知图的代理脑构建
仅依赖动态神经图仍然难以直接解释高层认知状态。为此，本文进一步引入认知单元图，将低层神经活动抽象为一组潜在认知单元。认知单元可以对应注意、记忆、情绪、任务规则、行为意图、不确定性等高层认知因素，是连接神经信号和代理决策的中间状态变量。
给定动态神经图 Gtn，认知单元集合定义为：
Ct=fθcGtn=c1t,c2t,…,cKt,
其中，K 表示认知单元数量，fθc⋅ 表示神经到认知的抽象函数，cit 表示第 i 个认知单元。
每个认知单元可以表示为：
cit=zit,αit,νit,μit,rit,
其中，zit 表示认知嵌入向量，αit 表示认知单元的激活水平，νit 表示情绪效价或情感状态，μit 表示该认知单元的不确定性，rit 表示该节点在认知图中的结构角色特征，例如中心性、聚类系数或模块归属。
认知单元之间并非相互独立，而是存在动态交互关系。例如，情绪状态可能调节注意分配，记忆状态可能影响当前意图生成，任务规则可能约束行为决策。因此，我们进一步构建认知单元图：
Gtc=Vtc,ℰtc,Ct,
其中，Vtc 表示认知单元节点集合，ℰtc 表示认知单元之间的交互关系，Ct 表示认知单元特征。
认知图也可以简写为：
Gtc=Ct,Atc,
其中，Atc 表示认知单元之间的邻接矩阵。其可由认知单元特征自适应学习得到：
Atc=softmaxCtWQcCtWKc⊤d,
其中，WQc 和 WKc 为可学习参数。
认知单元图的引入使得模型不再直接从脑信号预测输出，而是首先形成一个具有结构化语义的潜在认知状态空间。该状态空间能够表达认知单元之间的激活、抑制、协同和调节关系，从而为后续认知状态预测和代理决策提供可解释基础。
3.3 代理脑的交互性构建
基于动态神经图和认知单元图，本文进一步定义代理脑的总体目标。与传统 BCI 只优化任务预测性能不同，代理脑需要同时完成四类交互：任务解码、认知状态预测、结构化认知建模和不确定性感知交互决策。
首先，代理脑需要基于认知单元图预测当前任务标签或用户意图：
yt=fθyGtc,
其中，fθy⋅ 表示任务预测函数。
其次，代理脑需要估计当前预测的不确定性：
ut=fθuGtc,
其中，ut 表示模型在时间 t 的预测不确定性，fθu⋅ 为不确定性估计函数。
同时，为了刻画认知状态随时间的演化过程，代理脑需要学习一个认知世界模型，用于预测下一时刻的认知单元图：
Gt+1c=fθwGtc,At,Et,
其中，fθw⋅ 表示认知世界模型，At 表示历史系统动作或反馈，Et 表示外部任务环境。
最后，代理脑根据预测结果、不确定性和任务环境生成系统动作：
at=πθyt,ut,Et,
其中，πθ⋅ 表示代理决策策略。该策略可以根据不确定性决定是否执行预测结果、请求用户确认或延迟决策：
at=executeyt,ut≤δ,request_confirmation,ut>δ,
其中，δ 表示不确定性阈值。
这里面体现出代理脑的从按需交互-

3.4 总体模型的构建
综合上述建模，本文将代理脑学习定义为一个多目标优化问题。模型不仅需要最小化任务预测误差，还需要保证认知状态可预测、认知图结构合理、不确定性估计可靠，并降低高风险交互决策的代价。
总损失函数定义为：
ℒ=ℒtask+λ1ℒwm+λ2ℒsw+λ3ℒunc+λ4ℒrisk+λ5ℒalign,
其中，ℒtask 表示任务预测损失，ℒwm 表示认知世界模型预测损失，ℒsw 表示认知单元图的小世界结构约束，ℒunc 表示不确定性校准损失，ℒrisk 表示交互风险损失，ℒalign 表示神经状态与认知状态之间的对齐损失，λ1,…,λ5 为权重系数。
任务预测损失可以定义为交叉熵形式：
ℒtask=−t=1Tk=1Kyyt,klogpt,k,
其中，Ky 表示任务类别数量，yt,k 表示真实标签，pt,k 表示模型预测为第 k 类的概率。
认知世界模型预测损失定义为：
ℒwm=t=1T−1Ct+1−Ct+122.
如果真实认知状态不可直接观测，也可以采用下一时刻脑信号重构作为替代监督：
ℒwm=t=1T−1Xt+1−RθCt+122.
小世界结构约束定义为：
ℒsw=−αCGtc+βLGtc−γQGtc+ηAtc1,
其中，CGtc 表示认知图的平均聚类系数，LGtc 表示平均最短路径长度，QGtc 表示模块度，∥Atc∥1 为稀疏约束项。该损失鼓励认知图具有局部聚类、全局短路径和适度模块化结构，同时避免退化为完全连接图。
不确定性损失可以定义为校准误差：
ℒunc=ECEpt,yt,
也可以采用负对数似然形式：
ℒunc=−t=1Tlogpθyt∣Xt.
交互风险损失定义为：
ℒrisk=t=1Tk=1KyRyt,kpθyt=k∣Xt,
其中，Ryt,k 表示真实标签为 yt 时错误执行类别 k 的风险代价。
神经—认知对齐损失定义为：
ℒalign=t=1TPoolHtn−gθCt22,
其中，Pool⋅ 表示神经图表征的池化操作，gθ⋅ 表示将认知单元映射回神经表征空间的对齐函数。
因此，完整的代理脑优化问题可以写为：
minθℒtask+λ1ℒwm+λ2ℒsw+λ3ℒunc+λ4ℒrisk+λ5ℒalign.
其约束条件包括：
Gtn=fθnXt,
Gtc=fθcGtn,
Gt+1c=fθwGtc,At,Et,
yt,ut=fθdGtc,
at=πθyt,ut,Et,
ut≤δ, if at is executed.
该优化问题表明，本文提出的代理脑模型并非单一分类器，而是一个联合优化的层级化认知建模系统。它同时学习神经状态表征、认知单元抽象、认知状态预测、不确定性估计和风险感知决策，从而将脑机接口从传统的被动神经解码扩展为主动、可解释且可信的代理脑建模过程。

4. Methodology
4.1 Overview of Hierarchical Agentic Brain
整体结构图。

Figure: Overall framework of Hierarchical Agentic Brain. The framework takes
multimodal brain signals and external task context as inputs, first converts
raw signals into dynamic neural graphs, then abstracts neural states into a
cognitive-unit graph, and finally uses a small-world cognitive world model to
predict state evolution. The right side of the framework corresponds to three
evaluation-facing outputs: task or intention decoding, uncertainty estimation,
and risk-aware interaction actions. The bottom training objective connects the
task loss, world-model prediction loss, small-world structural constraint,
uncertainty calibration loss, risk loss, and neural-cognitive alignment loss.
This figure should be used as the main framework figure in the paper because it
links the model modules directly to the planned experiments: decoding
performance, state prediction, uncertainty calibration, safe decision analysis,
and interpretability of cognitive units.

4.2 Dynamic Neural Graph Encoder
脑信号编码。
4.3 Neural-to-Cognitive Unit Abstraction 
神经状态到认知单元的抽象。
4.4 Small-world Cognitive World Model
世界认知图建模。
4.5 Uncertainty-Aware Agentic Decision Module
不确定性感知的代理决策。
4.6 Training Objective and Optimization
完整损失函数。
 
 
5. Experiments
This section provides the planned experimental design for the teacher-reviewed
draft. The goal at this stage is to make the experimental loop complete and
executable before running the final code. Therefore, the section specifies the
datasets, splits, baselines, metrics, ablations, visualizations, and result-table
templates, but does not report unverified numerical results. After the
experiments are completed, all placeholders marked as TBD should be replaced by
values produced from fixed code, fixed random seeds, and archived configuration
files.

The experiments are designed around five research questions:

RQ1. Does HAB improve neural decoding accuracy over classical EEG pipelines,
convolutional neural decoders, and graph-based neural encoders?

RQ2. Does the neural-to-cognitive abstraction provide useful intermediate
state representations beyond direct signal-to-label decoding?

RQ3. Does the cognitive world model improve temporal state prediction and
cross-session or cross-subject generalization?

RQ4. Does the small-world structural constraint produce more stable and
interpretable cognitive-unit graphs without sacrificing decoding performance?

RQ5. Does the uncertainty-aware decision module reduce unsafe executions under
selective prediction and confirmation-based interaction protocols?

5.1 Datasets and Experimental Settings

The main experiments use public EEG datasets because they provide reproducible
signals, labels, and established baselines. Although the framework is formulated
for EEG, fMRI, fNIRS, and multimodal BCI, the first experimental closure should
not claim multimodal superiority unless those modalities are actually evaluated.

Table 1 summarizes the planned datasets.

| Dataset | Role | Scale and signal | Task | Use in this paper |
| --- | --- | --- | --- | --- |
| BNCI2014_001 / BCI Competition IV 2a | Main dataset | 9 subjects, 2 sessions, 22 EEG channels plus 3 EOG channels, 250 Hz | Four-class motor imagery: left hand, right hand, feet, tongue | Primary within-subject cross-session, cross-subject, ablation, calibration, and interpretability experiments |
| PhysioNet EEGMMIDB | External robustness dataset | 109 subjects, 64 EEG channels, multiple motor and imagery runs | Compatible motor imagery / movement classification tasks | Robustness test under different channel layout, acquisition protocol, and subject scale |
| SEED | Optional transfer dataset | 15 subjects, 62 EEG channels, emotion elicitation | Positive, neutral, negative emotion classification | Optional cross-task validation only after the motor-imagery loop is complete |

For BNCI2014_001, we use two evaluation protocols. In the within-subject
cross-session protocol, session 1 is used for training and validation, and
session 2 is held out for final testing. In the cross-subject protocol, we use
leave-one-subject-out evaluation: one subject is held out for testing, while the
remaining subjects are used for training and validation. Validation subjects or
validation trials are selected only inside the training pool.

Preprocessing is fixed before final testing. The first runnable protocol should
retain EEG channels, use EOG only for artifact inspection, apply a band-pass
filter, segment each trial into task-relevant time windows, and standardize each
channel using statistics computed only from the training split. All filtering
ranges, window boundaries, overlap ratios, resampling choices, rejected trials,
and random seeds must be written to the run logs.

Table 2 lists the configuration items that must be frozen before reporting final
results.

| Item | Planned setting | Must be archived before final reporting |
| --- | --- | --- |
| Data split | Within-subject cross-session and leave-one-subject-out | Exact train, validation, and test indices |
| Preprocessing | EEG-only input, band-pass filtering, windowing, channel-wise standardization | Filter range, window length, stride, overlap, resampling, rejected trials |
| Optimization | Same optimizer family and early stopping rule across neural methods | Optimizer, learning rate, batch size, max epochs, patience, scheduler |
| Repeated runs | At least 5 random seeds per neural method | Seed list and mean plus standard deviation |
| Runtime | Single-GPU training and batch-size-1 inference logging | Hardware, software versions, parameters, FLOPs if available, latency |

5.2 Baselines

The baselines cover classical feature engineering, end-to-end EEG neural
decoding, and graph-based modeling. All methods use the same data split and
preprocessing. Hyperparameters are selected on the validation split only, and no
test-set result may be used for model selection.

| Category | Method | Purpose |
| --- | --- | --- |
| Classical EEG pipeline | CSP + LDA | Standard spatial-filter baseline for motor imagery |
| Classical EEG pipeline | FBCSP + LDA | Stronger multi-band spatial-filter baseline |
| Convolutional decoder | EEGNet | Compact and widely used EEG decoding baseline |
| Convolutional decoder | ShallowConvNet | Frequency-power-oriented neural baseline |
| Convolutional decoder | DeepConvNet | Deeper end-to-end temporal-spatial baseline |
| Graph model | Dynamic Neural Graph Only | Tests the benefit of graph neural signal encoding without cognitive units |
| Proposed model | HAB | Full model with neural graph, cognitive-unit graph, world model, uncertainty calibration, and risk-aware decision |

5.3 Main Comparison Experiments

The main comparison evaluates task decoding, calibration, and efficiency. For
classification, we report accuracy, macro-F1, and Cohen's kappa. Because HAB
contains an uncertainty-aware decision module, we also report negative
log-likelihood, expected calibration error, and Brier score. To avoid
overstating gains from a small number of subjects, we report per-subject results
and aggregate mean plus standard deviation across subjects and seeds.

Table 3 is the main result table to fill after final runs.

| Protocol | Method | Acc. (%) | Macro-F1 | Kappa | NLL | ECE | Params (M) | Latency (ms/sample) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Within-subject cross-session | CSP + LDA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Within-subject cross-session | FBCSP + LDA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Within-subject cross-session | EEGNet | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Within-subject cross-session | ShallowConvNet | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Within-subject cross-session | DeepConvNet | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Within-subject cross-session | Dynamic Neural Graph Only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Within-subject cross-session | HAB | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Leave-one-subject-out | EEGNet | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Leave-one-subject-out | Dynamic Neural Graph Only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Leave-one-subject-out | HAB | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

In addition to the table, the paper should include one grouped bar chart showing
accuracy and macro-F1 across methods, and one calibration plot comparing HAB
with the strongest non-HAB neural baseline.

5.4 Cognitive World Model Evaluation

To verify that the world model is not merely an auxiliary regularizer, we
separately evaluate next-state prediction. If explicit cognitive-state labels
are unavailable, we use next-window neural graph representations or
next-window signal reconstruction as proxy targets. We also evaluate whether
the predicted next cognitive state improves next-step task recognition.

| Method | Representation prediction error | Reconstruction error | Next-step Acc. | Purpose |
| --- | --- | --- | --- | --- |
| Last-state copy | TBD | TBD | TBD | Copies the current state as the next-state prediction |
| GRU predictor | TBD | TBD | TBD | Sequence model without cognitive-unit graph structure |
| HAB world model | TBD | TBD | TBD | Predicts state evolution with cognitive units and context |

5.5 Ablation Study

Ablations test which component contributes to performance, calibration,
structure, and risk-aware interaction. All variants keep the same data split,
optimizer, early stopping rule, and hyperparameter budget.

| Variant | Removed or replaced component | Acc. | Macro-F1 | ECE | Selective risk | Small-worldness |
| --- | --- | --- | --- | --- | --- | --- |
| Full HAB | All modules retained | TBD | TBD | TBD | TBD | TBD |
| w/o Neural Graph | Replaces dynamic graph encoder with channel-independent temporal encoder | TBD | TBD | TBD | TBD | TBD |
| w/o Cognitive Units | Pools neural graph representation directly for classification | TBD | TBD | TBD | TBD | TBD |
| w/o Small-World Constraint | Removes the small-world structural loss | TBD | TBD | TBD | TBD | TBD |
| w/o World Model | Removes next-state prediction objective | TBD | TBD | TBD | TBD | TBD |
| w/o Calibration | Keeps classifier probabilities without calibration loss | TBD | TBD | TBD | TBD | TBD |
| w/o Risk Policy | Always executes the maximum-probability prediction | TBD | TBD | TBD | TBD | TBD |

5.6 Uncertainty and Safe Decision Analysis

Offline public datasets cannot prove real closed-loop BCI safety, but they can
support a reproducible selective-decision simulation. For each test window, the
model outputs a predicted class and an uncertainty score. If uncertainty is
below a threshold, the system executes the predicted action. If uncertainty is
above the threshold, the system requests confirmation or delays the decision.
Sweeping the threshold produces a coverage-risk curve.

We report coverage, selective accuracy, selective risk, expected cost,
confirmation rate, ECE, NLL, and Brier score. Expected cost should be evaluated
under at least two cost settings: one where wrong execution is much more costly
than confirmation, and one where maintaining high interaction coverage is also
important.

| Strategy | Target coverage | Actual coverage | Selective Acc. | Selective risk | Expected cost | Confirmation rate | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Always execute | 100% | TBD | TBD | TBD | TBD | 0% | TBD |
| Entropy threshold | 90% | TBD | TBD | TBD | TBD | TBD | TBD |
| Entropy threshold | 80% | TBD | TBD | TBD | TBD | TBD | TBD |
| Calibrated threshold | 90% | TBD | TBD | TBD | TBD | TBD | TBD |
| Calibrated threshold | 80% | TBD | TBD | TBD | TBD | TBD | TBD |

This analysis should be framed as offline evidence for risk-aware decision
behavior, not as proof of real online BCI safety. If online interaction becomes
a central claim, a pseudo-online or small-scale closed-loop user study must be
added.

5.7 Interpretability and Structural Analysis

Interpretability is evaluated at three levels. At the neural-graph level, we
visualize dynamic connectivity matrices and channel topographies. At the
cognitive-unit level, we visualize unit activation heatmaps, inter-unit
connectivity, temporal trajectories, and class-conditional unit usage. At the
decision level, we analyze high-uncertainty windows and compare the graph states
that lead to execution, confirmation, or delayed decisions.

For the small-world structure, we report the average clustering coefficient,
average shortest-path length, modularity, sparsity, and small-worldness. Random
graph references should be resampled multiple times to reduce baseline
variance. Because the cognitive units are learned latent variables, the paper
should not claim that a unit literally equals attention, memory, or emotion
without independent cognitive labels. The supported claim is that the learned
units provide task-relevant, structured, and inspectable intermediate states.

5.8 Statistical Testing and Reporting Protocol

For each primary metric, we compare HAB against the strongest baseline under the
same evaluation protocol. If metric distributions are approximately normal, we
use paired t-tests across subjects; otherwise, we use Wilcoxon signed-rank
tests. Multiple comparisons are corrected with Holm-Bonferroni correction.
Effect sizes should be reported together with p-values. For all neural methods,
we report mean and standard deviation across at least 5 random seeds.

The final paper should include a claim-evidence matrix:

| Claim | Required experiment | Required evidence before claiming |
| --- | --- | --- |
| HAB improves neural decoding | Main comparison on BNCI2014_001 | Better mean accuracy / macro-F1 than strongest baseline with statistical support |
| Cognitive units are useful | w/o Cognitive Units ablation | Performance or calibration drop after removing the abstraction layer |
| World model improves temporal modeling | Next-state prediction and w/o World Model ablation | Lower prediction error or better next-step accuracy |
| Small-world prior improves structure | w/o Small-World Constraint and graph metrics | Better structural metrics without unacceptable task-performance loss |
| Risk-aware policy improves safety | Selective-decision simulation | Lower selective risk or expected cost at matched coverage |
| Framework is robust beyond one dataset | PhysioNet EEGMMIDB external test | Consistent trend under fixed external protocol |
 
6. Conclusion
本文提出一种层级化代理脑框架，将 BCI 从脑信号分类扩展为神经—认知—意图—交互的动态建模问题。通过认知单元抽象、小世界结构约束、认知世界模型和不确定性决策机制，该框架能够实现更准确、更可解释、更可信的脑机接口。
 
