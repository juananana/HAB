**Hierarchical Agentic Brain: A Multi-Scale World Model for Cognitive
State Modeling and Brain--Computer Interface**

本文提出一种层级化代理脑模型，将脑机接口中的神经信号解码问题重新表述为一个多尺度动态状态估计、认知单元追踪与任务意图优化决策问题。

旨在提出一种面向脑机接口与认知状态建模的**层级化代理脑框架**，将传统脑机接口中的"脑信号到类别标签"的静态解码范式，扩展为"神经信号---认知单元---意图推理---代理决策"的动态建模过程。该框架认为，人脑并非单纯的信号发生器，而是一个具有层级结构、记忆机制、状态预测和自我监控能力的内部世界模型。因此，本文首先从
EEG、fMRI、fNIRS
等多模态脑信号中构建动态神经图，刻画脑区或通道之间的功能连接；随后进一步将低层神经活动抽象为由注意、记忆、情绪、任务规则、行为意图和不确定性等组成的认知单元图；在此基础上，引入小世界网络结构约束，使认知单元在保持局部聚类的同时具备全局高效整合能力；最后，通过认知世界模型预测认知状态的动态演化，并结合不确定性感知的代理决策机制，实现更加可信、可解释和自适应的脑机交互。整体而言，本文试图将脑机接口从被动的神经解码器提升为主动的代理脑模型，为认知状态识别、意图理解、情绪建模以及安全脑机交互提供统一的计算框架。

 

 

# 1. Introduction

**Background: From Neural Decoding to Agentic BCI**

介绍传统 BCI 和智能交互趋势。

**Limitations of Existing Methods**

指出现有方法存在：

- 缺乏层级建模；

- 缺乏动态认知状态追踪；

- 缺乏结构先验；

- 缺乏可信决策机制。

**Motivation: Human Brain as a Hierarchical Small-World World Model**

引出人脑的层级、小世界和世界模型特征。

Proposed Framework

提出 Hierarchical Agentic Brain。

**Contributions**

列出 4 个贡献：

提出一种面向 BCI
的层级化代理脑框架，将脑信号解码转化为神经---认知---意图---交互的动态建模问题。

提出认知单元图建模方法，将低层神经活动抽象为注意、记忆、情绪、意图和不确定性等中间认知变量。

引入小世界结构约束，使认知单元图同时具备局部功能聚合和全局信息整合能力。

构建认知世界模型与不确定性感知决策机制，实现可预测、可解释、可信的脑机交互。

# 2. Related Work

**2.1 Brain--Computer Interface and Neural Decoding**

**2.2 Graph Neural Networks for Brain Network Modeling**

**2.3 World Models and Agentic AI**

# 3. Problem Formulation

本节将脑机接口中的传统神经解码任务重新表述为一个层级化代理脑建模问题。与现有方法直接将脑信号映射为任务标签或控制指令不同，本文认为脑机接口系统应同时建模低层神经状态、高层认知单元、认知状态转移、不确定性估计以及交互决策过程。因此，我们将脑机接口建模为一个由动态神经图、认知单元图和代理决策目标共同组成的多尺度优化问题。

**3.1 代理脑的层次化建模**

传统脑机接口方法通常将神经解码视为一个监督分类或回归问题。给定时间 $t$
的脑信号输入 $X_{t}$，模型直接预测对应的任务标签、认知状态或控制指令：

$${\widehat{y}}_{t} = f_{\theta}\left( X_{t} \right),$$

其中，$X_{t}$ 表示 EEG、fMRI、fNIRS
或其他脑信号观测，${\widehat{y}}_{t}$
表示模型预测的用户意图、情绪状态、运动想象类别或脑机控制指令，$f_{\theta}( \cdot )$
为参数为 $\theta$ 的神经解码模型。

然而，这种直接映射范式存在明显局限。首先，它将人脑简化为一个静态信号源，忽略了脑状态随任务、环境和反馈动态变化的过程。其次，它缺乏对高层认知状态的显式建模，难以解释神经信号背后的注意、记忆、情绪、意图和不确定性等认知因素。最后，传统解码模型通常被迫输出确定性结果，缺少对预测可靠性和交互风险的建模。

为此，本文将 BCI
问题从传统的神经解码扩展为代理脑建模问题。给定连续脑信号序列、外部任务环境和历史交互反馈，模型需要构建动态神经状态，进一步抽象为认知单元状态，并基于认知状态完成意图预测、不确定性估计和代理决策。整体过程可表示为：

$$X_{1:T},E_{1:T},A_{1:T} \rightarrow \mathcal{G}_{1:T}^{n}...\mathcal{G}_{1:T}^{c} \rightarrow {\widehat{y}}_{1:T},u_{1:T},a_{1:T},$$

其中，$X_{1:T}$ 表示从 $1$ 到 $T$ 时刻的脑信号序列，$E_{1:T}$
表示任务环境或上下文信息，$A_{1:T}$
表示历史系统动作或反馈，$\mathcal{G}_{1:T}^{n}$
表示动态神经图序列，$\mathcal{G}_{1:T}^{c}$
表示认知单元图序列，${\widehat{y}}_{1:T}$
表示预测的用户意图或任务标签，$u_{1:T}$ 表示预测不确定性，$a_{1:T}$
表示系统最终采取的交互动作。

因此，本文的核心目标不是简单学习从脑信号到标签的映射，而是学习一个代理脑模型，使其能够从神经信号中推断潜在认知状态，预测认知状态的动态演化，并在不确定性感知的条件下做出可靠的脑机交互决策。

这里面涉及到代理脑的层次化的设计，层次化越高，智能型越强

**3.2 代理脑的世界模型刻画精度**

**3.2.1 图神经网络架构**

为了刻画脑信号中不同脑区、通道或神经节点之间的动态交互关系，本文首先将低层脑信号建模为动态神经图。对于时间
$t$ 的脑信号输入 $X_{t}$，其可以表示为：

$$X_{t} \in \mathbb{R}^{C \times L},$$

其中，$C$ 表示脑信号通道数或脑区数量，$L$
表示时间窗口长度。经过神经编码器后，可以得到节点级神经表征：

$$\mathbf{H}_{t}^{n} = f_{\theta}^{n}\left( X_{t} \right),$$

其中，$\mathbf{H}_{t}^{n} \in \mathbb{R}^{C \times d}$ 表示时间 $t$
的神经节点特征矩阵，$d$ 表示特征维度，$f_{\theta}^{n}( \cdot )$
表示神经状态编码函数。

基于节点表征，可以构建动态神经图：

$$\mathcal{G}_{t}^{n} = \left( \mathcal{V}_{t}^{n},\mathcal{E}_{t}^{n},\mathbf{H}_{t}^{n} \right),$$

其中，$\mathcal{V}_{t}^{n}$ 表示神经节点集合，$\mathcal{E}_{t}^{n}$
表示神经节点之间的动态连接集合，$\mathbf{H}_{t}^{n}$ 表示节点特征。

动态神经连接可由邻接矩阵 $\mathbf{A}_{t}^{n}$ 表示。对于任意两个神经节点
$i$ 和 $j$，其连接强度可以定义为：

$$e_{ij}^{t} = sim\left( \mathbf{h}_{i}^{t},\mathbf{h}_{j}^{t} \right),$$

其中，$\mathbf{h}_{i}^{t}$ 和 $\mathbf{h}_{j}^{t}$ 分别表示第 $i$ 和第
$j$ 个神经节点的特征，$sim( \cdot , \cdot )$ 表示相似度函数。

也可以采用可学习的注意力机制构建动态邻接矩阵：

$$\mathbf{A}_{t}^{n} = softmax\left( \frac{\mathbf{Q}_{t}\mathbf{K}_{t}^{\top}}{\sqrt{d}} \right),$$

其中：

$$\mathbf{Q}_{t} = \mathbf{H}_{t}^{n}\mathbf{W}_{Q}^{n},\quad\quad\mathbf{K}_{t} = \mathbf{H}_{t}^{n}\mathbf{W}_{K}^{n}.$$

$\mathbf{W}_{Q}^{n}$ 和 $\mathbf{W}_{K}^{n}$
为可学习参数矩阵。由此，动态神经图可进一步写为：

$$\mathcal{G}_{t}^{n} = \left( \mathbf{H}_{t}^{n},\mathbf{A}_{t}^{n} \right).$$

动态神经图的作用是将原始脑信号从孤立的时序特征转化为结构化脑状态，从而为后续认知单元抽象提供基础。

**3.2.2 基于认知图的代理脑构建**

仅依赖动态神经图仍然难以直接解释高层认知状态。为此，本文进一步引入认知单元图，将低层神经活动抽象为一组潜在认知单元。认知单元可以对应注意、记忆、情绪、任务规则、行为意图、不确定性等高层认知因素，是连接神经信号和代理决策的中间状态变量。

给定动态神经图 $\mathcal{G}_{t}^{n}$，认知单元集合定义为：

$$\mathbf{C}_{t} = f_{\theta}^{c}\left( \mathcal{G}_{t}^{n} \right) = \left\lbrack c_{1}^{t},c_{2}^{t},\ldots,c_{K}^{t} \right\rbrack,$$

其中，$K$ 表示认知单元数量，$f_{\theta}^{c}( \cdot )$
表示神经到认知的抽象函数，$c_{i}^{t}$ 表示第 $i$ 个认知单元。

每个认知单元可以表示为：

$$c_{i}^{t} = \left\lbrack \mathbf{z}_{i}^{t},\alpha_{i}^{t},\nu_{i}^{t},\mu_{i}^{t},\mathbf{r}_{i}^{t} \right\rbrack,$$

其中，$\mathbf{z}_{i}^{t}$ 表示认知嵌入向量，$\alpha_{i}^{t}$
表示认知单元的激活水平，$\nu_{i}^{t}$
表示情绪效价或情感状态，$\mu_{i}^{t}$
表示该认知单元的不确定性，$\mathbf{r}_{i}^{t}$
表示该节点在认知图中的结构角色特征，例如中心性、聚类系数或模块归属。

认知单元之间并非相互独立，而是存在动态交互关系。例如，情绪状态可能调节注意分配，记忆状态可能影响当前意图生成，任务规则可能约束行为决策。因此，我们进一步构建认知单元图：

$$\mathcal{G}_{t}^{c} = \left( \mathcal{V}_{t}^{c},\mathcal{E}_{t}^{c},\mathbf{C}_{t} \right),$$

其中，$\mathcal{V}_{t}^{c}$ 表示认知单元节点集合，$\mathcal{E}_{t}^{c}$
表示认知单元之间的交互关系，$\mathbf{C}_{t}$ 表示认知单元特征。

认知图也可以简写为：

$$\mathcal{G}_{t}^{c} = \left( \mathbf{C}_{t},\mathbf{A}_{t}^{c} \right),$$

其中，$\mathbf{A}_{t}^{c}$
表示认知单元之间的邻接矩阵。其可由认知单元特征自适应学习得到：

$$\mathbf{A}_{t}^{c} = softmax\left( \frac{\left( \mathbf{C}_{t}\mathbf{W}_{Q}^{c} \right)\left( \mathbf{C}_{t}\mathbf{W}_{K}^{c} \right)^{\top}}{\sqrt{d}} \right),$$

其中，$\mathbf{W}_{Q}^{c}$ 和 $\mathbf{W}_{K}^{c}$ 为可学习参数。

认知单元图的引入使得模型不再直接从脑信号预测输出，而是首先形成一个具有结构化语义的潜在认知状态空间。该状态空间能够表达认知单元之间的激活、抑制、协同和调节关系，从而为后续认知状态预测和代理决策提供可解释基础。

**3.3 代理脑的交互性构建**

基于动态神经图和认知单元图，本文进一步定义代理脑的总体目标。与传统 BCI
只优化任务预测性能不同，代理脑需要同时完成四类交互：任务解码、认知状态预测、结构化认知建模和不确定性感知交互决策。

首先，代理脑需要基于认知单元图预测当前任务标签或用户意图：

$${\widehat{y}}_{t} = f_{\theta}^{y}\left( \mathcal{G}_{t}^{c} \right),$$

其中，$f_{\theta}^{y}( \cdot )$ 表示任务预测函数。

其次，代理脑需要估计当前预测的不确定性：

$$u_{t} = f_{\theta}^{u}\left( \mathcal{G}_{t}^{c} \right),$$

其中，$u_{t}$ 表示模型在时间 $t$
的预测不确定性，$f_{\theta}^{u}( \cdot )$ 为不确定性估计函数。

同时，为了刻画认知状态随时间的演化过程，代理脑需要学习一个认知世界模型，用于预测下一时刻的认知单元图：

$${\widehat{\mathcal{G}}}_{t + 1}^{c} = f_{\theta}^{w}\left( \mathcal{G}_{t}^{c},A_{t},E_{t} \right),$$

其中，$f_{\theta}^{w}( \cdot )$ 表示认知世界模型，$A_{t}$
表示历史系统动作或反馈，$E_{t}$ 表示外部任务环境。

最后，代理脑根据预测结果、不确定性和任务环境生成系统动作：

$$a_{t} = \pi_{\theta}\left( {\widehat{y}}_{t},u_{t},E_{t} \right),$$

其中，$\pi_{\theta}( \cdot )$
表示代理决策策略。该策略可以根据不确定性决定是否执行预测结果、请求用户确认或延迟决策：

$$a_{t} = \left\{ \begin{matrix}
execute\left( {\widehat{y}}_{t} \right), & u_{t} \leq \delta, \\
request\_ confirmation, & u_{t} > \delta,
\end{matrix} \right.\ $$

其中，$\delta$ 表示不确定性阈值。

这里面体现出代理脑的从按需交互-

**3.4 总体模型的构建**

综合上述建模，本文将代理脑学习定义为一个多目标优化问题。模型不仅需要最小化任务预测误差，还需要保证认知状态可预测、认知图结构合理、不确定性估计可靠，并降低高风险交互决策的代价。

总损失函数定义为：

$$\mathcal{L =}\mathcal{L}_{task} + \lambda_{1}\mathcal{L}_{wm} + \lambda_{2}\mathcal{L}_{sw} + \lambda_{3}\mathcal{L}_{unc} + \lambda_{4}\mathcal{L}_{risk} + \lambda_{5}\mathcal{L}_{align},$$

其中，$\mathcal{L}_{task}$ 表示任务预测损失，$\mathcal{L}_{wm}$
表示认知世界模型预测损失，$\mathcal{L}_{sw}$
表示认知单元图的小世界结构约束，$\mathcal{L}_{unc}$
表示不确定性校准损失，$\mathcal{L}_{risk}$
表示交互风险损失，$\mathcal{L}_{align}$
表示神经状态与认知状态之间的对齐损失，$\lambda_{1},\ldots,\lambda_{5}$
为权重系数。

任务预测损失可以定义为交叉熵形式：

$$\mathcal{L}_{task} = - \sum_{t = 1}^{T}{\sum_{k = 1}^{K_{y}}y_{t,k}}\log{\widehat{p}}_{t,k},$$

其中，$K_{y}$ 表示任务类别数量，$y_{t,k}$
表示真实标签，${\widehat{p}}_{t,k}$ 表示模型预测为第 $k$ 类的概率。

认知世界模型预测损失定义为：

$$\mathcal{L}_{wm} = \sum_{t = 1}^{T - 1}\left. \parallel\mathbf{C}_{t + 1} - {\widehat{\mathbf{C}}}_{t + 1} \right.\parallel_{2}^{2}.$$

如果真实认知状态不可直接观测，也可以采用下一时刻脑信号重构作为替代监督：

$$\mathcal{L}_{wm} = \sum_{t = 1}^{T - 1}\left. \parallel X_{t + 1} - R_{\theta}\left( {\widehat{\mathbf{C}}}_{t + 1} \right) \right.\parallel_{2}^{2}.$$

小世界结构约束定义为：

$$\mathcal{L}_{sw} = - \alpha C\left( \mathcal{G}_{t}^{c} \right) + \beta L\left( \mathcal{G}_{t}^{c} \right) - \gamma Q\left( \mathcal{G}_{t}^{c} \right) + \eta\left. \parallel\mathbf{A}_{t}^{c} \right.\parallel_{1},$$

其中，$C\left( \mathcal{G}_{t}^{c} \right)$
表示认知图的平均聚类系数，$L\left( \mathcal{G}_{t}^{c} \right)$
表示平均最短路径长度，$Q\left( \mathcal{G}_{t}^{c} \right)$
表示模块度，$\parallel \mathbf{A}_{t}^{c} \parallel_{1}$
为稀疏约束项。该损失鼓励认知图具有局部聚类、全局短路径和适度模块化结构，同时避免退化为完全连接图。

不确定性损失可以定义为校准误差：

$$\mathcal{L}_{unc} = ECE\left( {\widehat{p}}_{t},y_{t} \right),$$

也可以采用负对数似然形式：

$$\mathcal{L}_{unc} = - \sum_{t = 1}^{T}\log p_{\theta}\left( y_{t} \mid X_{t} \right).$$

交互风险损失定义为：

$$\mathcal{L}_{risk} = \sum_{t = 1}^{T}{\sum_{k = 1}^{K_{y}}R}\left( y_{t},k \right)p_{\theta}\left( {\widehat{y}}_{t} = k \mid X_{t} \right),$$

其中，$R\left( y_{t},k \right)$ 表示真实标签为 $y_{t}$ 时错误执行类别
$k$ 的风险代价。

神经---认知对齐损失定义为：

$$\mathcal{L}_{align} = \sum_{t = 1}^{T}\left. \parallel Pool\left( \mathbf{H}_{t}^{n} \right) - g_{\theta}\left( \mathbf{C}_{t} \right) \right.\parallel_{2}^{2},$$

其中，$Pool( \cdot )$ 表示神经图表征的池化操作，$g_{\theta}( \cdot )$
表示将认知单元映射回神经表征空间的对齐函数。

因此，完整的代理脑优化问题可以写为：

$$\min_{\theta}\mathcal{L}_{task} + \lambda_{1}\mathcal{L}_{wm} + \lambda_{2}\mathcal{L}_{sw} + \lambda_{3}\mathcal{L}_{unc} + \lambda_{4}\mathcal{L}_{risk} + \lambda_{5}\mathcal{L}_{align}.$$

其约束条件包括：

$$\mathcal{G}_{t}^{n} = f_{\theta}^{n}\left( X_{t} \right),$$

$$\mathcal{G}_{t}^{c} = f_{\theta}^{c}\left( \mathcal{G}_{t}^{n} \right),$$

$${\widehat{\mathcal{G}}}_{t + 1}^{c} = f_{\theta}^{w}\left( \mathcal{G}_{t}^{c},A_{t},E_{t} \right),$$

$${\widehat{y}}_{t},u_{t} = f_{\theta}^{d}\left( \mathcal{G}_{t}^{c} \right),$$

$$a_{t} = \pi_{\theta}\left( {\widehat{y}}_{t},u_{t},E_{t} \right),$$

$$u_{t} \leq \delta,\quad\text{if }a_{t}\text{ is executed}.$$

该优化问题表明，本文提出的代理脑模型并非单一分类器，而是一个联合优化的层级化认知建模系统。它同时学习神经状态表征、认知单元抽象、认知状态预测、不确定性估计和风险感知决策，从而将脑机接口从传统的被动神经解码扩展为主动、可解释且可信的代理脑建模过程。

# 4. Methodology

**4.1 Overview of Hierarchical Agentic Brain**

整体结构图。

**4.2 Dynamic Neural Graph Encoder**

脑信号编码。

**4.3 Neural-to-Cognitive Unit Abstraction**

神经状态到认知单元的抽象。

**4.4 Small-world Cognitive World Model**

世界认知图建模。

**4.5 Uncertainty-Aware Agentic Decision Module**

不确定性感知的代理决策。

**4.6 Training Objective and Optimization**

完整损失函数。

 

** **

# 5. Experiments

**5.1 Datasets and Experimental Settings**

**5.2 Baselines**

**5.3 Main Results**

**5.4 Ablation Study**

**5.5 Optimization and Hyperparameter Analysis**

**5.6 Uncertainty and Safe Decision Analysis**

**5.7 Interpretability of Cognitive Units and Small-World Structure**

 

# 6. Conclusion

本文提出一种层级化代理脑框架，将 BCI
从脑信号分类扩展为神经---认知---意图---交互的动态建模问题。通过认知单元抽象、小世界结构约束、认知世界模型和不确定性决策机制，该框架能够实现更准确、更可解释、更可信的脑机接口。

 
