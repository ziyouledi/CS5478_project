# 文献笔记：低维接口与 LLM 控制灵巧手

整理日期：2026-11-12。来源：2026-11-11 至 11-12 的网络检索。

**可信度标注**

- 【已核对】检索结果确认了会议或期刊。
- 【预印本】arXiv、GitHub、技术报告，未经同行评审。
- 【凭记忆】检索中没有核对 venue，引用前必须自己查。
- 【社交媒体】X、小红书、博客，只能当线索。

所有数字来自摘要或网页抓取片段，写进报告前请对照原文。`Docs/Literature review/` 里已有的 `lin25d`、`xu25b`、`xu25d`、`p104` 四个 PDF 当时没能打开，可能与下面有重复。

---

## 一、灵巧手维度压缩 / synergy

| 文献 | 出处 | 要点 |
|---|---|---|
| Santello et al., *Postural Hand Synergies for Tool Use* | J. Neurosci. 1998【凭记忆】 | 人手抓握姿态，2 个主成分解释 80% 以上方差 |
| Ciocarlie & Allen, eigengrasps | IJRR 2009【凭记忆】；[RSS 2007 workshop 版](https://projects.csail.mit.edu/manipulation/rss07/paper__dexterous_grasping_via_eigengrasps_a_low_dimensional_approach_to_a_high_complexity_problem__ciocarlie.pdf) | 抓取规划在低维 eigengrasp 空间里做 |
| [CrossDex](https://iclr.cc/media/iclr-2025/Slides/28010.pdf) | ICLR 2025【已核对】 | 用 MANO 手模型的 eigengrasp 作跨手统一动作空间，45 维压到低维 |
| [DiscoSyn](https://arxiv.org/pdf/2110.01530)，He & Ciocarlie | ICRA 2022【凭记忆】 | 多任务 RL 中端到端学 synergy |
| [MyoDex](https://proceedings.mlr.press/v202/caggiano23a/caggiano23a.pdf) | ICML 2023【已核对】 | 肌肉骨骼手的多任务先验 |
| SAR | RSS 2023【凭记忆】 | 肌肉 synergy 表示迁移 |
| [West et al., Synergy-Based Complexity Index](https://michaelwestjr.github.io/assets/pdf/The_Study_of_Dexterous_Hand_Manipulation_A_Synergy-Based_Complexity_Index.pdf) | IEEE T-MRB, vol. 7 no. 1, 2025【已核对】 | 用累计方差定义任务复杂度指标 VAF-CI，能区分不同操作任务 |
| [MoDex](https://arxiv.org/pdf/2409.10983) | 【预印本】venue 未确认 | 附录比较"用筷子"与"转方块"所需 synergy 数，结论是随任务变化 |
| [VAE synergy + RL 抓取](https://arxiv.org/html/2602.20915v1) | 【预印本】2026 | 抓取成功率：VAE synergy 83%，PCA 71%，原始关节 66% |
| [LAMP](https://arxiv.org/pdf/2607.06323) | 【预印本】2026，内容未核对 | 潜在运动先验引导真机灵巧手学习 |
| [交互式模仿学习综述](https://arxiv.org/pdf/2506.00098) | 【预印本】2025 | 列举压缩灵巧动作空间的方法，指出 VAE 比 PCA 能学更复杂的映射 |
| Liu, Jiang, Yang, Liu, *Analysis of Hand and Wrist Postural Synergies in Tolerance Grasping of Various Objects*（[全文](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5007036/fullTextXML)） | PLoS ONE 11(8): e0161772, 2016【已核对，2026-11-13 读全文】 | 静态抓握（6 个物体 × 27 种手–物位置，保持 3 s）。手 15 DoF（CyberGlove III）+ 手腕 3 DoF（Polhemus）。PCA 合并和分开都做了：达到 >80% 方差，手+腕 18 DoF 需 6 个 PC，只有手 15 DoF 需 4 个，手腕 3 DoF 前 2 个就到 87%。"the wrist module had the lowest correlation among all inter-finger and inter-joint modules"，手腕 PC 在不同被试之间的一致性也低于手，所以作者主张手和手腕分开分析 |
| Jarque-Bou, Scano, Atzori, Müller, *Kinematic synergies of hand grasps: a comprehensive study on a large publicly available dataset*（[全文](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6540541/fullTextXML)） | J NeuroEng Rehabil 2019, doi 10.1186/s12984-019-0536-6【已核对，2026-11-13 读全文】 | NinaPro DB1/2/5，77 人，20 种抓握（伸手–抓–放，模仿视频，没有手内操作）。17 个角度，其中手腕 2 个（屈伸、桡尺偏），和手指放在同一个 PCA 里。手腕主要出现在 Synergy 2（"wrist flexion with palmar arch (CMC5) flexion"）。作者结论："the hand and wrist synergies need to be considered for a separate analysis"，因为手指协调不受物体位置影响，手腕协调会受影响 |

**对项目的启发**

- 你的 expert 动作谱是平的（95% 方差要 14/20 维），与 Santello 的"2 维解释 80%"相反。可以这样写：抓握是低维的，手内重定向不是。
- M2（单轴旋转）的关节位置维度从 15 降到 11，这是"维度随任务变"的第一个数据点；West 的 VAF-CI 和 MoDex 附录是最直接的对照。
- 手腕要不要并进 synergy：人手抓握文献（Liu 2016、Jarque-Bou 2019）主张分开，理由是手腕和手指相关性弱。没有找到在控制或意图通道上比较"手腕合并 PCA"（hand-k）和"手腕单独 + 手指 PCA"（wrist2-finger-k）的论文（2026-11-13 检索）。在 Shadow 专家动作上，手腕和手指的相关性并不弱，和手指之间的相关性相当，见 `Experiment Plan/results/wrist-finger-corr/`。
- 这一类文献几乎都只谈维度，不谈更新频率。你的 rate sweep 显示低于 20 Hz 时动作级接口的维度 k 已经不重要，这是它们没覆盖的轴。

---

## 二、高层 VLM/LLM + 低层灵巧控制器的接口

| 文献 | 出处 | 接口形式 |
|---|---|---|
| [DexterityGen](https://roboticsproceedings.org/rss21/p103.html) | RSS 2025【已核对】 | RL 预训练大量手内 primitive，蒸馏成一个底层控制器；上层给粗糙指令（目前是人遥操作） |
| [Scaffolding Dexterous Manipulation with VLMs](https://papers.nips.cc/paper_files/paper/2025/hash/862644b156e51c35dea5a7446d640b14-Abstract-Conference.html) | NeurIPS 2025【已核对】 | VLM 出 3D keypoint 和手腕轨迹；手指由 residual RL 学。Allegro 手，ManiSkill3 上 8 个任务 |
| [DexGraspVLA](https://arxiv.org/pdf/2502.20900) | AAAI 2026（检索显示 oral） | VLM 规划 + diffusion 控制器，杂乱场景抓取成功率 90% 以上 |
| [ZeroDex](https://arxiv.org/abs/2606.19340) | 【预印本】2026-06 | VLM 规划 + 可复用原子动作库（物体 6D 轨迹），Inspire 手。作者明确说不支持手内操作 |
| [RoboDexVLM](https://www.alphaxiv.org/abs/2503.01616) | 【预印本】2025 | VLM 任务规划 + Inspire 五指手 |
| [3D HAMSTER](https://arxiv.org/pdf/2606.31329) | 【预印本】2026 | 指出 2D 引导与 3D 低层策略不匹配，改为 VLM 直接出 3D 路点（非灵巧手） |
| [Hierarchical Planning with Latent World Models](https://www.researchgate.net/publication/403530027_Hierarchical_Planning_with_Latent_World_Models) | 【预印本】2026 | 高低层之间的 latent action 通常是人设计的，提出联合学习（非灵巧手） |
| Language to Rewards | CoRL 2023【凭记忆】 | LLM 通过 reward 间接控制 |
| Eureka | ICLR 2024【凭记忆】 | LLM 写 reward，训练出转笔等灵巧技能 |
| SayTap | CoRL 2023【凭记忆】 | LLM 输出足端接触模式，是"为 LLM 设计低维接口"的范例 |

**对项目的启发**

- 现有工作各选了一种接口（keypoint、手腕轨迹、粗指令、reward），没有人在同一个低层学习者上系统比较接口层级。你的 M1 和 rate sweep 做的正是这个比较。
- DexterityGen 的上层目前是人，换成慢速模型是公开的空位，和你的问题最接近。
- ZeroDex 不支持手内操作，是直接的切入口。
- 你的结果给这一类设计提供了一个判据：10 Hz 以下，目标级接口（goal-4）明显好于动作级（hand-8）和子目标级（obj-4）。

---

## 三、LLM 直接输出动作 / 前沿模型零样本控制

### 3.1 机械臂 + 夹爪

| 文献 | 出处 | 要点 |
|---|---|---|
| [LLMs as General Pattern Machines](https://proceedings.mlr.press/v229/mirchandani23a.html) | CoRL 2023【已核对】 | LLM 外推数值序列做低层控制；指出延迟、上下文长度、算力是瓶颈 |
| [Language Models as Zero-Shot Trajectory Generators](https://arxiv.org/pdf/2310.11604) | RA-L 2024【凭记忆】 | GPT-4 零样本输出末端位姿序列，夹爪，开环执行 |
| [FAEA: Demonstration-Free Robotic Control via LLM Agents](https://arxiv.org/html/2601.20334v1) | 【预印本】2026 | 粗操作任务上追平 100 条示范训练的模型；每个决策周期 2–8 秒，不能做实时反应控制 |
| [Frontier models on LIBERO-10](https://arxiv.org/pdf/2610.30051) | 【预印本】2026-10 | 六个前沿模型，最好 44% |
| [An Unexpected Robot Policy（Astra on RoboDojo）](https://arxiv.org/pdf/2609.24170) | 【预印本】2026-09 | 42 个任务 2100 次试验，平均成功率 22.48%；语义类任务好，精度、动态、双臂协调差 |
| [Astranomy](https://research.nvidia.com/labs/srl/projects/astranomy/)（NVIDIA） | 【预印本】2026 | 每次推理 11–19 秒（π0.5 是 47 毫秒），仿真在推理时暂停。直接输出末端位姿 72.7%，关节位置 48.7%，B 样条 37.3% |
| [Large Multimodal Models for Direct Robot Control](https://arxiv.org/abs/2610.28259v1) | 【预印本】2026-10 | 几何约束基准上零样本 11.1%，延迟高 |
| [Show-Harness](https://arxiv.org/pdf/2609.10522)（NUS Show Lab） | 【预印本】2026-09 | VLM agent 零样本控制机器人，只用了夹爪；灵巧手列为未验证的局限 |
| AA-Robotics v0.1（Artificial Analysis） | 【社交媒体/行业评测】2026-10 | 模型比人慢约 100 倍 |
| [ALRM](https://arxiv.org/pdf/2601.19510) | 【预印本】2026 | LLM 以写代码或工具调用两种方式控制机械臂 |

### 3.2 灵巧手（2026 年 9–10 月，GPT-6 Astra 发布之后）

| 文献 | 出处 | 要点 |
|---|---|---|
| [Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies](https://arxiv.org/html/2609.38537)（Galbot） | 【预印本】2026-09，只读了前一半 | **必读。** 模型直接下发 22 个（Sharpa）或 16 个（Allegro）关节指令，每次推理执行 1–5 个 20 Hz 步，对比逐步控制的专用 RL。见下表 |
| [Has GPT-6-Astra Achieved Embodied Intelligence?](https://arxiv.org/html/2610.22471v2) | 【预印本】2026-10 | 真机双臂 + Wuji 手抓玩具：无 primitive 时 0/3，有人写的 primitive 时 3/3。仿真 DexJoCo 87.3%，依赖运动参考和控制器 helper。人形机器人上只用手的二值开合 |
| [dexterous-astra](https://github.com/jianglongye/dexterous-astra) | 【预印本/GitHub】2026-09 | 转笔：零样本"手指关键帧规划器"转一圈，PPO 转三圈。魔方：零样本双手脚本控制器只做两步。零样本指模型写代码，不是实时闭环；无成功率 |
| [Awesome-Astra-Embodied-AI](https://github.com/zjwzcx/Awesome-Astra-Embodied-AI) | 【社交媒体汇总】 | 约 10 个灵巧手案例，全部在仿真；多数是模型写代码做重定向或训练 RL |
| X 帖"Stanford 用 Astra 一次解出魔方" | 【社交媒体】未找到论文 | 对照上面的仓库，实际只做了两步，属夸大 |
| Adafruit 博客：Claude 控制 16 关节腱驱动手 | 【社交媒体】2026-03，页面未能打开 | 只有手势，没有操作物体 |

Galbot 论文的手内操作结果：

| 任务 | Astra 直接控制 | 专用 RL |
|---|---|---|
| 手内旋转圆柱（在容差内的步数占比） | 0.51% | 76.9% |
| 手内旋转长方体 | 4.40% | 63.5% |
| 手内平移 | 1/5 | 4/5 |
| 平移 + 旋转 | 0/5 | 4/5 |
| 十任务抓放基准平均分 | Direct 16.6 | Hybrid 61.6；微调的 π0.5 为 44.2 |

作者结论：保持支撑比重新组织手指接触容易；物体不怎么掉，但转得太慢。

### 3.3 Fable 5.2（2026-11-09 发布）

- [Anthropic 发布页](https://www.anthropic.com/claude-fable-and-mythos-5-2)【厂商自测】：RoboDojo Sim（仿真双臂，从相机图像控制）Fable 5.2 64.9%，Opus 5.5 39.4%，Gemini 4 Argon 34.2%，GPT-6 Astra 29.8%，Fable 5.1 27.6%。全部是 Anthropic 用自己的 harness 内部跑的。
- 真机演示视频：解魔方、拧瓶盖、画机器人。
- [Humanoids Daily 报道](https://www.humanoidsdaily.com/news/anthropic-claude-fable-5-2-robotics-robodojo)列出未披露项：机器人型号和末端执行器类型、控制接口、延迟、是否加速、试验次数和成功率、64.9% 是成功率还是带部分得分的分数。
- **截至 11-11 没有任何 Fable 5.2 控制灵巧手的结果**，官方和社交媒体上都没有。系统卡 PDF 过大未能打开；HN 讨论帖只读了前四分之一。

**对项目的启发**

- 更正：最初说"没有灵巧手零样本结果"是错的。有，结论是关节级直接控制基本失败，primitive 级或二值开合可以。
- 现有结果只有两个端点。中间的"维度 × 频率"曲线，以及它如何随任务变化，是空的，rate sweep 正好填这段。
- 动作空间越接近关节级，零样本越差（Astranomy：72.7% → 48.7% → 37.3%），与你的"接口层级"结论同向。
- 延迟数字（2–19 秒每次决策）给你的更新频率轴提供了现实锚点：你的任务目标每 0.37 秒变一次，Astra 类模型每个 episode 发不出一个目标。
- Galbot 的手内旋转设置可直接当对照基线引用。
- Fable 5.2 在灵巧手上的水平是公开空白；你有现成的 Shadow Hand 环境和评估脚本，可以测，但这是新方向，尚未决定做。

---

## 四、VLA 中的手部动作压缩与快慢双系统

| 文献 | 出处 | 要点 |
|---|---|---|
| [XL-VLA: Cross-Hand Latent Representation](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Cross-Hand_Latent_Representation_for_Vision-Language-Action_Models_CVPR_2026_paper.html) | CVPR 2026【已核对】 | 各手专用编码器映射到共享 latent 动作空间，一个 latent 可解码到多种手 |
| UniDex | CVPR 2026（文件夹里已有） | 跨手通用控制；基于少量共享功能性 primitive（拇指食指捏合、手指卷曲）；数据含魔方任务 |
| [Towards High-DoF Dexterous Manipulation through VLA Post-Training](https://arxiv.org/html/2609.19666) | 【预印本】2026 | 时序手部 codec，每只手压到 9 维 latent；latent residual RL 修正冻结的 VLA |
| [UniDexTok](https://arxiv.org/pdf/2606.10683) | 【预印本】2026 | 跨手统一的离散 tokenizer |
| [GR-Dexter](https://arxiv.org/pdf/2512.24210) | 【预印本/技术报告】2025 | 21 自由度手接入 VLA |
| [Fast-in-Slow VLA](https://neurips.cc/virtual/2025/poster/119931) | NeurIPS 2025【已核对】 | 把快执行模块嵌进慢推理的 VLM 里，共享参数；117.7 Hz（动作块为 8，4090 上测，第三方摘要） |
| [Hume](https://arxiv.org/html/2505.21432v1) | 【预印本】2025 | System 2 慢推理 + System 1 快策略（90 Hz），级联去噪 |
| GR00T 2（NVIDIA，2026-11） | 【技术报告】 | 五指手机器人 Dexter：零样本 40%，每任务 1 条示范后 71%，5 条后 95% |

**对项目的启发**

- VLA 路线默认把手部动作压到约 9 维 latent 并以高频输出；你的结果说明，这类动作级 latent 一旦更新变慢就会失效，快慢双系统的"慢"层不应输出动作级信号。
- 双系统工作给出了"快层要多快"的工程答案（90–120 Hz），你的 rate sweep 给出"慢层最慢能多慢、该传什么"的实验答案，两者互补。
- GR00T 2 的"零样本 40% → 5 条示范 95%"是"哪些任务仍需训练"的一个量化例子。

---

## 五、Teacher–student 传什么

| 文献 | 出处 | 要点 |
|---|---|---|
| Qi et al., *In-Hand Object Rotation via Rapid Motor Adaptation* | CoRL 2022【凭记忆】 | 特权信息压成 8 维 latent |
| [PTLD](https://arxiv.org/pdf/2603.04531) | 【预印本】2026 | 蒸馏特权触觉 latent 而非动作 |
| [ArtManip](https://arxiv.org/pdf/2609.12498) | 【预印本】2026 | 学生预测教师的 latent，冻结的教师策略在预测 latent 上运行 |
| [铰接工具手内操作](https://arxiv.org/pdf/2509.23075) | 【预印本】2025 | 学生匹配 oracle 动作；on-policy DAgger 学生常早期掉物，改用稳定 oracle 的 rollout |
| [Dex-X](https://arxiv.org/pdf/2609.07747) | 【预印本】2026 | DAgger，64 维 PointNet 输出，29 维动作 |

**对项目的启发**

- 检索中没找到对 bottleneck 维度做系统扫描的工作。你的 M1 曲线（hand-2 0.50 → hand-4 0.71 → hand-8 0.87，hand-20 0.81）在这里有位置。
- 这一类的 latent 都是每步更新的。"教师信号可以多慢"没人问过。

---

## 六、技能组合：用已训练的基础技能零样本做新任务

按"技能怎么接进来"分四类。

| 接入方式 | 代表工作 | 零样本组合？ | 用灵巧手？ |
|---|---|---|---|
| 加载多个独立 policy，按阶段切换 | Sequential Dexterity、DexCompose | 否，需微调或训练残差 | 是 |
| Agent 把 policy / VLA 当工具调用 | Harness VLA、EBench、Pigey、ZeroDex | 是 | 多数是夹爪 |
| 一个通用底层控制器，上层给粗指令 | DexterityGen、UniDex | 上层可换，底层预训练 | 是 |
| Agent 自己写代码技能并存成库 | RegenHarness、dexterous-astra | 部分 | 少 |

| 文献 | 出处 | 要点 |
|---|---|---|
| [Sequential Dexterity](https://arxiv.org/abs/2309.00987)（Stanford） | CoRL 2023【已核对】 | 串联多个灵巧手 RL 策略；学"转移可行性函数"判断交接状态，并反向微调前一个策略 |
| [DexCompose](https://arxiv.org/abs/2606.28323)（UNC） | 【预印本】2026-06 | 单手"一边保持一边操作"：两个冻结策略，测出保持第一个技能所需的手指，其余手指做第二个技能。Shadow Hand 仿真，16 个组合任务平均 77.4%；只支持两个技能顺序组合 |
| [Harness VLA](https://arxiv.org/abs/2607.08448) | 【预印本】2026-07 | 前沿模型规划，只在接触密集阶段调用冻结的 VLA，带记忆模块 |
| [EBench profiling](https://arxiv.org/abs/2610.35832) | 【预印本】2026-10 | 模型调用训练过的 VLA 能追平最强单一策略，但达不到逐任务上限：调用时机不佳、抓取被打断 |
| [Pigey](https://arxiv.org/html/2607.21725v1) | 【预印本】2026-07 | 前沿 VLM 规划、选技能、验证、恢复；上限由底层工具决定，每步 API 调用有延迟 |
| [RegenHarness](https://arxiv.org/pdf/2609.27612) | 【预印本】2026-09，细节未核对 | agent 自我改进并积累技能 |
| [Guava](https://arxiv.org/html/2606.18363) | 【预印本】2026 | 把前沿 VLM 的工具调用蒸馏到 4B 小模型 |
| [LodeStar](https://arxiv.org/pdf/2508.17547) | 【预印本】2025 | 指出朴素串联常在交接点失败 |
| [Semantic handoff 诊断](https://arxiv.org/pdf/2607.06256) | 【预印本】2026-07 | BEHAVIOR-1K 上，单独能用的技能串起来会退化 |
| [DeCo](https://arxiv.org/html/2505.00527) | 【预印本】2025 | 任务分解 + 技能组合的零样本泛化基准（非灵巧手） |
| [VLBiMan](https://arxiv.org/html/2509.21723v4) | ICLR 2026【已核对】 | 一条示范拆成左右臂 primitive，VLM 适配新场景（夹爪级） |
| DexImit | RSS 2026（检索显示，内容未核对） | 从人类视频生成双手灵巧操作数据 |
| OpenAI, *Solving Rubik's Cube with a Robot Hand* | 2019【凭记忆】 | 解法序列由经典求解器给出，手部是一个端到端 RL 策略，不是组合已有技能 |
| [FINGR](https://arxiv.org/abs/2609.33973) | 【预印本】2026-09 | 真机单手转魔方层，300 次转动成功率 99.0%，专门训练 |

**对项目的启发**

- "会转方块 + 会抓 + 一手固定一手操作 → 零样本转魔方"这个完整例子还没人做成。
- 共同障碍是技能交接：上一个技能结束的状态落在下一个技能能处理的范围之外。所以上层要传的除了"调用哪个技能"，还有"在什么状态下交接"，这仍是接口问题。
- 你有两个冻结的 Shadow Hand expert（任意朝向重定向、绕 z 轴旋转）。已写进 Phase 2 计划的技能切换试点测的是：1 bit 的慢速选择器能否在二者间切换。已知弱点：yaw 目标是 Stage 1 目标的子集，所以它测的是"从没见过的状态接手"，不是互补技能的组合。

---

## 七、总体启发

1. **研究问题的重新表述。** 从"意图接口可以多低维"改为"维度 × 更新频率 × 上游能否生成"。LLM 通道的约束主要是慢，不只是窄。
2. **已有结果的定位。**
   - 平谱（95% 方差要 14 维）与维度随任务变（M2：15 → 11）对应第一类文献。
   - M1 的维度曲线对应第五类，那里没人做过系统扫描。
   - rate sweep 的排名反转对应第二、三类之间的空白。60 Hz 时 hand-8（0.87）与 goal-4（0.77）不可区分；10 Hz 以下 goal-4 全面领先；2 Hz 时 goal-4 0.43，其余不超过 0.02。
3. **"目标带宽"这个概念。** 这个任务的目标每 21.9 步（0.37 秒，2.7 Hz）变一次，3 个自由度。一个任务能否由慢速模型零样本驱动，可以用"每秒目标事件数 × 目标自由度"来刻画。这对应你最初说的 task-awareness。
4. **文献给的现实锚点。** 前沿模型每次决策 2–19 秒；关节级直接控制灵巧手在容差内的步数占 0.5–4%；给了 primitive 后真机抓取从 0/3 变成 3/3。
5. **仍然空着的位置。**
   - 灵巧手上，接口层级 × 更新频率的系统曲线（正在做）。
   - 同一曲线在第二个任务上的对比（Phase 2 第 1、3 块）。
   - 目标通道能容忍的延迟和精度（Phase 2 的 P1）。
   - Fable 5.2 在灵巧手上的任何结果。
   - DexterityGen 式底层控制器配慢速模型上层。
6. **关于"上层"是什么。** 实验里的慢速上层是用 hold 模拟的抽象通道，没有真实模型，结论因此不依赖具体模型。若接真实模型：目标以文字给出用 LLM 即可，要从画面读目标则需要 VLM；不必用 SOTA，放两三个不同档位的模型，在曲线上各标一个点更有说服力。
