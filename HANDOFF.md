# 低维意图接口实验交接

更新日期：2026-10-27（新加坡时间）。范围：实验设计、代码、已验证结果、数据位置与后续工作。

## 一句话状态

Shadow Hand 的 Isaac Lab 环境、seed-0 PPO expert 训练、best/final checkpoint 评估与冻结均已完成。2026-10-27 在 cluster 上核实：pilot rollout（job `1023495`、`1023529`）和 1,228,800 条 transition 的正式 PCA 数据集（job `1023537`）均已 `PASS`。PCA 基底和学生策略实验尚未开始。

## 研究目标与时间约束

- 任务：Isaac Lab 的 `Isaac-Repose-Cube-Shadow-Direct-v0`，用 Shadow Hand 在手内重定向 cube。
- 核心问题：高层意图传给低层控制器时，手部动作、手腕动作和物体子目标各需要多少维，怎样影响学习效率与性能恢复。
- 阶段一：goal 可见；比较 Blind、Wrist-2、Hand-2/4/20、Obj-1/4，重点看学习曲线 AUC 与达到 expert 性能 80% 的训练量 N80。
- 阶段二 M1：actor 隐藏 goal，比较意图维度与性能恢复；计划至少 3 个 seeds（`0, 1, 2`）。M2（第二任务）和 M3（人类 synergy 基底）原计划按时间二选一；用户 2026-10-27 决定暂不砍掉，视进度再定。
- 当前排期（用户 2026-10-27 确认）：**Project Milestone 2026-10-27**（当日尚未提交）、**Demo 2026-11-18**、**Final report 2026 年 11 月底**。取代此前记录的 10-20 / 11-22，以及 v3 原文中的“12 月 1 日那周”。

完整设计见 [低维意图接口实验设计 v3（分阶段）](Docs/Experiment%20Plan/低维意图接口实验设计%20v3（分阶段）.md)。集群脚本及操作说明见 [cluster_smoke/README.md](Experiment%20Plan/cluster_smoke/README.md)。

## 已验证的运行环境

| 项目 | 结果 |
| --- | --- |
| Isaac Lab | v2.3.1，commit `5c2ec81cb17532d32f7922dd7fcaae40d123b71a` |
| Isaac Sim | 5.1.0.0 |
| Python | 3.11.17 |
| PyTorch | 2.7.0+cu128 |
| RL-Games | 1.6.1 |
| 已验证 GPU | Tesla T4，15,360 MiB；driver 580.178.04 |
| 物理 / 控制频率 | 120 Hz / 60 Hz，`decimation=2` |
| 动作 / policy observation | 20-D / 157-D |
| Episode 上限 | 10 秒，约 600 个控制步 |

此前泛化 GPU 请求落到一个无法被 `nvidia-smi` 看到的节点，所以现有 GPU Slurm 脚本均约束 `xgpf`。Isaac Sim 模块需要在 `AppLauncher` 初始化后使用；CPU 节点上直接导入依赖 `omni` 的任务模块会失败。安装过程中曾有包版本冲突，最终 GPU smoke test 与 PPO 训练均通过；不要仅凭 `pip check` 的三条已知元数据冲突判定运行环境失效。

### 动作与观测约定

- Action 0、1 分别是 `robot0_WRJ1`、`robot0_WRJ0`；action 2–19 是 18 个手指驱动。完整顺序记录在 `collect_shadow_rollout.py` 的 `ACTION_NAMES`。以运行时 `actuated_dof_indices` 为准；配置文件中的关节名列表顺序与运行时 action 列顺序不同。
- 157-D observation 包含关节位置/速度、物体位置/姿态/速度、目标姿态、相对旋转、指尖状态和上一动作。精确切片见 `collect_shadow_rollout.py` 的 `OBSERVATION_SLICES`。
- `observations[:, 137:157]` 是上一动作；PCA 使用 `actions.npy` 中的当前 20-D expert 动作。
- Expert rollout 用 RL-Games PPO actor 的确定性输出（高斯策略均值），环境初始状态仍由 seed 决定。

## 已完成实验与结果

| 阶段 | Job / 结果 |
| --- | --- |
| GPU probe | `914541`；Tesla T4 可用，Ubuntu 24.04.5 / glibc 2.39 |
| 软件栈 finalize | `914875` 通过 |
| Shadow random-action smoke | `914893`；64 envs、200 steps、20-D action、157-D observation，`PASS` |
| PPO 短跑 | `914933`；512 envs、5 iterations、43.61 秒命令耗时，`PASS` |
| 运行时接口核对 | `914999`、`915009`；动作映射、频率、observation 字段已确认 |
| Expert 速度基准 | `915059`；2048 envs、20 iterations / 655,360 transitions，稳态总 FPS 约 30k |
| Seed-0 expert 正式训练 | `915072`；5000 iterations / 163,840,000 transitions，Slurm 1:34:11，峰值显存 3738 MiB，`COMPLETED 0:0` |
| Expert 冻结 | 2026-10-06；`checkpoints/expert-seed0/frozen/`，SHA-256 `b9b7128f151b44c8241603e0138d5110435e70e13fad902431a0bbe96f2074f9` |
| Pilot rollout ×2 | `1023495`、`1023529`；256 envs × 700 steps = 179,200，各 `COMPLETED 0:0`；两次汇总指标完全一致（确定性复现） |
| 正式 PCA rollout | `1023537`；512 envs × 2400 steps = 1,228,800，`COMPLETED 0:0`，用时 2:39，`rollout_collection_status=PASS`、`rollout_job_status=PASS`；2050 个完成 episode，平均 reward 7486.40，平均长度 592.7；数据 806 MB |

正式训练得到 best 和 final 两个 checkpoint。独立确定性评估各跑 1024 episodes：

| Checkpoint | 平均 episode reward ± 标准差 | 平均 consecutive successes | 平均 episode length |
| --- | ---: | ---: | ---: |
| best | 7497.23 ± 1052.85 | 14.74 | 未在对话中提供正式值 |
| final | 7447.82 ± 945.64 | 15.49 | 未在对话中提供正式值 |

两者平均 reward 相差约 0.66%，final 的 consecutive successes 高约 5.07%，reward 波动较低，因此选 **final** 作为 rollout expert；best 保留。该选择是既定实验决策，不代表已做配对显著性检验。评估指标来自 `infos.episode.consecutive_successes`。

## 数据位置和可复现入口

Cluster 持久目录采用 `~/software/isaaclab-stack` 和 `~/data/low-dim-intent/{runs,checkpoints,datasets,cache,metadata}`。临时 `/tmp` 是作业隔离空间，不保存长期产物。

- 冻结 expert 的规范路径：`~/data/low-dim-intent/checkpoints/expert-seed0/frozen/expert-seed0.pth`。`10_freeze_expert.sh` 会复制 checkpoint、生成 SHA-256、评估副本和 `manifest.json`，并将文件设为只读。对话没有保留实际冻结命令输出；用户后来报告 pilot 通过，仍建议在 cluster 上核实该目录和校验文件。
- 评估结果：`~/data/low-dim-intent/runs/expert-eval-job<job-id>/{best,final}.json`。正式评估 job ID 未在本对话中记录。
- Rollout 数据：`~/data/low-dim-intent/datasets/expert-rollout-seed0-job<job-id>/`，包含 `observations.npy`、`actions.npy`、`rewards.npy`、`dones.npy`、`env_ids.npy`、`episode_ids.npy`、`episode_steps.npy`、`manifest.json` 和 `SHA256SUMS`。
- 脚本位于本仓库 `Experiment Plan/cluster_smoke/`：`08_train_expert.sbatch`、`09_evaluate_expert.sbatch`、`evaluate_shadow_checkpoint.py`、`10_freeze_expert.sh`、`11_collect_expert_rollout.sbatch`、`collect_shadow_rollout.py`。更早的 `00`–`07` 脚本负责 probe、安装、smoke、接口检查和 benchmark。
- 本地 Git remote 是 HTTPS GitHub 仓库；Git 用于代码、配置和小型结果摘要。`.npy` 数据集、`.pth` checkpoint、完整日志与缓存保留在 cluster。创建本文时本地工作树仍有未提交的实验脚本和设计文档，不要假定它们已经推送。

### Rollout 数据格式

采集器按向量步、环境编号依次写入 NumPy memmap；同一行保存动作执行前的 `observation_t` 与 `action_t`，以及动作执行后的 reward / done。每条还有环境 ID、该环境内 episode ID 和 episode step。`manifest.json` 记录 seed、checkpoint SHA-256、shape/dtype、动作列顺序、observation 切片及汇总指标。数据写入 `.partial` 目录，完成后改名为正式目录；正式目录和 `rollout_collection_status=PASS` 才代表完整数据。

RL-Games 的批量推理需要 `player.has_batch_dimension = True`。早期评估器遗漏该设置，导致 `[256,157]` 被错误展平为 `1×40192`；现有评估与采集脚本已经包含修复和形状检查。

## 当前待办（按依赖顺序）

1. ~~**核实 cluster 状态**~~：已于 2026-10-27 完成，见上方结果表。冻结 checkpoint 的 `SHA256SUMS` 存在且与各 rollout manifest 的 `checkpoint_sha256` 一致。
2. ~~**确认正式 PCA 数据集**~~：job `1023537` 已完成，路径 `~/data/low-dim-intent/datasets/expert-rollout-seed0-job1023537/`。剩余一项：数据目录内的 `sha256sum -c SHA256SUMS` 尚未运行，应放在 PCA 的 Slurm 作业内执行，不在 login node 上读 800 MB。
3. **生成 PCA 基底**（已提交 Slurm `1024096`，脚本 `analyze_action_pca.py` + `12_action_pca.sbatch`，待验收）：读取正式数据集的 `actions.npy`，计算 20-D 均值 `μ`、全维主成分 `W_robot`、每维解释方差及累计方差曲线；额外对手指 18-D 动作计算基底供 `Wrist-2 + Finger-k` 使用。固定输入数据集路径、checkpoint hash、列顺序、PCA 符号约定和输出文件校验值。当前仓库没有 PCA 分析脚本或输出。
4. **实现 intent 与学生实验**：Hand-k、Wrist-2、Obj-1/4，先 shape-level dry run，再固定统一的训练预算、评估方式和 seeds `{0,1,2}`。阶段一 goal 可见，阶段二 M1 隐藏 actor 的 goal。当前对话没有这些模块或训练结果。
5. **更新排期与仓库**：v3 文首已加排期说明；路线图关口仍需按 11-18 demo / 11 月底 report 重新规划；将实验脚本、配置及小型结果摘要提交 Git。不要把 checkpoint、NumPy 大数组、缓存和完整日志加入 Git。

## 交接时需要避免的误判

- 训练日志中的 best reward `6877.17` 与 final reward `6691.57` 是训练时的记录；正式 checkpoint 选择依据是独立的 1024-episode 评估。
- `consecutive_successes` 是任务指标，评估器从 `infos.episode.consecutive_successes` 提取；它不是逐 episode 二值成功率。
- pilot 的 `179,200` 条 transition 是管线验证，不能替代约 `1,228,800` 条的正式 PCA 数据集。
- pilot 的平均 consecutive successes 为 3.38，正式数据集为 16.25，但两者 episode reward 几乎相同。该指标疑似跨 reset 累积的滑动平均，短 rollout 读数偏低；用作学生策略指标前须确认定义。
- 正式 PCA 数据集已完成（job `1023537`）；PCA 基底和学生策略实验尚无产物。
