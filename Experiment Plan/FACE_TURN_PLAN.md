# 迭代 6 计划：face-turn 第三任务与两技能组合（2026-11-16 → 11-22）

写于 2026-11-16 21:10。状态：**E1 的新定义 1–5 已由用户确认（2026-11-17），E1 已提交**；E0 和 E2 已被 11-16/11-17 的实际工作取代（两个冻结 expert、切换测试、handoff 续训、unified expert，见 CONTEXT.md）。原状态：草案，待用户确认。标了「待确认」的条目是科学定义上的新选择，确认前不提交对应作业。每一步完成后在这里打勾并写上 job ID，和 `M1_PLAN.md` / `RATE_SWEEP_PLAN.md` 的做法一致。

## 约束

- 报告 2026-11-22 提交，5–8 页（不含参考文献），NeurIPS 格式。
- **实验截止：11-19 24:00。** 顺序（用户 11-18）：先 E2'，再 E3。 之后不再提交训练作业，只允许重画图和补评估。
- 集群：最多 8 张 GPU（6 训练 + 2 评估），最多 32 个排队作业，只用 `xgpf`；`gpu` 分区单个作业上限 3 小时。
- 单次耗时（实测）：face-turn expert 5000 iterations 约 2.1 小时；非对称 critic 的学生约慢 15%，估计 2.4 小时（未实测）。
- 迭代 2–5 的结果已经定稿，报告这几部分从 11-17 起并行写，不等本计划。

## 要回答的问题

1. **Q1（复现）**：「60 Hz 时动作级接口和目标级接口相当，变慢后只有目标级还能用」在第三个任务上是否成立？face-turn 的目标是离散的（每次 ±90°），和前两个任务的连续姿态目标不同。
2. **Q2（等待时间模型）**：目标通道的延迟损失是否仍然等于「每个目标多等 d 步」？用 face-turn 自己的 T0 做样本外预测。
3. **Q3（组合）**：一个慢速选择器在「重定向」和「转面」两个冻结技能之间切换时，损失是只有等待时间，还是还有交接损失（一个技能从另一个技能的结束状态接手失败）？

Q3 是这次迭代的新内容；Q1、Q2 复用已有管线，主要花 GPU 时间。

## 沿用不变

- 铰接 cube：扭矩上限 0.05 N·m，90° 卡位，`FACE_TURN_HOLD_STEPS=10`。
- PPO 配置、2048 envs、60 Hz 控制、600 步 episode、seeds 0–2。
- 学生设定同 M1：actor 看不到目标字段 `[64:72]`，critic 非对称、看完整 157 维。
- 评估：全部 checkpoint × 1024 envs，seed 20261028，每个 episode 的成功次数。
- Hold-N 与延迟 d 的定义同 `RATE_SWEEP_PLAN.md`。

## 新定义（待确认）

1. **学生的状态里加入铰链角。** face-turn 的 157 维观测里，铰链角只通过 `[68:72]`（铰链误差）出现；按 M1 的做法遮掉 `[64:72]` 后，actor 既看不到目标也看不到铰链角。Task 1 / Task 2 遮掉目标后物体姿态仍可见，为了保持同样的「状态可见、目标隐藏」，face-turn 的 actor 输入 = 149 维 + (sin θ, cos θ) + 意图，θ 为当前铰链角。Blind 也有这 2 维。
2. **目标级意图 = `goal-yaw`（2 维）。** face-turn 的 `goal_rot` 是绕 z 的纯旋转，现有 `goal-yaw` 公式直接适用，不新增意图类型。
3. **动作级意图 = `hand-8`。** 基底来自 face-turn expert 自己的 rollout（512 × 2400，seed 20261007，和前两个任务相同）上的动作 PCA。
4. **参考值。** P_E = 冻结 face-turn expert 在评估器下的每 episode 成功次数；P_B = face-turn Blind 学生的终值。P_B 预计不为 0（盲转也能碰上目标），所以和 Task 2 一样同时报告原始成功次数。
5. **训练预算。** 学生先跑 5000 iterations（单个作业能放进 3 小时）。如果冻结的 expert 是 10000 iterations 的 checkpoint，并且第一轮读数时学习曲线仍在上升，就用已验收的续训机制接到 10000；否则停在 5000，并在报告里注明。
6. **组合任务 flip-turn（只评估，不训练）。** 同一个铰接 cube 上目标交替出现：
   - R：把 cube 转到「铰链轴朝上 + 随机偏航角」，成功判据同 Task 1（底部姿态误差 ≤ 0.1 rad）；
   - F：顶层转 ±90° 并保持 10 步，判据同 face-turn。
   - R 目标限制为轴朝上，是因为 face-turn expert 只在顶层背离手掌的姿态下训练过。
   - 控制器都是冻结的、目标可见的 expert：R 用 Stage 1 expert（`expert-seed0`），F 用 face-turn expert；各自拿到自己任务格式的观测。
   - 选择器是 oracle：上游消息 =（技能 1 bit，当前目标）。消息未更新时，下层继续用旧技能和旧目标。
7. **交接损失。** 交接损失(类型) = 混合任务里该类型每个目标的平均步数 − 该技能单独运行时的平均步数。单独运行的基准：F 用 face-turn expert 的评估；R 用 Stage 1 expert 在「轴朝上 + 随机偏航」目标上的单独评估。
8. **停滞率。** 评估输出里新增：成功次数少于 5 的 episode 占比。视频里 16 个环境有 2 个停滞，需要一个 1024 episode 上的数字。

## 步骤

### E0 冻结 face-turn expert（11-16 晚 → 11-17 上午）

- [ ] 读 `1151347`：确认跑了 5001–10000，`expert_train_status=PASS`。
- [ ] `30_eval_face_turn.sbatch` 评估 10000 的 final + best（hold 10，256 envs，1024 episodes），加上停滞率；同一作业里重评 `ep_5000` 以得到它的停滞率。
- [ ] 冻结规则（事先定好）：10000 的 checkpoint 同时满足「保持转动次数 ≥ 27.7」「停滞率不高于 `ep_5000`」「平均倾角 ≤ 21°」就冻结它，否则冻结 `ep_5000`。
- [ ] 录一段视频（16 envs，中位数环境），冻结到 `checkpoints/expert-faceturn-seed0/frozen/`，写 SHA-256 和 manifest。
- [ ] 用 `17_evaluate_students.sbatch` 的协议评估冻结 expert 的整条曲线，得到 P_E 和 T0 = 600 / P_E。

### E1 face-turn 学生网格（代码 11-17，训练 11-17 晚 → 11-18）

- [ ] 代码：
  - `16` / `15` / `17` 增加 `STUDENT_STAGE=faceturn-hidden`（任务、expert、`HINGE_*` 变量）；
  - `train_student.py`、`evaluate_students.py`、`collect_shadow_rollout.py` 导入 `hinged_cube_tasks`；
  - wrapper 增加铰链角 2 维（新定义 1），写入 `intent.json`，评估器读取；
  - CPU 测试：铰链角公式、遮罩后维度、`goal-yaw` 在 face-turn rollout 上重构 `goal_rot`。
- [ ] rollout → 动作 PCA → `validate_intent_modules.py`（重建的 expert 与记录动作一致）。
- [ ] GPU dry run（256 envs × 3 iterations）：全部 7 个（意图，N）组合，检查 actor 维度 151 + k、critic 157。
- [ ] 网格 21 runs：`blind`；`goal-yaw` × N = 1, 6, 30；`hand-8` × N = 1, 6, 30；seeds 0–2。
  - 提交顺序：blind → goal-yaw → hand-8，这样先出目标级结果。
  - 21 × 2.4 小时 ÷ 6 张卡 ≈ 9 小时；评估用另外 2 张卡，`aftercorr` 在训练还在排队时提交。
- [ ] 只评估的延迟：`goal-yaw` N = 1 的终点 checkpoint，d = 0, 6, 30, 60（Q2）。
- [ ] 分析脚本：原始成功次数和 Recovery 对刷新率；`goal-yaw` 对 `hand-8` 的 Welch 检验；延迟结果对 600 / (T0 + d) 的逐 seed 误差。输出到 `results/faceturn-rate-sweep/`。

### E2 组合试点（代码 11-17 → 11-18，评估 11-18 → 11-19）

- [ ] 环境 `ShadowHandFlipTurnEnv`（`hinged_cube_env.py`）和选择器包装；CPU 测试目标交替、两种观测格式、hold / delay 缓冲。
- [ ] 基准评估：Stage 1 expert 在「轴朝上 + 随机偏航」目标上单独运行（得到 T_R）。
- [ ] **关口 G-E2**：oracle 选择器 60 Hz，1024 episodes。
  - 通过：每 episode 完成的目标数 ≥ 0.7 × 600 / ((T_R + T_F) / 2)，并且停滞率 ≤ 20%。
  - 不通过：不再调参，把交接失败本身作为结果报告（哪个方向的交接失败、失败时的状态分布、一段视频），并跳过下面的扫描。是否加一次「从混合任务状态出发微调 face-turn expert」的训练（约 2 小时）由用户决定。
- [ ] 扫描（只评估，每个条件约 1 分钟）：选择器 hold N = 6, 30；延迟 d = 6, 30, 60；另加「只用 R」和「只用 F」两个对照。
- [ ] 分析：每类目标的步数、交接损失、跌落率、停滞率；实测对等待时间预测 600 / (T + 等待)。输出到 `results/flipturn-composition/`，加一段 2×1 视频。

### E2' 选择信号速率扫描（用户确认做，2026-11-18 09:10；替代上面的 E2）

系统 = 已有的双专家交接（Stage 1 `expert-seed0` + 交接版转动专家 `477f3d43…`，60 Hz 零延迟基准 `1153366`：`up,turn` 7.51 循环 / 600 步）。不新建 flip-turn 环境，直接在 `ShadowHandSwitchEnv` + `evaluate_switch.py` 上做。

- 语义（新定义 6）：上游消息 =（技能 bit，当前目标）。环境内部照常换目标、照常判成功；**策略看到的**技能 bit 和目标是消息里的版本。消息没更新时，下层继续用旧技能、旧目标（刚完成的目标 → 原地保持）。
  - hold N：消息每 N 步刷新一次（零阶保持）；delay d：策略看到的是 d 步前的消息。定义同 `RATE_SWEEP_PLAN.md`。
  - 实现：`evaluate_switch.py --selector_hold N --selector_delay d`；环境记录每步的（kind, goal_rot, hinge_goal），观测里的目标字段 `[64:72]` 用消息版本重算（turn 格式的 `[68:72]` = hinge_rot ⊗ goal⁻¹），专家按消息里的 bit 选。默认 N = 1、d = 0 时与现有评估逐位一致（CPU/GPU 回归测试：对 `1153366` 的 `up-turn.json`）。
- 条件：N = 1（基准，重跑核对）、6、30；d = 6、30、60；循环 `up,turn` 和 `any,up,turn`；对照：只用 Stage 1 专家、只用转动专家（N = 1）。1024 envs、600 步、seed 20261006。
- 分析：循环数 / 600 步对等待时间预测 600 / (T + 等待)（等待 = d，或 hold 的平均 (N−1)/2）；分阶段步数、交接后的跌落和停滞。输出 `results/e2-selector-sweep/`。
- [x] 代码 + CPU 测试 → [x] 回归（N=1, d=0 对 `1153366`；`1160295` 不一致，已修；`1160401` 逐位一致）→ [x] 扫描 → [x] 分析（`results/e2-selector-sweep/`）
- 2026-11-18 ~09:30 代码完成（未提交 git，已同步集群）：`intent_modules.selector_view`（消息 = (kind, goal_rot) 走 `IntentChannel`，按消息 kind 重算 `[64:72]`）；`evaluate_switch.py --selector_hold/--selector_delay/--only_expert`，结果 JSON 多 `selector`（kind/goal 过时的 env-step 比例，直通时 `passthrough_max_abs_goal_field_diff` 应为 0）；`31` 条件写法 `cycle@h6` / `@d30` / `@only-reorient` / `@only-turn`。CPU 测试 `test_selector_view.py` 8/8 PASS（本地）。
- **提交 `1160295`**（xgpf6，09:28 开始运行；日志 `logs/e2-selector-sweep-1160295.out`，输出 `runs/switch-eval-job1160295/`）：16 个条件，前两个 = 回归 `up,turn` / `any,up,turn`（N=1, d=0，必须和 `1153366` 的 `up-turn.json` / `any-up-turn.json` 一致，cycles 7.510 / 5.48），然后 {up,turn, any,up,turn} × h6, h30, d6, d30, d60，以及两个循环 × only-reorient / only-turn。转动专家 = 冻结交接版 `477f3d43…`；cap 0.05，hold 10，1024 envs，600 步，seed 20261006。回归不一致 → 扫描结果作废。
- **`1160295` COMPLETED 0:0（22 min，16/16 PASS），回归 FAIL**：N=1/d=0 `up,turn` 7.603 vs `1153366` 7.510，`any,up,turn` 5.356 vs 5.482（各项都不同）。原因：重算的目标字段与环境原值差 ≤1.5e-7（`passthrough_max_abs_goal_field_diff`），确定性专家把它放大。**修复**：消息与当前 (kind, goal) 相同的 env 保留环境原字段，只替换过时的 env（`fraction_env_steps_fields_replaced`，直通时应为 0）；only-expert 对照同理（布局相同的阶段保留原字段）。`evaluate_switch.py` sha256 `0d2ac2f2…`，已同步。结果作废但已拉回 `results/switch-eval-job1160295/`（仅供参考：额外步数/阶段 ≈ 等待 + 0–5 步，落在等待时间线上）。
- **重交 `1160401`**（10:00，xgpf6，同样 16 个条件）。判据：两个回归条件与 `1153366` 逐位一致；若仍不一致而 `fields_replaced`=0 → 跨运行/节点不确定性，改用本 job 自己的 N=1 作基准并在报告注明。
- 分析脚本 `cluster_smoke/analyze_e2_selector_sweep.py`（本地运行；`--run_dir results/switch-eval-job<J> --baseline_dir results/switch-eval-job1153366 --output_dir results/e2-selector-sweep/`）：回归核对、等待时间线 600/(T0+L·w)、额外步数/阶段对 w、过时比例实测对预测、各类阶段中位步数和停滞、图 `e2_selector_sweep.png`。在 `1160295` 上测试通过（status 按回归 = FAIL）。

### E3 v4 统一专家的学生（用户要求做，2026-11-18 09:10；定义 a–d 用户确认按草案，2026-11-18 09:20）

老师 = 冻结 v4 seed 0（`expert-unified-cap0.05_hold10_v4-seed0/frozen/expert-unified-v4-seed0.pth`，`ca3d9367…`）。设定照 E1 / M1：actor 看不到目标，critic 非对称、看完整 162 维。

- **已确认 a：actor 遮掉哪些维。** 遮 `[64:72]`（目标四元数 + 姿态差）以及 v4 新增维里由目标导出的 `sin/cos(hinge_error)`、`is_turn`、`hold_count`；保留 `hinge_vel`，并像 E1 一样加当前铰链角 (sin θ, cos θ)。actor = 149 + 2 (铰链角) + 1 (hinge_vel) + k。
- **已确认 b：目标级意图。** `goal-6` = 目标四元数 (4) + 铰链目标 (sin, cos)，再加 `is_turn` bit，共 7 维（用户选「带 bit」）。（turn 阶段四元数不变，只有铰链目标变；没有 bit 时学生要从铰链目标的变化推断阶段。）
- **已确认 c：动作级意图** = `hand-8`，基底来自 v4 自己的 rollout（512 × 2400，seed 20261007）上的动作 PCA。
- **已确认 d：网格。** blind；目标级 × N = 1, 6；`hand-8` × N = 1, 6；seeds 0–2 → 15 runs × 约 2.5 h（T4）。训练环境 = v4 训练环境（50% 交接起点、120 步超时终止）；5000 iterations。
- 评估：`evaluate_switch.py` 加学生路径（意图包装器），`up,turn` / `any,up,turn` 的循环数，和老师 v4（18.9 / 10.5）、双专家交接比。
- 代码：`intent_env_wrapper.py` / `intent_modules.py` 支持 162 维；`11` 加 `ROLLOUT_TASK=unified`；`14` 校验；`15`/`16` 加 `STUDENT_STAGE=unified-hidden`；`evaluate_switch.py` 学生路径。
- [x] 定义确认（09:20）→ [x] 代码 + CPU 测试 → [x] rollout + PCA + 校验 → [x] dry run → [x] 评估侧（`1160722` PASS）→ [~] 网格已提交 `1160744` / `1160763` → [ ] 评估结果 → [ ] 分析
- 2026-11-18 训练侧代码（已同步集群，未提交 git）：
  - `intent_modules.py`：v4 观测布局常量；扩展观测 164 维 = 162 维观测 + 当前铰链角 (sin, cos)（包装器从环境读，162 维里只有铰链误差）；`goal-7` = 规范化目标四元数 + 铰链目标 (sin, cos)（= 铰链角 − 铰链误差）+ is_turn；`unified_actor_obs` = 152 维（去掉 [64:72]、[157:161]，保留 hinge_vel，加铰链角）；`FrozenExpert(obs_dim=162)`，老师只吃前 162 维；统一任务上只允许 blind / hand-K / goal-7。
  - `intent_env_wrapper.py` `unified=True`：actor 152 + k，critic = 裁剪后的 162 维；前 50 次观测把重建的铰链目标和 is_turn 对环境自己的值核对（误差 > 2e-3 直接报错）。
  - `train_student.py`：统一任务自动识别，intent.json 记录全部 UNIFIED_* 设置和起点状态 sha256。
  - `unified_env.sh`：v4 训练环境的全部设置（同 `1159044`；起点状态 sha256 不符就退出），`11`/`15`/`16` 共用；`STUDENT_STAGE=unified-hidden`，`ROLLOUT_TASK=unified`；`14` 加 `CHECK_UNIFIED=1`（`validate_intent_modules.py --unified`）；`15` 加 `dry_run_timing`（稳态 fps → 5000 轮预计用时）。
  - 采集器按环境的观测宽度写数据，统一任务另存 `hinge_angle/hinge_goal/phase_kind.npy`。
  - 本地 CPU 测试 `test_unified_intents.py` 11/11 PASS（含假老师 + 假 rollout 上跑通 `--unified` 校验），`test_selector_view.py` 8/8，Stage 1 路径不变（157 维老师照常，162 维老师在 Stage 1 路径被拒）。
- **提交（2026-11-18，作业链 afterok）**：rollout `1160549`（512 × 2400，seed 20261007，v4 训练环境）→ PCA `1160550`（`datasets/action-pca-expert-unified-v4-rollout-seed0-job1160549-job1160550/`）→ 校验 `1160551` → T4 dry run `1160552`（`unified-hidden`，2048 envs × 60 轮，blind / goal-7 / hand-8 × hold 1、6；`--partition=gpu,gpu-long --time=01:30:00`）。判据：校验 0 项失败；dry run 6/6 PASS；`dry_run_timing` 的 5000 轮预计用时 + 评估 ≤ 约 2.7 h，否则训练和评估拆成两个作业。
- 注意：v4 老师是 8192 envs 训练的（2000 轮左右才起飞），学生按用户要求用 2048 envs，每轮样本是老师的 1/4，5000 轮内可能学不起来。
- **作业链结果（2026-11-18，全部 COMPLETED）**：
  - rollout `1160549` PASS（xgpf11，3 min；1,228,800 步，2051 episodes，平均 episode reward 10528）。数据集 `datasets/expert-unified-v4-rollout-seed0-job1160549/`。
  - PCA `1160550` PASS：20 维累计 k = 2 / 4 / 8 为 26.5% / 44.8% / 69.8%（Stage 1 35.8 / 56.8 / 81.1，face-turn 45.0 / 68.0 / 87.8）；手指 18 维 k8 73.8%。
  - 校验 `1160551` PASS，0 项失败。
  - T4 dry run `1160552` PASS 6/6（xgpf11，17 min）：actor 152 / 159 / 160，critic 162，起点状态 sha256 对上，`unified_goal_check` 6 个 run 最大误差 3.4e-7。稳态 17.1k–17.7k fps → 5000 轮预计 **2.58–2.66 h**（只算训练）。加评估超过 2.7 h 判据 → 训练和评估拆成两个作业；训练离 16 号脚本内部 10500 s 超时只有约 9% 余量，超时的 run 用 `STUDENT_RESUME_FROM` 从最后一个 checkpoint 续训。
- **评估侧完成 + 网格提交（2026-11-18）**：`evaluate_switch.py --student_run_dir [--student_checkpoint]`（按 intent.json 用训练时的包装器重建学生，unified=True；评估协议同老师 `1159062`：相位超时关、不加载起点状态、固定 UNIFIED_CYCLE；核对扭矩上限 / hold 步数 / turn 容差 / 观测宽度与训练一致；结果 JSON 有 `student` 块）；`31` 加 `STUDENT_RUN_DIR` / `STUDENT_RUN_DIRS`（数组按 task id 取，输出 `runs/switch-eval-job<J>_<i>/`）。sha256 evaluate_switch `6ffb4152…`，31 `3113f8a2…`，已同步，未提交 git。测试 `1160722`（dry run 的 goal-7 hold6 / hand-8 hold1 学生，`up,turn` / `any,up,turn`）2/2 PASS，`unified_goal_check` 最大误差 1.5e-7，循环 ≈ 0（60 轮，符合预期）。
  - **网格**：训练 `1160744_0-14`（`16`，`STUDENT_STAGE=unified-hidden`，`blind:1 goal-7:1 goal-7:6 hand-8:1 hand-8:6` × seeds 0–2，任务 i → 意图 i // 3，seed i % 3，5000 轮，2048 envs，PCA `…-job1160549-job1160550`）；评估 `1160763_0-14`（`31`，`aftercorr`，条件 `random any up up,turn any,up,turn`）。清单 `~/cluster_smoke/e3_grid_jobs.txt`。第一次评估数组 `1160745` 路径错（本地打印计划时 `SLURM_SUBMIT_DIR` 未设，`unified_env.sh` 找不到 → run 名为空），已取消，未运行。
  - 判据：每个训练 `student_job_status=PASS`（超时 → `STUDENT_RESUME_FROM` 续训，再单独评估）；评估 `switch_eval_status=PASS`；和老师 v4（`up,turn` 18.9，`any,up,turn` 10.5）及双专家交接（7.51 / 5.48）比。
- **网格结果（读取 2026-11-18）**：训练 `1160744_0-14` 15/15 COMPLETED、`student_job_status=PASS`（5000 轮，9199–9771 s，没有超时）；评估 `1160763_0-14` 15/15 COMPLETED，每个 5/5 条件 `switch_evaluation_status=PASS`，`unified_goal_check` ≤ 3e-7。JSON 在 `results/e3-unified-students-job1160763/task<i>/`，`summary_up_turn.json`。
  - 每 600 步循环数（`up,turn` / `any,up,turn`；seeds 0/1/2）：blind 0.01 / 0.00；goal-7 5.55, 6.10, 6.96 / 0.01；goal-7@h6 3.28, 6.46, 3.16 / ≤0.02；hand-8 13.79, 13.73, 1.52 / 7.46, 7.28, 1.58；hand-8@h6 0.08, 0.75, 0.24 / ≤0.01。老师 v4 18.9 / 10.5，双专家交接 7.51 / 5.48。
  - Welch（`up,turn`，n = 3，不校正）：hand-8 vs goal-7 p = 0.48；hand-8@h6 vs goal-7@h6 0.36 vs 4.30，p = 0.063；goal-7 vs blind p = 0.004。
  - **goal-7 学不会 `any` 目标**：3 个 seed 都是 `any` 约 0.01 个 / 600 步（老师 29.4，hand-8 16–20），`any` 阶段约 50% 停滞，几乎不跌落；`up`（11–14）和 turn（5.3–6.7 次 / 600 步，停滞 ≤1%）学会了。所以 `any,up,turn` 卡在第一个阶段。原因没查。
  - hand-8 seed 2 的 turn 停滞 49%（`up,turn` 1.52），但 `any`/`up` 正常 → 又是种子差异。hand-8@h6 在所有条件下都接近 0，goal-7@h6 保留 `up,turn` 的约 70%（seeds 1 / 0、2 分化）。
- **goal-7 `any` 诊断（用户批准，2026-11-18）**：trace（`1160763_3/9` 的 `any-trace.npz`）显示 goal-7 seed 0 在 `any` 阶段姿态误差 141° → 100 步后 136°（倾斜、绕 z 都停在约 90°），120 步内到过 <10° 的 env 1.1%；hand-8 seed 0：138° → 35°，98.9%。提交 `1163578_0-2`（`31`，goal-7 hold 1 seeds 0–2，`STUDENT_EVAL_ITERATIONS="1000 2000 3000 4000 5000"`，条件 `any up`）：(1) 新增 `goal7_intent_check`：actor 实际收到的 7 维意图 vs 环境当前 (规范化 goal_rot, sin/cos hinge_goal, is_turn)，按阶段类型记最大误差；(2) 中间 checkpoint 的 `any` / `up` 完成数。代码：`evaluate_switch.py --student_iteration`、goal-7 核对（sha256 `3c008d3a…`），`31` `STUDENT_EVAL_ITERATIONS`（`8350882b…`），已同步，未提交 git。输出 `runs/switch-eval-job1163578_<t>/{any,up}-it<N>.json`。
  - **结果（`1163578_0-2` COMPLETED，13–14 min，3/3 PASS；本地 `results/e3-goal7-any-diag-job1163578/`）**：(1) 不是 bug：30 次评估里，actor 收到的四元数和规范化 `goal_rot` 误差恰好为 0，整个 7 维 ≤ 2.7e-7，误差 >1e-3 的步数为 0（`any`、`up` 都一样）。(2) `any` 从来没学会：3 个 seed 在 1000–5000 轮每 600 步都只完成 0.001–0.018 个，`any` 阶段 50% 停滞。`up` 在 1000 轮约为 0，2000 轮 1.8 / 9.4 / 12.3，3000 轮之后平台在 8.5–14.5。it-5000 的结果和 `1160763` 一致。结论：信息是对的，学生只学会了绕 z 的目标，没学会翻转；原因应在训练分布或预算（`any` 约占 1/4 阶段、120 步超时终止、2048 envs），没有直接验证。
- **E3 分析（2026-11-19，报告写法用户确认：主比较 `up,turn`，`any` 失败单独作为观察）**：`cluster_smoke/analyze_e3_unified_students.py`（本地运行：`--grid_dir results/e3-unified-students-job1160763 --diag_dir results/e3-goal7-any-diag-job1163578 --teacher_dir results/switch-eval-job1159062 --handoff_dir results/switch-eval-job1153366 --output_dir results/e3-unified-students-analysis`），`e3_unified_students_analysis_status=PASS`。输出 json / 每个 run 的 csv / goal-7 诊断 csv / 图 `e3_unified_students.png`（左：`up,turn` 循环数随意图刷新率，种子为点；右：goal-7 训练过程中 `up` 与 `any` 的完成数）。核对：15 个 run × 5 个条件齐全、评估协议相同、hand-8 都用 v4 老师 `ca3d9367…`（blind / goal-7 不加载老师）、诊断 goal-7 意图误差为 0、诊断的 5000 轮与网格一致。
  - 数字：`up,turn` 每 600 步 goal-7 6.20 ± 0.71 → hold 6 4.30 ± 1.88（保留 0.69）；hand-8 9.68 ± 7.07 → 0.36 ± 0.35（保留 0.04）；blind 0.01；老师 18.9，交接 7.51。Welch：hold 1 hand-8 vs goal-7 p = 0.48；hold 6 p = 0.063；goal-7 vs blind p = 0.004。
  - 报告写法草稿：统一任务上 hold 1 两种意图差不多（hand-8 的一个 seed 没学会转），hold 6 时 hand-8 跌到接近 0、goal-7 保留约 70%，方向与前三个任务一致，但 n = 3 只是趋势（p = 0.063）。观察：goal-7 学生没有学会 `any`（三维翻转）目标，1000–5000 轮都约 0，`up`（绕 z）在 2000–3000 轮学会；意图与环境目标逐位一致，已排除实现错误；推测原因是训练分布 / 预算（`any` 约占 1/4 阶段、120 步超时终止、2048 envs），未验证。所以 `any,up,turn` 不作为主指标。
- **`any` 失败原因诊断 V1 / V2（用户批准，2026-11-19；诊断，不进主比较）**：
  - 零成本对照：M1 goal-4（2048 envs，全部是 `any` 类目标）在 1000 / 1400 / 2000 轮成功 3.0, 0.6, 0.3 / 9.8, 5.1, 2.7 / 14.5, 10.3, 8.5 次 / episode；goal-7 每轮的 `any` 练习约为 M1 的 1/4，5000 轮 ≈ M1 1250 轮，按此应有 1–7 次，实际约 0.01，且 1000–5000 轮不涨 → 单靠「比例 + 预算」解释不了，续训到 10000 不做（推断粗略，学习不一定随练习量线性）。
  - 假设：120 步阶段超时终止让早期学生碰不到 `any` 的成功奖励（M1 原版没有单目标时限）。
  - V1 `E3_DIAG=notimeout`：`UNIFIED_MAX_PHASE_STEPS=0`，其他同 v4 训练环境。V2 `E3_DIAG=anyonly`：`UNIFIED_CYCLE=any`（固定顺序不加载起点状态），超时保留。都是 goal-7 hold 1，seeds 0–1，3000 轮，2048 envs。
  - 代码：`unified_env.sh` 的 `E3_DIAG` 覆盖、`train_student.py` 只在 anyonly 时允许固定顺序并在 intent.json 记 `e3_diag`、`16` run 名加 `-diag<name>`（sha256 `14fb7870…` / `c8c8b44b…` / `69f5766d…`，已同步，未提交 git）。默认（不设 E3_DIAG）路径不变。
  - 提交：V1 训练 `1164588_0-1` → 评估 `1164589_0-1`；V2 训练 `1164590_0-1` → 评估 `1164591_0-1`（`31`，aftercorr，条件 `any up up,turn random`，评估协议同网格）。清单 `~/cluster_smoke/e3_diag_jobs.txt`。
  - 预先定的判据：最后一个 checkpoint 的 `any` 条件每 600 步 ≥ 1 个 = 学会（基线 0.01）。V1 学会 → 超时是原因；V2 学会 → 混合目标是原因；V2 学不会 → 超时本身足以挡住。
  - **结果（2026-11-19 读）**：训练 4/4 COMPLETED + `student_job_status=PASS`（3000/3000，1 h 26 min–1 h 32 min），评估 4/4 COMPLETED + `switch_eval_status=PASS`；JSON 的 run_dir / checkpoint（ep 3000）/ intent 已核对。每 600 步完成数，seed 0 / 1（网格 goal-7 同为 3000 轮，来自 `1163578`，seeds 0–2）：

    | 条件 | 网格 3000 轮 | V1 notimeout | V2 anyonly |
    |---|---|---|---|
    | `any` | 0.013 / 0.016 / 0.018 | 0.028 / 0.003 | **3.03 / 4.62** |
    | `up` | 8.5 / 13.6 / 14.5 | 2.60 / 1.03 | 3.94 / 6.88 |
    | `up,turn` 循环 | — | 0.73 / 0.54 | 0.46 / 0.49 |
    | random 阶段 | — | 0.57 / 0.32 | 2.02 / 2.77 |

  - 按判据：V1 学不会（两个 seed 都 < 0.03），去掉超时不仅没救 `any`，`up` 还从 8.5–14.5 掉到 1–2.6 → 超时不是原因（先前假设被否定）。V2 两个 seed 都 ≥ 3 → 带着同样的 120 步超时也能学会 `any`，超时本身挡不住；问题出在混合目标的训练。V2 没练过 `up`/`turn`，`up` 3.9–6.9 来自 `any` 目标族的泛化，`turn` 几乎不会。
  - 局限：V2 同时把 `any` 的练习量提高到约 4 倍，分不开「混合目标互相干扰」和「`any` 练得太少」；M1 对照和 V1（卡住的 `any` 阶段不被截断，`any` 时间更多却仍学不会）都不支持单纯练习量的解释，但这两条都是间接证据。各 2 个 seed，3000 轮。
  - **A / B 已提交（用户批准 A + B，2026-11-19）**：A 老师屏蔽评估 `1166087`；B goal-11 训练 `1166088_0-1` → 评估 `1166089_0-1`；代码已同步（sha256 intent_modules `e072e9ae…`、train_student `a168abbd…`、evaluate_switch `e9af0a0f…`、31 `b3378971…`），未提交 git。
  - **A 结果（`1166087` COMPLETED 13 min，`switch_eval_status=PASS`）**：不屏蔽的 any / up / up,turn = 29.42 / 50.00 / 18.88，与 `1159062` 一致。每 600 步完成数：

    | 屏蔽 | `any` | `up` | `up,turn` 循环 |
    |---|---|---|---|
    | 无 | 29.42 | 50.00 | 18.88 |
    | 相对旋转 [68:72] | 0.033 | 0.034 | 0.019（turn 0） |
    | 目标四元数 [64:68] | 25.29 | 45.82 | — |
    | 两者 [64:72] | 0.032 | 0.031 | — |

    老师几乎完全靠相对旋转：拿掉它，`any` 和 `up` 都从几十降到 0.03（约一半阶段卡满 120 步），`up,turn` 因 `up` 完不成而几乎没有 turn；拿掉目标四元数只降 10–15%。也就是老师没学「从目标 + 当前朝向自己算差值」，`up` 也不例外。这说明老师的能力建立在这 4 维上，但不证明学生缺了它就学不会（goal-7 学生没有它也学会了 `up`）；B 回答这个。结果本地 `Experiment Plan/results/e3-teacher-mask-job1166087/`。
  - **B 结果（训练 `1166088_0-1` COMPLETED + PASS 3000/3000，1 h 36–37 min；评估 `1166089_0-1` PASS）**：意图核对（含相对旋转）最大误差 ≤ 5e-7。每 600 步，seed 0 / 1：`any` 0.034 / **14.99**，`up` 15.05 / 31.11，`up,turn` 循环 6.99 / 12.12，random 阶段 2.01 / 19.47。
    - 按判据：seed 1 学会 `any`，seed 0 没有 → 1/2。对照：goal-7 0/3（3000 和 5000 轮）、V1 0/2；V2（只练 `any`）2/2。
    - 训练回报（ep 1200 / 2200 / 3000）：goal-11 s0 436 / 929 / 946，s1 978 / 3917 / 5077；网格 goal-7 hold 1 三个 seed 3000 轮 795 / 929 / 865（5000 轮 829–947）；hand-8 hold 1 s0/s1 6514 / 6366，s2 1383。即有两个平台：约 800–1000 =「只会 `up`/`turn`」，约 5000–6600 = 三类都会；goal-11 s0 停在低平台。
    - 解读：补上相对旋转让学生能在混合训练里学会 `any`（goal-7 从未做到），`up` 也更好（15–31 vs goal-7 3000 轮 8.5–14.5），但不保证——是否跳出低平台随 seed 变。缺相对旋转是一个原因，不是全部；2 个 seed 不足以估计成功率。
    - 结果本地 `Experiment Plan/results/e3-goal11-job1166089/`。
  - **加 seed（用户，2026-11-19）已完成**：goal-11 seeds 2–5 训练 `1167374_0-3` → 评估 `1167375_0-3`；goal-7 seeds 3–5 训练 `1167376_0-2` → 评估 `1167377_0-2`；设置同 `1166088`（3000 轮，混合目标，hold 1），脚本 `cluster_smoke/submit_e3_relrot_seeds.sh`，清单 `~/cluster_smoke/e3_relrot_seeds_jobs.txt`。全部 7 个训练 `student_job_status=PASS`、7 个评估 `switch_eval_status=PASS`，JSON 拉回 `Experiment Plan/results/e3-relrot-seeds/`。合起来各 6 个 seed（goal-7 seeds 0–2 用 `1163578` 的 3000 轮结果）：
    - `any` 每 600 步完成数（seeds 0–5）：`goal-7` 为 `[0.013, 0.016, 0.018, 0.009, 0.021, 0.015]`（学会 ≥1 的比例 **0/6**）；`goal-11` 为 `[0.034, 14.991, 0.057, 0.424, 10.039, 0.159]`（学会 ≥1 的比例 **2/6**，s1 与 s4；另有 s3=0.42、s5=0.16）。
    - `up` 每 600 步完成数（seeds 0–5）：`goal-7` 为 `[8.52, 13.61, 14.49, 8.14, 14.81, 13.54]`（均值 **12.18 ± 3.03**）；`goal-11` 为 `[15.05, 31.11, 16.14, 17.65, 21.74, 16.47]`（均值 **19.69 ± 6.06**）。`goal-11` 的 `up,turn` 循环数为 `[6.99, 12.12, 7.87, 8.30, 1.54, 7.33]`（均值 **7.36 ± 3.40**）。
    - 统计检验：二值学会比例（2/6 vs 0/6）单侧 Fisher 精确检验 $p = 0.2273$；但按连续完成数，`goal-11` 的全部 6 个 seed 在 `any`（最小 0.034 > 最大 0.021，单侧 Mann-Whitney $U = 36, p = 0.0011$）和 `up`（最小 15.05 > 最大 14.81，双侧 Welch $p = 0.0286$，Mann-Whitney $p = 0.0011$）上均严格高于 `goal-7` 的全部 6 个 seed。结论：显式相对旋转误差 `obs[68:72]` 是混合训练下学会 `any` 的必要条件（无它则 0/6 全军覆没），而 `goal-11` 内部 2/6 起飞的双峰分布来自 2048 envs（老师 8192 envs 的 1/4 样本量）在混合任务中的探索方差。
  - **加 seed 结果（2026-11-19 读）**：训练 7/7 COMPLETED + `student_job_status=PASS`（3000/3000，1 h 34–45 min），评估 7/7 COMPLETED + `switch_eval_status=PASS`；启动时意图核对 ≤ 4.5e-7。本地 `Experiment Plan/results/e3-relrot-seeds/<评估 job>/`。每 600 步（3000 轮）：

    | | seed | `any` | `up` | `up,turn` 循环 | 回报 ep 1200 / 2200 / 3000 |
    |---|---|---|---|---|---|
    | goal-11 | 0（`1166089_0`） | 0.034 | 15.05 | 6.99 | 436 / 929 / 946 |
    | goal-11 | 1（`1166089_1`） | **14.99** | 31.11 | 12.12 | 978 / 3917 / 5077 |
    | goal-11 | 2 | 0.057 | 16.14 | 7.87 | 770 / 814 / 880 |
    | goal-11 | 3 | 0.424 | 17.65 | 8.30 | 844 / 957 / 1249 |
    | goal-11 | 4 | **10.04** | 21.74 | 1.54 | 341 / 823 / 1179 |
    | goal-11 | 5 | 0.159 | 16.47 | 7.33 | 501 / 881 / 887 |
    | goal-7 | 0–2（`1163578` it-3000） | 0.013 / 0.016 / 0.018 | 8.52 / 13.61 / 14.49 | — | 3000 轮 795 / 929 / 865 |
    | goal-7 | 3 | 0.009 | 8.14 | 4.76 | 492 / 791 / 804 |
    | goal-7 | 4 | 0.021 | 14.81 | 6.56 | 149 / 516 / 890 |
    | goal-7 | 5 | 0.015 | 13.54 | 6.35 | 406 / 932 / 897 |

    - 预定判据：goal-11 学会 `any` 2/6（seeds 1、4），goal-7 0/6。Fisher 精确检验单侧 p = 0.23，双侧 p = 0.45，**不显著**；6 个 seed 不足以确认相对旋转是必要的。
    - 训练回报：只有 goal-11 s1 明显跳出低平台（5077）；s3、s4 在 3000 轮刚到 1180–1250，还在上升，其余 7 个 run 都在 800–950。s4 `any` 10.0 但 `up,turn` 只有 1.5（会了 `any` 却丢了 turn）；s3 回报上升但 `any` 0.42 未过线。
    - 事后观察（非预注册）：goal-11 的 `up` 6/6 都高于 goal-7 的 6/6（15.05–31.11 vs 8.14–14.81，Mann–Whitney 精确 p = 0.002），但两组的边界很近（15.05 vs 14.81）。
    - 解读：补上相对旋转后，部分 seed 在 3000 轮内学会 `any`，goal-7 一个都没有，方向一致但统计上不能下结论；更稳的结论是「混合训练下 `any` 很难学，3000 轮内跳出低平台靠 seed」。是否需要更长训练（s3/s4 还在升）未测。
  - A / B 设计：代码对照发现老师比 goal-7 学生多的、和 `any` 有关的输入只有 obs[68:72] 相对旋转 = quat_mul(object_rot, conj(goal_rot))（goal 四元数、转轴误差、is_turn 学生都有或能算出；hold 计数只关 `turn`）。假设：学生要自己学四元数乘法，混合训练里 `any` 太少学不出来，`up` 不需要它；老师直接有 [68:72]。另一差别：老师 8192 envs，学生 2048。
    - A：冻结 v4 老师，`evaluate_switch.py --teacher_mask goal|relrot|both`（每步把这些列换成另一个 env 的同列，随机排列），31 条件 `<cycle>@mask-<x>`；条件 any / up / up,turn 原样（须复现 `1159062`）+ relrot ×3 + goal ×2 + both ×2。
    - B：`goal-11` = goal-7 + obs[68:72]（`intent_modules.py`，CPU 测试 14/14），v4 训练环境（随机混合，同网格），hold 1，seeds 0–1，3000 轮；评估 any up up,turn random。判据同 V1/V2：`any` ≥ 1 / 600 = 学会 → 缺相对旋转是原因。
    - 提交脚本 `cluster_smoke/submit_e3_relrot_diag.sh`（先同步 intent_modules.py、train_student.py、evaluate_switch.py、31_switch_eval.sbatch、test_unified_intents.py 和这个脚本，再在登录节点 `bash submit_e3_relrot_diag.sh`），记录 `~/cluster_smoke/e3_relrot_jobs.txt`。
  - 结果本地 `Experiment Plan/results/e3-any-diag-v1v2/<评估 job>/`，集群 `~/data/low-dim-intent/runs/switch-eval-job116458{9,91}_{0,1}/`，日志 `~/cluster_smoke/logs/e3-diag-*.out`。
- **wrist2-finger-k 判断（用户问，2026-11-18）**：hand-8 留住的手腕方差 Stage 1 93% / 93%，M2 88% / 68%，face-turn 67% / 81%（手腕只占动作方差 5.9%），v4 77% / 63%（手腕占 11.3%）。v4 按阶段（30 万行抽样）：hand-8 手腕 / 手指 / 全部 = 翻转 0.69 / 0.69 / 0.69，转顶层 0.72 / 0.69 / 0.69；wrist2-finger-6 = 1.00 / 0.61 / 0.66，转顶层 1.00 / 0.60 / 0.64。hand-8 在 v4 上丢的手腕和手指一样多（约 30%），两种阶段也一样，不是专门丢手腕；wrist2-finger-6 保住手腕，代价是手指降到约 0.61，总保留方差 0.66 < hand-8 0.70。是否加入网格：待用户决定。

### 2026-11-19 收尾与当前即时实验（优先级 0 + 优先级 1：零成本感知诊断）

> 说明：此部分为当前计划的收尾与即时可跑的零成本评测（不需要重训策略或 VLM），后续涉及 BFS 闭环、SFT、GRPO 和模块化对比的完整路线已写入 [GOAL_LEVEL_RL_PLAN.md](GOAL_LEVEL_RL_PLAN.md)。

#### 优先级 0：底层资产与在途队列收口（已完成，2026-11-19）

- [x] **P0.1 读取 `Unified v4 Seed 1`（训练 `1160193` PASS，评估 `1160196` PASS）**：终值训练回报 `8567.8`（seed 0 为 `8970.2`）；5 个切换条件结果已拉回 `Experiment Plan/results/switch-eval-job1160196/`。每 600 步（Seed 0 / Seed 1 / 均值）：`up,turn` 循环 **18.88 / 17.16 / 18.02**（turns 18.57 / 16.88，中位步数 18 / 19，停滞率 0.02% / 0.01%，跌落 0.014 / 0.012）；`any,up,turn` 循环 **10.50 / 9.65 / 10.08**（turns 10.16 / 9.32，停滞率 0.01% / 0.06%）；`any` 29.42 / 27.28；`up` 50.00 / 47.73；`random` 总阶段 33.69 / 31.12。`Unified v4` 在两个种子上高度一致（相差 5%–9%），可复现性确认。
- [x] **P0.2 读取 `Block 3b`（训练 `1160178_0-5` PASS，评估 `1160184_0-5` PASS）**：M2 10 Hz (`hold=6`) 补齐 seeds 3–4 后（$n=5$，详见 `RATE_SWEEP_PLAN.md`），每回合成功次数：`goal-yaw` **31.05 ± 2.38**（Recovery 0.992），`goal-4` **27.96 ± 1.81**（Recovery 0.724），`hand-8` **23.06 ± 4.50**（Recovery 0.296）。`hand-8` vs `goal-yaw` 双侧 Welch **$p = 0.0124$**（Mann-Whitney $U = 0, p = 0.0079$，无重叠），满足预注册 $p < 0.05$ 判据，10 Hz 反转在 M2 上正式显著。
- [x] **P0.3 读取 E3 相对旋转扩展种子（`goal-11` seeds 2–5 `1167374`/`1167375`；`goal-7` seeds 3–5 `1167376`/`1167377`）**：已读（2026-11-19）。`any` 学会（≥1）goal-11 2/6 vs goal-7 0/6，二值 Fisher 单侧 p = 0.23（不显著），但连续完成数与 `up` 完成数上 goal-11 6/6 全高于 goal-7 6/6（Mann–Whitney 双侧精确 p = 0.002，Welch `up` p = 0.029）；详见 E3「加 seed 结果」。

#### 优先级 1：零样本 VLM 细粒度感知瓶颈快速破局（已完成，2026-11-20；4/4 PASS）

背景：在 `1167491`（`z1-mcq-v2`，180 帧 test）中，Qwen3-VL-8B-Instruct 的粗粒度 `up` 准确率达 0.73–0.91，但细粒度卡位（`turned` 0.11–0.27、`band` 0.03–0.22）接近随机，且多相机输入比单相机更差（`c1` 0.221 → `c1+s040+s220` 0.069）。在现有的 `z1-main-job1163684` 图像和分割掩码上完成了 4 组视觉输入改造评测（`z1_mcq_eval.py --visual_mode crop|montage|seam|som`，样图在 `Experiment Plan/results/z1-p1-preview/`，完整 JSON 在 `Experiment Plan/results/z1-p1-eval/`）：

- [x] **P1.1 工作区 ROI 自动裁剪放大 + 单图多视角拼贴（`crop` `1168586` PASS，`montage` `1168587` PASS）**
- [x] **P1.2 基于现有分割标签的 Set-of-Mark (SoM) 视觉锚点消融（`som` `1168603` PASS；首次尝试 `1168588` 落到 ARM 节点 `xgpj0` 报错，排除后重交成功）**
- [x] **P1.3 上下层交界线/接缝对比度增强（`seam` `1168589` PASS）**

**结果汇总（test 180 帧，CircularEval 准确率，括号内为 vanilla）：**

| 题目 / 视角 (`c1+s040+s220` 为主) | `raw` (`1167491`) | `crop` (`1168586`) | `montage` (`1168587`) | `seam` (`1168589`) | `som` (`1168603`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `up` (`c1+s040+s220`) | 0.806 (0.928) | 0.861 (0.967) | **0.900** (0.939) | 0.878 (0.967) | 0.867 (0.961) |
| `up` (`c1`) | 0.850 (0.911) | **0.906** (0.956) | **0.906** (0.956) | **0.906** (0.961) | 0.894 (0.961) |
| `turned` (`c1+s040+s220`) | 0.178 (0.517) | 0.100 (0.633) | 0.311 (0.689) | 0.228 (0.706) | **0.800 (0.800)** |
| `band` (`c1+s040+s220`, $n=159$) | 0.069 (0.226) | 0.031 (0.239) | 0.075 (0.151) | 0.075 (0.195) | **0.679 (0.792)** |
| `band` (`c1`, $n=77$) | 0.221 (0.481) | 0.195 (0.403) | 0.195 (0.403) | 0.182 (0.325) | **0.714 (0.831)** |
| **三题全对 `all_levels` (`c1+s040+s220`)** | **0.000** | **0.000** | **0.000** | 0.025 | **0.509**（`c1`: **0.571**） |

**核心结论（按判据转入优先级 2 + 方案 C）：**
1. **排除低级视觉/分辨率假说**：`crop`（放大 4 倍）、`montage`（单图拼贴）、`seam`（亮青色分界缝）均无法让零样本读出 `band`（CircularEval 仅 0.031–0.075，三题全对 0.000–0.025），但 `montage` 将多相机 `up` 从 0.806 显著提升至 **0.900**（`red` 朝上从 0.29 升至 0.71），且单图上下文开销更低，定为后续视觉输入底座。
2. **瓶颈是空间指代（Spatial Grounding），而非 ViT 视觉盲区**：在同一张 `crop` 图上仅加 `[1]`/`[2]` 数字标号（`som`），零样本 `turned` 跃升至 **0.800**、`band` 跃升至 **0.679**（vanilla 0.792，平均 token 从 1180 降至 335），三题全对从 **0.000 → 0.509**。说明 ViT 特征完全看得清 1/3 窄带颜色，正式按判据转入 [GOAL_LEVEL_RL_PLAN.md](GOAL_LEVEL_RL_PLAN.md) 的优先级 2 与方案 C（Map-then-Reason LoRA SFT + 闭环 GRPO）。

## 日程

| 日期 | 实验 | 报告 |
|---|---|---|
| 11-16 晚 | E0：读续训结果，提交评估和视频 | — |
| 11-17 | E0 冻结；E1 代码、rollout、PCA、dry run，晚上提交网格；E2 环境 | 方法、迭代 2–4 的结果 |
| 11-18 | E1 网格跑完并评估；E2 基准和关口 | 迭代 5 的结果和图 |
| 11-19 | E1 分析；E2 扫描和分析；**24:00 实验截止** | 相关工作 |
| 11-20 | 只补图 | 迭代 6 一节，讨论，局限 |
| 11-21 | — | 全文通读，压到 8 页以内，核对每个数字的 job ID |
| 11-22 | — | 最后检查，提交 |

## 砍掉的顺序

时间不够时从下往上砍，砍掉的写进 future work：

1. E2 的 hold / delay 扫描（保留 60 Hz 关口结果）。
2. E1 续训到 10000。
3. E1 的 `hand-8` × N = 6。
4. 整个 E2（11-18 晚环境还没通过 CPU 测试就砍）。
5. E1 的全部 `hand-8`（只剩 blind 和 `goal-yaw`，Q1 降为「目标通道在第三个任务上的速率曲线」）。

**底线**：11-19 晚如果 E1 没有通过评估的结果，迭代 6 在报告里只写 expert（任务定义、27.7 次保持转动、视频帧），占一段，其余归入 future work。

## 结果怎么读

- E1：`hand-8` 在 10 Hz 以下低于 `goal-yaw` → 反转在三个任务上都出现。`hand-8` 没有崩溃 → 结论收窄为「取决于任务」，并对比三个任务 expert 动作的保持误差。
- E1 延迟：落在等待时间线上 → 延迟预算就是任务目标速率的算术；低于线 → 离散目标在等待期间有额外代价（例如顶层被碰离卡位）。
- E2：交接损失约为 0，且慢速选择器落在等待时间线上 → 两个闭环技能加一个慢速选择器即可组合，上游只需要 1 bit + 目标。交接损失大 → 组合的瓶颈在技能的起始状态分布，不在通道速率。

## 已知风险

- **face-turn 学得晚**：expert 训练到一半时 reward 只有约 1200，学生在 5000 iterations 内可能学不起来。对策见新定义 5。
- **expert 有停滞**（16 个环境里 2 个）：会直接压低 E2 的结果，所以 E0 先量出停滞率，E2 的指标把停滞单独列出。
- **只有一个 expert seed**，学生 3 seeds；检验不做多重比较校正。和前面各迭代相同，报告里照实写。
- **R 目标限制为轴朝上**，所以组合结果只对这个目标子集成立。

## 日志

### 2026-11-17 01:50 E1 提交（用户确认「都按你说的来」）

- 确认的定义：actor 加铰链角 (sin θ, cos θ)；目标级 `goal-yaw`；动作级 `hand-8`；5000 iterations。老师 = handoff 版冻结 expert（`expert-faceturn-cap0.05_hold10_handoff-seed0/frozen/expert-faceturn-handoff-seed0.pth`，sha256 `477f3d43…e0f30a`），和切换实验用的是同一个；与计划原文「10000 的 checkpoint」不同。
- 任务设置：`HINGE_EFFORT_LIMIT=0.05`，`FACE_TURN_HOLD_STEPS=10`，只用平放起始（不用 `FACE_TURN_START_STATES`）；脚本里写死。
- 代码（已同步到集群，未提交 Git）：
  - `intent_modules.hinge_angle_obs`：θ 由观测自身算出，`quat_mul(obs[68:72], obs[64:68])` = 铰链旋转；本地 CPU 测试误差 3.6e-7，`goal-yaw` 在合成 face-turn 观测上误差 0。
  - `intent_env_wrapper.py` 参数 `hinge_angle`（actor = 149 + 2 + k，铰链角不经过意图通道）；`train_student.py --hinge_angle_obs`，写入 `intent.json`（连同扭矩上限和 hold 步数）；`evaluate_students.py` 读回并核对。
  - `STUDENT_STAGE=faceturn-hidden`（`15` / `16`），`17` 在 `EVAL_TASK` 为 face-turn 时设置铰链变量；`11` 增加 `ROLLOUT_TASK=faceturn`；`14` 增加 `CHECK_FACE_TURN=1`、`INTENT_EXPERT_FILE`；`validate_intent_modules.py --face_turn`（跳过物体意图检查，增加 goal-yaw、铰链角、目标在卡位上的检查）。
- 作业链（全部 `afterok`；任何一步失败，后面的作业会一直挂起，需要取消、修复、重新提交）：
  - rollout `1153675`（512 × 2400，seed 20261007）→ 动作 PCA `1153676` → 校验 `1153677`；
  - dry run `1153678`（blind / goal-yaw / hand-8 × hold 1 / 6 / 30，256 envs × 3 iterations）→ dry checkpoint 评估 `1153679`；
  - 网格 `1153680_0-19`（`25_train_and_eval.sbatch`，训练后在同一作业里评估全部 checkpoint）：任务 i → 意图 `blind:1 goal-yaw:1 goal-yaw:6 goal-yaw:30 hand-8:1 hand-8:30 hand-8:6` 的第 i // 3 个，seed i % 3。
  - 输出：`runs/student-faceturn-hidden-<intent>[-holdN]-seed<s>-job1153680_<i>/`，评估 `runs/eval-student-faceturn-hidden-…-job1153680_0/`。集群上的清单：`~/cluster_smoke/faceturn_grid_jobs.txt`。
- **未提交（队列上限 32）**：网格任务 20（`hand-8:6` seed 2，同样的变量加 `--array=20`）；冻结 expert 在学生评估器下的 P_E（`17`，`EVAL_TASK` = face-turn，`EVAL_INTENT=blind EVAL_LAST_ONLY=1`，`EVAL_RUN_DIRS` = handoff 训练目录 `IsaacLab/logs/rl_games/shadow_hand_faceturn_cap0.05_hold10_handoff/2026-11-16_22-44-56`）。
- 还没有结果。之后：读链上每个作业；只评估的延迟（Q2）；分析脚本。

### 2026-11-17 02:10 作业链通过，补交剩下两项

- rollout `1153675` PASS：1,228,800 步，平均 episode reward 11238。
- 动作 PCA `1153676` PASS：20 维累计方差 k = 2 / 4 / 8 为 45.0% / 68.0% / 87.8%（Stage 1 expert：35.8% / 56.8% / 81.1%）。数据集 `datasets/action-pca-expert-faceturn-rollout-seed0-job1153675-job1153676/`。
- 校验 `1153677` PASS，0 项失败：重建的 expert 与记录动作相差 5.7e-6；`goal-yaw` 重构目标四元数 6e-8；铰链角与环境的铰链误差相差 6.1e-4（限 2e-3）；目标都在 90° 卡位上。
- dry run `1153678` PASS（9 / 9）：actor 输入 blind 151，goal-yaw 153，hand-8 159。dry checkpoint 评估 `1153679` PASS（评估器的 face-turn 路径能跑；3 iterations 的 checkpoint 成功次数为 0，符合预期）。
- 补交：网格任务 20 = `1153770_20`（`hand-8:6` seed 2）；expert 的 P_E = `1153769`（`17`，`EVAL_INTENT=blind EVAL_LAST_ONLY=1`）。至此 21 个 run 全部在队列里，尚无训练结果。

### 2026-11-18 E2' 结果（`1160401`）

- `1160401` COMPLETED 0:0（xgpf6，21 min），16/16 `command_status=0`，`switch_eval_status=PASS`。JSON + 日志 `results/switch-eval-job1160401/`；分析 `analyze_e2_selector_sweep.py` → `results/e2-selector-sweep/`（csv/json/png），`e2_selector_sweep_analysis_status=PASS`。
- **回归 PASS**：N=1/d=0 两个循环与 `1153366` 逐位一致（全部比较字段相同；up,turn 7.510，any,up,turn 5.482），直通时 `fields_replaced` = 0。
- 循环数 / 600 步（括号 = 预测 600/(T0+L·w)）：
  - up,turn：base 7.51；h6 7.12 (7.07)；h30 5.40 (5.51)；d6 6.43 (6.53)；d30 3.99 (4.29)；d60 2.89 (3.00)。
  - any,up,turn：base 5.48；h6 5.03 (5.13)；h30 3.92 (3.92)；d6 4.53 (4.71)；d30 2.88 (3.01)；d60 2.09 (2.07)。
- 每阶段额外步数 ≈ 等待：w 2.5/14.5/6/30/60 → up,turn 2.2/15.6/6.7/35.2/64.0，any,up,turn 3.3/14.5/7.7/32.9/59.2。up,turn d30/d60 比线多约 4–5 步/阶段（循环数低 3–7%），其余在 ±2 步内。每类阶段的中位步数都按等待增加（d60：up +52，turn +60；any +53，up +60，turn +61），没有哪个方向的交接单独变差。
- 跌落不随等待增加（up,turn 0.103 → d60 0.075 / 600 步；阶段少了，交接也少）。turn 停滞（>120 步）3.5% → d60 8.4%（any,up,turn 4.5% → 10.8%），主要因为中位步数本身升到 93–94 而阈值没平移，不单独解读。
- 过时比例实测略低于预测（up,turn d60 0.543 vs 0.600）：模型忽略了跌落和重置（重置后消息立即刷新）。
- 对照：只用 Stage 1 专家 0.50 / 0.67 循环（完成 up/any 后卡在 turn），只用转动专家 0.006 / 0.001（跌落 0.33 / 600 步）→ 选择器是必需的。注：only-expert 条件的字段替换没计入 `fields_replaced`（计数在替换之前），不影响结果。
- 读法：两个闭环专家 + 慢速选择器可以组合；选择通道变慢的代价就是等待时间，和 rate sweep 里 goal-4 的等待时间线同一规律。单个专家种子、单个评估种子，不做检验。

