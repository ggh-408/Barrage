# Barrage：300 弹纯图像躲避

长测入口为 `tools/evaluate_targeted_dagger.py`，默认权重为
`best.pt`。在 `barrage` 环境中运行
`python -B tools/evaluate_targeted_dagger.py --mode fresh --episodes 3000`。
长测直接使用指定权重和当前代码，无需 `run_manifest.json`、历史源码快照或兼容性登记。
`fresh` 模式不读取历史评估结果，仅排除权重配置中的训练种子和固定留出种子；
相同 `--seed` 与权重配置可重复生成相同测试集。显式回归测试使用
`--mode regression --seeds-csv <种子CSV路径>`。
评估范围和局数按当前任务或用户明确要求确定，不再规定固定 200 局正式验收。
延迟优化和窗口测速不自动触发完整局数评估。

## 环境与依赖

项目命令统一在 `barrage` Conda 环境执行。Numba 与 llvmlite 直接从该环境加载，
当前依赖版本由 `requirements.txt` 固定为 0.61.2 和 0.44.0。
活动代码不再从项目 `.runtime` 目录加载包；Numba 编译仍在进程内完成。

```powershell
conda activate barrage
python -m pip install -r requirements.txt
python -B tools/smoke_environment_dependencies.py
```

依赖冒烟入口会阻止访问 `.runtime`，验证像素规划、RGB 并行处理、窗口推理及
多进程采集，并打印实际加载的包和 DLL 路径。它用于检查运行兼容性，不提供
训练吞吐或长局成功率结论。安装其他依赖时保留当前可用的 CUDA 版 PyTorch。

## 当前配置

模型版本为 12，仅训练动作策略头和教师代价辅助头；保留共享动作几何、
384 槽图像 tracker 及 0.5 秒 receding 像素规划。训练、评估、窗口默认均为 15 个决策步。默认关闭独立解析约束。

当前定向长测入口和窗口统一使用 `commit_safe_ranking`：有可覆盖下一步
33.3 毫秒的名义像素安全路线时，只在这些路线中排序；全部不满足时优先选择名义安全前缀最长的路线。
新控制器配置记录 `ranking_variant=commit_safe_ranking` 和
`ranking_safety_basis=nominal_pixel`。
测试配置记录实际排序版本，历史长测成绩保持原样。

窗口默认固定加载 `best.pt`，与已完成的 3000/3000 局长测使用同一候选权重，
并安装匹配的 `planner_readiness_20260921_parallel` 实验控制器、`nominal_pixel` 安全排序及 0.5 秒规划。
图像检测和跟踪特征沿用长测共享组件；窗口继续使用 CPU 单帧推理。新增训练轮次不会自动替换此窗口权重。
窗口仅接受版本 12 权重；已移除窗口中的旧列表子弹分支、四布尔方向输入和最新轮次权重解析器。
训练初始权重仍为 `diagnostics/risk_removal_20260920/policy_teacher_cost.pt`。共享加载器为迁移旧实验保留 10/11 版本 checkpoint
加载能力，仅跳过旧风险输出层；其余权重严格校验。旧训练运行需以
`--initial-checkpoint` 在新目录开始，不能沿用风险头优化器和历史选模分数恢复训练。

```powershell
conda activate barrage
python -B Barrage.py --ai --no-music
python -B -m barrage_rl.train_tracked_policy --output-dir runs/visual_set_v52
python -B tools/evaluate_targeted_dagger.py --checkpoint best.pt --mode fresh --episodes 3000
```

窗口统一使用 `tick_busy_loop` 精确限帧（100 FPS）。AI 和手动模式共用
启动配置、数组子弹存储及物理/渲染路径；手动命令行入口为
`python -B Barrage.py --no-ai --no-music`，不传模式参数时保留设置界面。
性能测试统一使用 `tools/test_visible_window.py`：添加 `--no-ai` 关闭 AI，
添加 `--immune` 开启免伤，例如
`python -B tools/test_visible_window.py --no-ai --immune --seconds 120`。

窗口 RGB 复制和前景扫描默认使用 4 个 Numba 线程，按图像行并行并保持
逐位一致。每次图像处理后恢复规划器原有线程设置，模型权重及决策规则保持不变。
可用 `--rgb-workers 2` 降低图像处理并行度，或用 `--rgb-workers 0` 回退串行；
该参数适用于 `Barrage.py` 与 `tools/test_visible_window.py`。
并行仅在窗口图像处理上下文启用，训练和上下文外的共享调用沿用串行实现。

窗口跟踪器使用单线程融合关联扫描，并复用既有两坐标缓冲区完成数组整理；
保留原有距离计算精度、同距匹配顺序、速度拟合及遮挡处理规则。
非标准精度或非有限关联输入回退原 NumPy 实现。共享跟踪器保留原算法，
窗口使用独立的 `WindowImageTracker` 实现，权重及模型输入格式保持一致。

默认训练采用 36 个环境、9 个采集进程、batch 512；窗口继续使用 CPU。
任务维持 300 弹、逐弹独立 10% 定向概率、120 Hz 物理、30 Hz 决策、120 秒局限。
评估局数按任务确定。

窗口渲染上限为 100 FPS；物理更新固定为 120 Hz，当前零延迟配置每 4 个物理步
同步决策一次，即 30 Hz（约 33.33 毫秒预算）。
计分按物理步数累计，每12步增加1分；120秒回合包含14400个物理步。

当前任务配置：300 发子弹、子弹大小 5、速度 240 px/s、10% 瞄准弹、
120 秒 episode 上限。评估局数和通过条件按具体任务确定。
`model_iqm`、CVaR5、RMST 与置信区间继续报告，
但 `best.pt` 只按 `success_at_limit` 选择。
暂不把尺寸或速度泛化作为训练目标。

部署策略只能接收图像。模拟器位置和速度只允许用于训练期教师标签、诊断和
正式上限评估，不得进入策略推理接口。

目标弹概率统一为10%，每颗新生成或出界补生的子弹独立按该概率抽样，不要求
单局目标弹数量严格等于总数的10%。每个结果必须同时记录弹数和目标弹概率；
不同任务规格的历史 checkpoint 不能直接视为当前300发任务已经验收。

## 开局生成语义

部署开局使用批量 `spawn_bullets`，后续出界统一按索引逐颗调用等价的
`spawn_bullet`。当前300发任务把开局子弹均分为10批，每批30发；第一批在
0.0秒生成，其余批次每隔0.1秒生成一次，最后一批在0.9秒生成。每个后续批次
都使用生成时刻的飞机位置与速度计算定向弹轨迹。开局批次只使用同一边缘生成器
采样，不执行当前碰撞、原地预警或多动作可恢复性门槛，也不会因门槛重新采样。
若总弹数无法被10整除，余数归入最后一批；总弹数少于10时，前若干批各生成
1发，其余批次为空。

## 单一共享游戏内核

`barrage_rl/runtime_core.py` 是窗口、训练、正式评估和测试唯一允许实现游戏语义的
模块。它统一拥有参数抽样、动作向量、开局生成、飞机积分、逐颗补生、整场子弹
推进、像素碰撞、世界RGB绘制和图像轨迹预测提示。`Barrage.py` 与
`barrage_rl/env.py` 只负责窗口/Gym状态适配，不得重新实现上述规则；旧的
`barrage_rl/dynamics.py` 已删除，避免出现第二个可导入的游戏语义入口。修改游戏规则后必须通过
同seed状态、RNG、RGB和碰撞对拍测试，并重新运行本机吞吐基准。

## 当前 tracked-policy 结构

```text
当前图像 → 384 槽持久 tracker → 轨迹 token 与全局图像特征
  → 9 个动作 query + 共享动作几何 + cross-attention
  → 动作策略头 / 教师代价辅助头
  → 策略动作 → 图像几何检查与 0.5 秒 receding 像素规划
  → 每个决策周期输出一个最终动作
```

教师代价辅助头用于训练，默认推理排序权重为零。几何时域仍保留原有维度，
以兼容共享编码器；内部旧回放接口的风险张量位置只返回常量，不含参数且不参与学习或过滤。
外部每次只提供当前图像；历史来自 tracker 和按局保存、重新验证的图像规划。
exact 教师只在训练采集进程读取模拟器状态，部署模型接口仍只接收图像。

训练器只按 `success_at_limit` 选择 `best.pt`，即最大化达到 episode 上限的
held-out 局数。成功率相同时采用最新 checkpoint，包括低于 100% 的同分情况。`model_iqm`、`model_mean`
与置信区间只报告，不参与选模。

## 上限评估

快速 smoke 只验证链路：

```cmd
python -m barrage_rl.evaluate_teacher --teacher-kind exact --workers 2 --output-dir .tmp\exact_teacher_smoke --smoke-test
python -m barrage_rl.evaluate_image_oracle --planner-kind recovery --observation-size 192 --workers 2 --output-dir .tmp\image_oracle_smoke --smoke-test
```

exact teacher 上限评估示例（每局 120 秒）：

```cmd
python -m barrage_rl.evaluate_teacher --teacher-kind exact --teacher-horizon-seconds 1.5 --workers 9 --seed 1600000 --bullets 300 --output-dir .tmp\exact_teacher_300_formal
```

长评估每完成 10 局打印进度，并原子更新 `teacher_episodes.partial.csv`。

随后用 recovery 规划器验证严格图像边界；它只能读取 192×192 语义帧并通过
持久 tracker 重建状态。当前300发正式命令为：

```cmd
python -m barrage_rl.evaluate_image_oracle --planner-kind recovery --observation-size 192 --workers 9 --seed 1700000 --bullets 300 --output-dir .tmp\image_oracle_recovery_300_formal
```

长评估支持在相同配置和输出目录上追加 `--resume`，从原子 partial CSV
继续未完成的 episode。

## 训练运行与恢复

通用训练入口为 `python -u -B -m barrage_rl.train_tracked_policy`。
采集和单个训练 epoch 每 30 秒打印吞吐、进度与 ETA，评估按配置的批次提交进度。
默认输出为 `runs/visual_set_v52`；启动前检查目录占用，非空目录需明确选择恢复
或使用下一个未占用的版本号。

`tools/train_targeted_dagger.py` 是单独的预配置实验入口，从
`diagnostics/dagger_ready_20260921/training_config.json` 读取配置，并安装匹配的
实验控制器。`--train` 会检查预检记录、启动源码快照和初始权重直接内容是否一致；
代码变化后需重新验证，不能直接沿用旧预检。该入口的参数与通用训练器分开管理。
定向长测仍依赖 `diagnostics/planner_readiness_20260921` 中的控制器源码和上述
训练配置；清理 diagnostics 时须保留这些活动依赖。窗口通过
`barrage_rl/window_runtime.py` 安装独立的 `WindowPixelGuard`，与长测保持匹配的
控制器标识和排序规则，实际实现位于 `barrage_rl/window_planner.py`。

新训练默认通过系统随机源生成数据收集种子和固定 held-out 评估种子，并检查
`runs/visual_set_v*` 的历史配置，保证两组实际使用的 episode 种子互不重叠。
解析后的种子会写入 `config.json` 和 `run_manifest.json`。同一 run 的每个 DAgger
round 使用不同的收集种子，但复用该 run 的固定评估集以公平比较 checkpoint；
`--resume` 也会强制恢复原种子。若显式传入 `--collection-seed` 或
`--evaluation-seed`，它们仍须保持项目内唯一；与历史 run 冲突时，训练会在创建
输出前拒绝启动。

训练器保存 `latest.pt`、`replay_latest.npz`，并按选模规则更新 `best.pt`。
通用训练中断后使用原输出目录恢复，例如：

```powershell
python -u -B -m barrage_rl.train_tracked_policy --output-dir runs/visual_set_v52 --resume --rounds <总轮数>
```

将占位符替换为实际总轮数；训练器会拒绝不兼容的 replay/模型配置。
预配置实验入口不支持用这条命令替代其控制器和启动校验。

显式 smoke 只验证链路，不使用历史权重，也不能作为性能或通过率证据：

```cmd
python -u -B -m barrage_rl.train_tracked_policy --output-dir .tmp\tracked_train_smoke --smoke-test
```

## 独立评估

评估器和可视化部署会从 checkpoint 自动恢复训练时的动作时序。当前零延迟
checkpoint 显式保存 `evaluation_causal_action_delay_steps=0`；一周期 checkpoint
仍使用因果队列。
`--causal-action-delay-steps` 只用于显式 smoke 配对诊断，正式评估必须与 checkpoint
记录的时序一致，避免旧权重被新的执行语义静默解释。

评估范围按任务确定。当前 `tools/evaluate_targeted_dagger.py` 支持通过 `--episodes`
设置完整时长的补充测试局数；`fresh` 默认 3000 局，`regression` 默认读取完整 CSV
种子池。默认使用 9 个评估进程、每批 20 局、9 个搜索线程和 CUDA，结果不参与选模。
`--smoke-test` 默认只运行 2 局、每局 5 秒。

通用 `barrage_rl.evaluate_tracked_policy` CLI 仍限制完整评估为 200 局，
教师与 image-oracle CLI 也保留该限制。`tools/train_targeted_dagger.py --evaluate`
固定运行 200 局并写入 `runs/visual_set_v52_eval`。这些属于当前入口实现，不能据此
自动扩展用户要求的评估范围；测试当前窗口权重优先使用匹配的定向长测入口。
评估结果保存 episode CSV、完整 summary、checkpoint 路径及动作直方图，来源核验采用直接内容比较。

### 部署时序

可见窗口根据checkpoint元数据选择时序，在当前控制边界从当前RGB直接生成当前动作。
固定步评估在
config/summary中记录`causal_action_delay_steps`，正式评估必须与checkpoint一致，
不能混合不同时序的结果选模。首帧PyTorch冷启动在物理计时开始前完成。

### 方案二：图像轨迹预测与残差恢复

方案二已在实时路径启用。首帧仍运行完整RGB检测；后续帧仅使用先前图像检测建立的
轨迹预测局部中心。视觉上不歧义的候选直接采用，接触/交叉等歧义区域交回原有确定性
残差覆盖算法；边缘和轮换竖带优先发现新生/重生目标，随后只处理仍未解释的白色像素。
若latest-only队列跳过控制边界，预测和轨迹时间戳按实际经过的决策数推进。该路径不
读取子弹或飞机的模拟器位置、速度或其他游戏状态。

同时，RGB阈值检测改为uint8视图，避免每次决策复制三份820×820 int16通道；轨迹
关联只排序门控内的有限候选，并在实时恒速弹道路径中避免重复拟合已经确定的速度。
训练和固定留出评估仍保留原有速度重拟合默认值，因而不会静默改变实验语义。

tracked-policy 评估按配置分批并行执行。每批完成后
原子重写 `evaluation_episodes.partial.csv` 和 `evaluation_progress.json`；最终
`evaluation_episodes.csv`、`evaluation_summary.json` 与动作直方图仍只在全部完成后
原子提交。
复测命令：

```cmd
python tools\benchmark_tracked_collection.py --workers 4 6 8 9 10 --env-count 32 --decisions 32 --teacher-kind exact --observation-size 384 --output .tmp\tracked_collection_exact_obs384_benchmark.json
```

## 当前训练参数更新

`train_dagger.cmd` 与 `barrage_rl.train_tracked_policy` 默认使用 300 发弹、
精确教师 `teacher_reaction_seconds=0.10` 秒（120 Hz 物理下为 3 个决策步）。
模型与 tracker 保持 384 槽、定向概率保持 0.10；评估局数按任务确定，
评估弹数默认300发，可通过 `--evaluation-bullets` 单独设置，训练仍为300发。
历史 run 的已保存配置与成绩保持原值；新部署、训练和评估统一默认300发。
新训练默认不加载历史 replay，避免混入旧弹数或旧动作保持时间的标签。
已有 run 的 resume 参数必须保持原值；新参数应使用一个未占用的新版本目录。

## 项目不变量

- 主任务目标弹概率固定为 `targeted=0.10`，且不随子弹总数变化；每颗新生成的
  子弹独立抽样，因此单局实际比例允许随机波动。
- 部署、tracked DAgger 训练和轮次评估统一默认300发，
  模型和 tracker 容量均为384。训练 manifest 与轮次评估必须跟随训练配置。
- 项目 Python/Conda 环境固定为 `barrage`，项目命令和依赖安装均在该环境中执行。
- 实际测试与部署时，模型及控制策略只能接收当前图像帧，不得读取或注入子弹、
  飞机的位置、速度或其他模拟器/游戏状态；策略外围的辅助决策逻辑同样受此限制。
- 正式部署、训练采集及评估默认使用策略动作与 receding 图像像素保护。
- `runs` 下主训练目录只能命名为 `visual_set_vN`；同一版本最多添加一个简短用途
  后缀，例如 `visual_set_vN_eval`，不得串联多个后缀或把超参数、轮次和临时说明
  写入目录名。
- 评估局数和范围按当前任务或用户明确要求确定；延迟优化和窗口测速不自动触发固定局数验收。
- 策略评估按入口配置并行分批提交；通用评估默认每批 10 局，当前定向长测默认
  每批 20 局。中间文件和最终汇总都必须原子写入。
- `best.pt` 只按 `success_at_limit` 选择；任何同分情况均采用最新 checkpoint。
  `model_iqm`、`model_mean` 与置信区间只报告、不参与选模。
- `results.png` 的面板、指标、布局、格式、标签、颜色、图像格式及整体视觉样式
  必须与当前实现完全一致；右上角可靠性面板主纵轴最大值固定为100，其下界与
  其余曲线和副轴继续独立使用中间80%有限值范围规则。
- smoke 结果不能作为性能证据。
- 不自动覆盖非空 `runs` 目录。
- 代码只保留当前 tracked DAgger、固定评估、RGB 部署和对应性能验证链路；
  历史 checkpoint 与结果目录不作为可继续训练的代码接口。

## 通用像素保护补充测试入口

```powershell
conda run -n barrage python -B tools/evaluate_pixel_guard_refined.py --episodes 3000 --workers 10 --batch-size 10 --search-workers 9
```

此入口默认使用 `diagnostics/risk_removal_20260920/policy_teacher_cost.pt` 和通用
receding 控制器。它与当前窗口 `best.pt` 的匹配实验控制器评估分别记录，成绩不能混用。
默认 3000 局、10 个评估进程、批量 10、9 个搜索线程、CUDA。
完整补充测试每局 120 秒、300 发、10% 定向弹、RGB 输入、零动作延迟。
种子由 `--seed` 确定，输出自动创建独立时间戳目录；该补充测试不参与选模。`--smoke-test` 仅执行 2 局、每局 5 秒的入口检查，
可用 `--smoke-limit-seconds 120` 检查完整单局时长。
