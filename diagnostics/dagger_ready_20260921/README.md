# DAgger 定向训练交付包

本目录交付启动入口和预检证据。准备过程不执行优化器更新，不创建正式训练运行；训练由用户手动启动。

## 手动启动

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\diagnostics\dagger_ready_20260921\start_training.ps1
```

只查看计划、不训练：

```powershell
powershell -ExecutionPolicy Bypass -File .\diagnostics\dagger_ready_20260921\start_training.ps1 -PlanOnly
```

也可在 `barrage` 环境中使用 `python -B tools/train_targeted_dagger.py --train`。直接运行 Python 入口而不提供 `--train` 只显示计划。

## 已准备的配置

- 输出：`runs/visual_set_v52`。非空目录会拒绝启动，避免覆盖历史结果。
- 初始权重：根目录 `best.pt`；重新创建优化器，不恢复旧风险头、旧优化器或旧选模成绩。
- 1 个 DAgger round，144,000 个采集状态，最多 2 个 epoch，学习率 `1e-5`，batch 512，回放容量 160,000。
- 36 个环境、9 个采集进程。9 个环境固定重访已知困难种子，27 个环境采用新种子；使用当前学生及候选规划器采集，不切回教师行为。
- 复用现有 `behavior_regret` 优先采样与死亡前样本权重，避免把所有动作分歧都视为错误。
- 保持 300 弹、逐弹独立 10% 定向概率、384 槽、教师反应 0.10 秒、RGB 输入、零额外动作延迟、120 秒完整时限。
- 使用上轮独立规划器的并行评分实现。训练采集和轮次评估由同一个进程级安装器接入，实测默认控制器保持原样。
- 保持初始权重的特征归一化口径；250 的弹数归一化分母属于当前权重输入定义，实际训练和评估环境均为 300 弹。此口径会记录在配置与 checkpoint 中。

144,000 个状态覆盖 36 个环境各至少一个完整 120 秒回合的采集预算；避免把仅开局的短数据当成正式定向训练。

## 正式训练启动后的评估量

手动启动后先进行 **200 局初始评估**，训练后再在 **同一固定 200 局**上评估。它们不会在准备测试期间自动运行。正式过程可能耗时较长，当前少量测试的耗时不能直接当成训练完成时间。

`best.pt` 只按 `success_at_limit` 选择，任何同分均选择最新权重。每次评估要求 200/200 才达到正式验收目标；准备阶段的 20 局测试不能替代这项要求。

## 实验权重的使用

初始权重与训练生成的权重包含实验控制器标记。预配置入口检查标记并安装匹配规划器；控制器不匹配或含其他复合决策模块的权重会被拒绝。

训练结束后，使用专用入口进行固定 200 局评估：

```powershell
conda activate barrage
python -B tools/train_targeted_dagger.py --evaluate runs/visual_set_v52/best.pt
```

评估输出为 `runs/visual_set_v52_eval`。实验权重暂不替换实测默认权重。

## 预检边界

预检使用现有采集、损失和并行评估实现，检查并发进程退出、输入形状、有限损失/梯度、权重未变、实验 checkpoint 的控制器绑定、固定 200 配置约束、持平选模和目录保护。测试阶段将优化器更新函数设为拒绝执行。

完整测试为 20 局。另使用 36 个环境各采集几个决策步测试采集链路，包含与完整测试重叠的 9 个困难种子，总计涉及 47 个不同种子，不扩大测试池。

查看 `preflight.json`、`bootstrap_check.json` 和 `evaluation/evaluation_episodes.csv` 获取最终证据。测试通过后，启动入口会核对代码与准备时保存的直接内容；代码改变时需要重新验证。

当前没有正式训练产物，也没有后台自动启动训练的任务。

预检先写入独立临时目录，全部通过后才更新验证记录、`initial_model_reference.pt` 与 `launch_source_snapshot`。保留评估配置、完整种子结果 CSV 和汇总；临时模型、manifest 测试副本、中间进度及缓存在验证结束后清理。

重新预检时，在 `barrage` 环境使用一个全新的临时输出目录，显式指定原来的种子 CSV：

```powershell
python -B diagnostics/dagger_ready_20260921/preflight.py --output-dir .tmp/prepared_preflight --seeds-csv diagnostics/dagger_ready_20260921/evaluation/evaluation_episodes.csv
python -B diagnostics/dagger_ready_20260921/bootstrap_check.py --output-dir .tmp/prepared_preflight
```

上述命令生成候选验证记录，不自动覆盖本目录证据。两步均通过后才能替换生效记录和启动快照；任一步失败时保留原记录，并继续阻止使用过期证据启动。
