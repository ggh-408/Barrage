# Barrage：300 弹纯图像躲避

一个支持手动操作与 AI 控制的弹幕躲避项目。AI 从游戏图像中识别并跟踪子弹，结合动作策略与像素规划选择躲避路线，推理过程只接收图像。

任务使用 300 发子弹，每颗新生成或补生的子弹独立按 10% 概率生成定向弹，概率不随弹数变化，单局比例允许波动。

## 实测表现

以下测试采用上述全局任务参数。

| 测试 | 测试范围 | 结果 |
|---|---|---|
| 实时窗口帧率 | CPU 推理、前台窗口、关闭免伤，持续 120 秒 | 平均 **99.79 FPS**，每秒 **99–100 FPS**，全程存活 |
| AI 决策耗时 | 同次窗口测试，共 3600 次决策 | 平均 **8.30 ms**，P95 **10.03 ms**，最大 **11.64 ms** |
| 已完成长测 | 3000 局，每局上限 120 秒 | **3000/3000** 局达到上限；95% 置信区间下限为 **99.87%** |

窗口使用的 [best.pt](best.pt) 与已完成长测保存的模型参数经直接比较一致。9 月 25 日的 3000 局长测采用 `optimized` 评估运行时、`window` 跟踪器、匹配的窗口控制器和名义像素安全排序，详见[实验配置](diagnostics/targeted_dagger_fresh_20260925_224928_265538/experiment_manifest.json)、[评估配置](diagnostics/targeted_dagger_fresh_20260925_224928_265538/evaluation/evaluation_config.json)及[评估汇总](diagnostics/targeted_dagger_fresh_20260925_224928_265538/evaluation/evaluation_summary.json)。后续版本 12 权重校验及训练兼容代码清理不改变该权重的优化评估决策路径；实时窗口性能以单独的窗口测试为准。

## 使用入口

依赖版本由 [requirements.txt](requirements.txt) 固定。

| 用途 | 入口 | 说明 |
|---|---|---|
| 游戏窗口 | [Barrage.py](Barrage.py) | `--ai` 启用 AI，`--no-ai` 使用手动控制；不传模式参数时显示设置界面 |
| 模型训练 | [train_tracked_policy.py](barrage_rl/train_tracked_policy.py) | 通用 DAgger 训练；`--resume` 恢复，`--rounds` 设置总轮数 |
| 定向长测 | [evaluate_targeted_dagger.py](tools/evaluate_targeted_dagger.py) | 默认加载窗口权重；`--episodes` 设置局数，支持 fresh 与显式种子回归测试 |
| 窗口测速 | [test_visible_window.py](tools/test_visible_window.py) | `--seconds` 设置时长，`--no-ai` 测手动模式；默认关闭免伤，`--immune` 开启免伤 |

评估默认使用 `--runtime optimized`，复用窗口规划与跟踪实现；`--runtime reference` 使用历史长测实现。训练恢复、预配置实验及其他评估入口见[技术说明](docs/technical.md)。

预配置训练入口 `tools/train_targeted_dagger.py` 以根目录 `best.pt` 为初始权重，重新创建优化器。启动前校验当前源码、权重及预检记录；替换权重或修改源码后须重新预检。训练产物写入独立运行目录，不自动覆盖根目录权重。

## 工作原理

```text
当前图像 → 持久 tracker → 轨迹 token 与全局图像特征
  → 动作 query + 共享动作几何 + cross-attention
  → 动作策略头 / 教师代价辅助头
  → 策略动作 → 图像几何检查与 receding 像素规划
  → 每个决策周期输出一个最终动作
```

模型结构版本为 12，仅训练动作策略头和教师代价辅助头；保留共享动作几何、
384 槽图像 tracker 及 0.5 秒 receding 像素规划。训练、评估、窗口默认均为 15 个决策步。默认关闭独立解析约束。

定向长测入口和窗口统一使用 `commit_safe_ranking`：有可覆盖下一步
33.3 毫秒的名义像素安全路线时，只在这些路线中排序；全部不满足时优先选择名义安全前缀最长的路线。

窗口统一使用 `tick_busy_loop` 精确限帧（100 FPS）。AI 和手动模式共享游戏内核，窗口在 CPU 上执行单帧推理；历史信息由图像跟踪和规划维护。

## 注意事项

- 窗口、训练、评估及蒸馏入口仅接受明确标记为版本 12 的权重。新增训练轮次不会自动替换窗口权重。
- 训练仅按 `success_at_limit` 选择最优权重，任何同分情况均选择最新 checkpoint；其他指标只用于报告。
- 评估范围按任务确定，测速和 smoke 检查不自动触发完整长测，也不提供长期成功率结论。
- 实现细节与实验操作见[技术说明](docs/technical.md)，开发约束和绘图规范见 [AGENTS.md](AGENTS.md)。
