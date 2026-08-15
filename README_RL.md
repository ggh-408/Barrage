# Barrage：纯图像 DAgger → QDagger 训练流程

部署策略只接收连续图像。模拟器坐标仅用于训练期教师、场景生成和监督标签，不会输入真实控制器。

## 模型与动作契约

```text
真实/模拟画面 → 4帧语义图 → 子弹对象集合 → Set Attention → GRU
              → 9个动作的 policy/Q 分数 → 动作
```

动作包括 stay、上下左右和四个对角方向。训练与 `Barrage.py` 都会归一化方向向量，因此飞机直线和斜向总速度均为 240 px/s。

## 两阶段流程

第一阶段是短局 DAgger：使用精确教师的标签训练视觉策略，但由学生执行动作来收集学生真正会进入的状态。一个全局调度器严格维持场景比例：

- 60% 核心真实分布：50发、5号、240 px/s、目标弹概率0.35；
- 25% 全范围正则：大小1–7、速度60–300的分层组合；
- 15% 困难组合：大弹、高速和更高目标弹密度。

采样器同时覆盖真实边缘出生、可恢复中局和动力学验证过的困难中局。Replay分别给开局、失败尾、普通和困难场景保留配额。第一阶段不训练或启用多时域安全过滤头，避免尚未校准的安全预测干扰策略学习；安全头留给第二阶段训练。

新一轮命令（必须使用新的空目录；以下 `r2` 只是新的 run 名称）：

```cmd
conda activate barrage && cd /d C:\Users\pc\Desktop\python\Barrage && python -m barrage_rl.train_visual_set --output-dir runs\visual_set_v9 --initial-checkpoint runs\visual_set_v8\best.pt --rounds 10 --samples-per-round 70000 --replay-capacity 350000 --epochs-per-round 8 --collection-episode-seconds 12 --evaluation-episodes 50 --device cuda
```

第二阶段 QDagger 从第一阶段 `best.pt` 开始，训练长时程价值、失败尾和安全头，并用固定的独立核心任务评估集选模：

```cmd
conda activate barrage && cd /d C:\Users\pc\Desktop\python\Barrage && python -m barrage_rl.train_qdagger runs\visual_set_v9\best.pt --output-dir runs\visual_set_v9_qdagger --evaluation-episodes 100 --evaluation-workers 8 --device cuda
```

训练器拒绝覆盖非空 run。每次实验都应使用新目录；checkpoint、配置、CSV和图使用原子写入，并记录源 checkpoint 哈希。

## 第一轮基准

第一阶段第一轮已经用100个固定盲种子复核。在相同120秒核心任务上，结果为：mean 50.87 s、IQM 42.85 s、P10 3.19 s、120秒通过率16%。按60秒截断后 IQM 39.34 s；完整训练到2M步的 `visual_set_v8_qdagger_s3` 为41.39 s，差距4.9%，因此第一轮没有结构性崩溃。第一轮P10比 s3 的1.57 s更高，但120秒通过率仍明显不足，必须继续第二阶段。

## 验收顺序

1. 先运行全部单元测试；smoke只验证链路，不代表模型性能。
2. 每轮都在从未用于采集/训练的固定核心种子上评估；不能用训练准确率判断可部署性。
3. 先看 `<10s`、P10/CVaR5，再看60秒和120秒通过率、IQM。
4. 候选模型先做1000个盲种子筛选；要声明99%通过率，建议至少10000局盲测并报告置信区间。

## 真实游戏与并行盲测

```cmd
python Barrage.py --ai-checkpoint runs\visual_set_v9_qdagger\best.pt --bullets 50 --bullet-size 5 --bullet-speed 240 --targeted-probability 0.35 --skin 0 --no-music
```

```cmd
python tools\evaluate_barrage_parallel.py runs\visual_set_v9_qdagger\best.pt --episodes 1000 --limit-seconds 120 --workers 8 --bullets 50 --bullet-size 5 --bullet-speed 240 --targeted-probability 0.35 --output runs\visual_set_v9_eval\evaluation_episodes.csv
```

评估 manifest 会绑定 checkpoint 哈希和完整任务参数，避免把不同实验续写进同一个 CSV。
