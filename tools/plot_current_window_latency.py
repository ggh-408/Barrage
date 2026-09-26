"""Plot existing visible-window measurements without changing the runtime."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = Path(sys.argv[1])
r = json.loads((p/'window.json').read_text(encoding='utf-8'))
b = json.loads((p/'latency_breakdown.json').read_text(encoding='utf-8'))
d = json.loads((p/'decision_timings.json').read_text(encoding='utf-8'))
with (p/'window.frames.csv').open(encoding='utf-8', newline='') as f:
    frames = list(csv.DictReader(f))
plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False})
fig, axes = plt.subplots(3, 1, figsize=(12, 13), dpi=150)
fps = np.asarray(r['fps_complete_seconds'])
t = np.arange(len(fps))+1
axes[0].plot(t, fps, color='#93b8eb', label='每秒实际渲染帧率')
axes[0].plot(t, fps.cumsum()/t, color='#2563eb', label='累计平均渲染帧率')
axes[0].axhline(r['rendered_fps'], color='#dc6834', ls='--', label=f"全程平均 {r['rendered_fps']:.2f} FPS")
axes[0].set(xlabel='实际时间（秒）', ylabel='FPS', title='当前 best.pt · 300 弹 · 可见窗口实测')
v = np.array([x['total_ms'] for x in d])
x = np.arange(len(v))+1
axes[1].plot(x, v, color='#a2b8d5', lw=.7, label='逐次读取图像至输出决策')
axes[1].plot(x, v.cumsum()/x, color='#2563eb', label='累计平均耗时')
axes[1].axhline(r['decision_budget_ms'], color='#dc6834', ls='--', label='决策周期预算')
axes[1].set(xlabel='决策序号（未记录单次决策时间戳）', ylabel='耗时（毫秒）')
names = {'RGB capture':'读取 RGB 图像','RGB detection':'图像检测','Tracking and features':'目标跟踪与特征', 'Detection prediction hints':'检测预测提示','Agent orchestration':'决策调度', 'Model forward':'模型前向其余计算','Pixel planner: apply':'像素规划调度','Pixel planner: _geometry':'规划几何构造','Pixel planner: _assess':'路径安全评估','Pixel planner: search':'像素路径搜索','Action selection':'动作选择'}
rank = [dict(z) for z in b['ranking']]
rank.append(dict(stage='其余未细分及计时开销', exclusive_mean_per_decision_ms=b['uninstrumented_ms']/b['decisions'],percent_of_decision_time=100*b['uninstrumented_ms']/b['total_decision_ms']))
rank.sort(key=lambda z:z['exclusive_mean_per_decision_ms'],reverse=True)
with (p/'stage_ranking.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f); w.writerow(['环节','平均独占耗时_ms','占比_percent'])
    for z in rank:w.writerow([names.get(z['stage'],z['stage']),z['exclusive_mean_per_decision_ms'],z['percent_of_decision_time']])
shown=[z for z in rank if z['exclusive_mean_per_decision_ms']>=.01]
ax=axes[2]
bars=ax.barh([names.get(z['stage'],z['stage']).replace('Neural module: ','神经网络：') for z in shown], [z['exclusive_mean_per_decision_ms'] for z in shown], color='#447ab5')
ax.invert_yaxis(); ax.bar_label(bars,fmt='%.3f ms',padding=4)
ax.set(xlabel='每次决策平均独占耗时（毫秒）',title='耗时降序：父调用扣除已计时子调用')
ax.set_xlim(0,max(z['exclusive_mean_per_decision_ms'] for z in shown)*1.3)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x' if ax is axes[2] else 'y',alpha=.18)
    if ax is not axes[2]:ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(p/'latency_fps.png'); plt.close(fig)
print(json.dumps({'fps':r['rendered_fps'],'ai':r['ai_decision'],'alive':r['alive_at_end'],'ranking':rank},ensure_ascii=False,indent=2))
