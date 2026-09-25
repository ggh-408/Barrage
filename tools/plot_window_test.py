"""Plot the standalone visible-window test using completed one-second buckets."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    fps = np.asarray(report['fps_complete_seconds'])
    if not len(fps):
        raise ValueError('At least one complete second is required')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    t = np.arange(len(fps)) + .5
    ax.plot(t, fps, color='#2563eb', linewidth=1.5, label='每秒实际帧率')
    ax.axhline(report['rendered_fps'], color='#0f766e', linestyle='--',
               label=f"全程平均 {report['rendered_fps']:.2f} FPS")
    ax.axhline(report['render_fps_cap'], color='#94a3b8', linestyle=':', label='渲染上限')
    ax.set(xlabel='实际经过时间（秒）', ylabel='帧率（FPS）',
           xlim=(0, len(fps)), ylim=(0, max(report['render_fps_cap'], float(fps.max())) * 1.12),
           title=f"当前窗口帧率 — v53 round1 · 300 弹 · CPU · 允许碰撞伤害\n"
                 f"实际 {report['duration_seconds']:.2f} 秒 · 每个点覆盖完整 1 秒 · 包含首次调用开销")
    ax.legend(frameon=False, loc='lower right')
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    output = args.report.with_name('fps_time.png')
    if output.exists():
        raise FileExistsError(output)
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
