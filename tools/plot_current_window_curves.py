"""Plot measured per-second foreground window FPS and decision latency."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / 'window.json').read_text(encoding='utf-8'))
    frames = json.loads((args.directory / 'frame_pacing.json').read_text(encoding='utf-8'))
    frames = [row for row in frames if 'start' in row and 'frame_ms' in row]
    origin = frames[0]['start']
    fps = np.asarray(report['fps_complete_seconds'], dtype=float)
    # Use completed frames' timestamps; preserve empty decision buckets as gaps.
    elapsed = np.asarray([row['start'] - origin + row['frame_ms'] / 1000 for row in frames])
    latency = np.asarray([row.get('decision_ms', np.nan) if row['decisions'] else np.nan for row in frames])
    means, p95, maxima = [], [], []
    for second in range(len(fps)):
        values = latency[(elapsed >= second) & (elapsed < second + 1) & np.isfinite(latency)]
        means.append(float(np.mean(values)) if len(values) else np.nan)
        p95.append(float(np.percentile(values, 95)) if len(values) else np.nan)
        maxima.append(float(np.max(values)) if len(values) else np.nan)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=160, sharex=True)
    t = np.arange(len(fps)) + .5
    axes[0].plot(t, fps, color='#2563eb', label='每秒实际 FPS')
    axes[0].axhline(report['rendered_fps'], color='#0f766e', linestyle='--',
                   label=f"全程平均 {report['rendered_fps']:.2f} FPS")
    axes[0].axhline(report['render_fps_cap'], color='#94a3b8', linestyle=':', label='渲染上限')
    axes[0].set_ylabel('帧率（FPS）')
    axes[1].plot(t, means, color='#2563eb', label='每秒决策延迟均值')
    axes[1].plot(t, p95, color='#f59e0b', label='每秒决策延迟 P95')
    axes[1].plot(t, maxima, color='#dc2626', alpha=.65, linewidth=.8, label='每秒最大决策延迟')
    axes[1].axhline(report['decision_budget_ms'], color='#64748b', linestyle=':', label='决策预算')
    axes[1].set(xlabel='实际经过时间（秒）', ylabel='RGB 观察到动作处理耗时（ms）', xlim=(0, len(fps)))
    for ax in axes:
        ax.legend(frameon=False, loc='upper right', ncol=2)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    foreground = float(np.mean([row.get('foreground', False) for row in frames]))
    fig.suptitle(f"当前部署窗口实测 · 300 弹 · v53 round1 · {report['duration_seconds']:.1f} 秒\n"
                 f"前台比例 {foreground:.1%} · 默认帧率控制 · 包含游戏内首次调用开销")
    fig.tight_layout()
    fig.savefig(args.directory / 'latency_fps_curves.png')
    plt.close(fig)
    summary = dict(foreground_fraction=foreground, rendered_fps=report['rendered_fps'],
                   decision=report['ai_decision'], over_budget=report['decision_over_budget_count'],
                   fps_min=float(fps.min()), fps_max=float(fps.max()),
                   scope='Decision processing latency, excluding monitor scanout and external input latency')
    (args.directory / 'curve_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
