"""Summarize matched CPU/CUDA window reports with the existing FPS binning."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tmp/matplotlib'))
import matplotlib.pyplot as plt
import numpy as np
from tools.compare_visible_fps import _check_comparable, _load_report, _summary
from tools.report_helpers import _write_json


def timing_summary(data):
    result = _summary(data)
    report = data['report']
    with data['path'].with_suffix('.frames.csv').open(encoding='utf-8', newline='') as stream:
        frames = list(csv.DictReader(stream))
    times = np.array([float(row['elapsed_seconds']) for row in frames])
    steps = np.array([int(row['physics_steps']) for row in frames])
    lag = times - np.cumsum(steps) / report['physics_fps']
    result.update(
        ai_device=report['ai_device'],
        actual_decision_hz=result['decision_count'] / result['duration_seconds'],
        game_speed_ratio=result['alive_physics_steps'] / report['physics_fps'] / result['duration_seconds'],
        summed_physics_steps=int(steps.sum()),
        all_scheduled_decisions_executed=result['decision_count'] == int(steps.sum()) // round(report['physics_fps'] / report['decision_hz_target']),
        frame_completion_intervals_above_250ms=int(np.count_nonzero(np.diff(times) > 0.25)),
        sampled_wall_minus_game_lag_final_ms=float(lag[-1] * 1000),
        sampled_wall_minus_game_lag_max_ms=float(lag.max() * 1000),
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix != '.png':
        parser.error('output must end in .png')
    for path in (output, output.with_suffix('.json'), output.with_suffix('.csv')):
        if path.exists():
            raise FileExistsError(path)
    cpu, cuda = _load_report(args.cpu), _load_report(args.cuda)
    _check_comparable(cpu, cuda)
    assert cpu['report']['ai_device'] == 'cpu'
    assert cuda['report']['ai_device'] == 'cuda'
    assert cpu['report']['torch_threads'] == cuda['report']['torch_threads'] == 10
    summaries = {'cpu': timing_summary(cpu), 'cuda': timing_summary(cuda)}
    seconds = min(cpu['complete_seconds'], cuda['complete_seconds'])
    a, b = summaries['cpu'], summaries['cuda']
    result = {
        'kind': 'cpu_cuda_visible_window_comparison',
        'cpu': a, 'cuda': b,
        'cuda_relative_to_cpu': {
            'fps_percent': (b['average_fps'] / a['average_fps'] - 1) * 100,
            'decision_mean_ms_delta': b['decision_mean_ms'] - a['decision_mean_ms'],
            'decision_mean_percent': (b['decision_mean_ms'] / a['decision_mean_ms'] - 1) * 100,
            'decision_p99_ms_delta': b['decision_p99_ms'] - a['decision_p99_ms'],
        },
        'notes': [
            'One sequential 120-second run per device, same seed and unchanged production source.',
            'Both window runs use immunity and do not measure survival acceptance.',
            'Decision timing includes CPU preprocessing, model execution and action synchronization.',
            'Wall-minus-game lag includes end-of-frame work and the substep remainder; it is not discarded time.',
            'Intervals are display completion timing; physical monitor output latency is not measured.',
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['start_seconds', 'cpu_fps', 'cuda_fps'])
        writer.writerows((i, int(cpu['fps'][i]), int(cuda['fps'][i])) for i in range(seconds))
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=150)
    fig.patch.set_facecolor('#f6f8fc')
    x = np.arange(seconds) + 0.5
    for data, label, color in ((cpu, 'CPU', '#2563eb'), (cuda, 'CUDA', '#d97706')):
        mean = data['report']['rendered_fps']
        ax.plot(x, data['fps'][:seconds], color=color, linewidth=1.6, label=f'{label}（平均 {mean:.2f} FPS）')
        ax.axhline(mean, color=color, linestyle=':', linewidth=1.2)
    ax.axhline(100, color='#94a3b8', linestyle='--', linewidth=1.2, label='渲染上限 100 FPS')
    ax.set(xlim=(0, seconds), ylim=(0, 115), xlabel='现实经过时间（秒）', ylabel='帧率（FPS）')
    ax.set_xticks(np.arange(0, seconds + 1, 15))
    ax.grid(axis='y', color='#e2e8f0')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', frameon=False)
    fig.suptitle('Barrage 最优权重：CPU / CUDA 窗口对照', x=0.08, ha='left', fontsize=18, fontweight='bold')
    ax.set_title(f"v42 · batch 1 · {cpu['report']['bullets']} 发子弹 · 10% 瞄准弹 · {cpu['report']['physics_fps']} Hz 物理 · {cpu['report']['decision_hz_target']:g} Hz 同步决策", loc='left', fontsize=10, pad=15)
    fig.text(0.08, 0.055, f"决策均值 CPU {a['decision_mean_ms']:.2f} / CUDA {b['decision_mean_ms']:.2f} ms    P99 {a['decision_p99_ms']:.2f} / {b['decision_p99_ms']:.2f} ms", fontsize=10)
    fig.text(0.08, 0.02, '同种子顺序运行；按完整一秒统计 display.flip 完成次数。免伤计时不构成避弹通过率。', fontsize=9, color='#64748b')
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.19)
    fig.savefig(output)
    plt.close(fig)
    _write_json(output.with_suffix('.json'), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
