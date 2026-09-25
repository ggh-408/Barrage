"""Plot nonoverlapping total decision latency from an existing window profile."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    root = parser.parse_args().directory.resolve()
    data = json.loads((root / 'latency_breakdown.json').read_text(encoding='utf-8'))
    window = json.loads((root / 'window.json').read_text(encoding='utf-8'))
    labels = {'Tracking and features': '图像跟踪与特征提取',
              'RGB detection': 'RGB 检测', 'RGB capture': 'RGB 图像采集',
              'Detection prediction hints': '检测预测提示',
              'Agent orchestration': '控制器调度', 'Action selection': '动作选择'}
    groups = {}
    for row in data['ranking']:
        stage = row['stage']
        if stage.startswith(('Model ', 'Neural module:', 'Shared geometry:', 'Action query cache')):
            name = '模型推理（含共享几何）'
        elif stage.startswith('Pixel planner:'):
            name = '像素规划与安全检查'
        else:
            name = labels.get(stage, stage)
        groups[name] = groups.get(name, 0.0) + row['exclusive_mean_per_decision_ms']
    groups['其他未分项开销'] = data['uninstrumented_ms'] / data['decisions']
    total = data['total_decision_ms'] / data['decisions']
    assert abs(sum(groups.values()) - total) < 1e-6
    rows = [dict(stage=k, mean_ms=v, percent=100*v/total)
            for k, v in sorted(groups.items(), key=lambda item: item[1], reverse=True)]
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                         'legend.frameon': False})
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    fig.subplots_adjust(left=.27, right=.94, top=.78, bottom=.19)
    fig.suptitle('总决策延迟拆解', fontsize=23, fontweight='bold', y=.96)
    fig.text(.5, .84,
             f"平均总延迟 {total:.3f} ms  |  {data['decisions']} 次决策 / {window['duration_seconds']:.0f} 秒\n"
             f"当前优化版本 · 前台窗口 · 精确忙等待 · 伤害免疫关闭 · 平均 {window['rendered_fps']:.2f} FPS",
             ha='center', fontsize=11, color='#526172')
    bars = ax.barh([r['stage'] for r in rows], [r['mean_ms'] for r in rows],
                   color=['#297DB3', '#7665A7', '#D78C40', '#44A08B', '#749BBC',
                          '#BD8589', '#869A6D', '#92969C', '#ADB5BD'][:len(rows)], height=.65)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{r['mean_ms']:.3f} ms  ({r['percent']:.1f}%)" for r in rows],
                 padding=7, fontsize=11)
    ax.set_xlim(0, max(r['mean_ms'] for r in rows)*1.4)
    ax.set_xlabel('平均耗时（ms / 决策）', fontsize=11)
    ax.tick_params(axis='y', length=0, labelsize=11)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.grid(axis='x', alpha=.17)
    ax.set_axisbelow(True)
    fig.text(.05, .07,
             '各项使用独占时间归并，合计等于总决策延迟；包含计时插桩开销。\n'
             '统计范围为图像采集至控制决策结束；渲染、帧率等待与显示器呈现延迟未计入。',
             fontsize=10, color='#526172')
    fig.savefig(root/'total_decision_latency.png', facecolor='white')
    plt.close(fig)
    (root/'total_decision_latency.json').write_text(json.dumps(
        dict(mean_total_ms=total, stages=rows), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(mean_total_ms=total, stages=rows), ensure_ascii=False))


if __name__ == '__main__':
    main()
