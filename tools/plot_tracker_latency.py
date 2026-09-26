"""Plot mutually exclusive tracker blocks from a sampled current-version run."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1]).resolve()
report = json.loads((root/'tracker_blocks.json').read_text(encoding='utf-8'))
names = {
 'normalize_detections':'检测坐标标准化', 'plane_motion':'飞机运动估计',
 'initial_tracks':'初始轨迹建立', 'predict_positions':'轨迹位置预测',
 'distance_matrix_and_gates':'距离矩阵与关联门限', 'mutual_nearest_pairs':'互为最近邻关联',
 'sort_sparse_pairs':'稀疏候选配对排序', 'empty_detections':'空检测处理',
 'build_assignments':'构造匹配列表', 'matched_track_updates':'匹配轨迹状态更新',
 'occlusion_and_missed_tracks':'遮挡与漏检轨迹处理', 'new_tracks':'新轨迹建立',
 'group_histories':'速度拟合：历史分组', 'regression_basis':'速度拟合：回归基底',
 'history_to_array':'速度拟合：历史数组构造', 'regression_and_norm':'速度拟合：批量回归与范数',
 'normalize_and_store_velocities':'速度拟合：归一化与写回',
 'stable_retention_sort':'轨迹保留排序', 'retention_arrays':'保留评估：数组构造',
 'retention_validation':'保留评估：有效性检查', 'retention_geometry_and_scores':'保留评估：几何威胁评分',
 'allocate_feature_buffers':'特征缓冲区分配', 'stack_track_arrays':'特征：轨迹数组组装',
 'relative_motion_ttc_clearance':'特征：相对运动与碰撞几何', 'track_metadata_arrays':'特征：轨迹元数据组装',
 'threat_sort':'特征：威胁排序', 'pack_object_features':'对象特征打包', 'global_features':'全局特征构造'}
rows=[]
residual=0.0
for r in report['ranking']:
    if r['block'].startswith('method:'):
        residual+=r['mean_exclusive_ms']
    else:
        rows.append(dict(stage=names.get(r['block'].split('.')[-1],r['block']),
                         mean_ms=r['mean_exclusive_ms'],block=r['block']))
rows.append(dict(stage='其他方法内部开销',mean_ms=residual))
accounted=sum(r['mean_ms'] for r in rows)
total=report['sampled_mean_ms']
assert total>=accounted-1e-6
rows.append(dict(stage='计时器管理及外层未分项开销',mean_ms=total-accounted))
rows.sort(key=lambda r:r['mean_ms'],reverse=True)
for r in rows:r['percent']=100*r['mean_ms']/total
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'legend.frameon':False})
fig,ax=plt.subplots(figsize=(12,12),dpi=180)
fig.subplots_adjust(left=.31,right=.95,top=.85,bottom=.12)
fig.suptitle('图像跟踪与特征提取 · 延迟拆解',fontsize=22,fontweight='bold',y=.975)
fig.text(.5,.905,f"当前优化版本 · 120 秒前台测试 · {report['sampled_updates']} 次代码块抽样\n"
         f"抽样更新平均 {total:.3f} ms  |  未插桩更新平均 {report['unsampled_mean_ms']:.3f} ms",
         ha='center',fontsize=11,color='#526172')
palette=['#297DB3','#7665A7','#D78C40','#44A08B','#749BBC','#BD8589','#869A6D']
bars=ax.barh([r['stage'] for r in rows],[r['mean_ms'] for r in rows],height=.66,
             color=['#A3A7AE' if '开销' in r['stage'] else palette[i%len(palette)] for i,r in enumerate(rows)])
ax.invert_yaxis()
ax.bar_label(bars,labels=[f"{r['mean_ms']:.3f} ms  ({r['percent']:.1f}%)" for r in rows],padding=6,fontsize=9)
ax.set_xlim(0,max(r['mean_ms'] for r in rows)*1.4)
ax.set_xlabel('平均耗时（ms / 抽样更新）',fontsize=11)
ax.tick_params(axis='y',length=0,labelsize=10)
ax.spines[['top','right','left']].set_visible(False)
ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
fig.text(.045,.045,'各项采用独占耗时，合计等于抽样更新平均延迟；占比以抽样总耗时为分母。\n'
         '每 10 次更新抽样一次，细分计时会增加开销；此图与上一轮总延迟图属于独立采样。',
         fontsize=10,color='#526172')
fig.savefig(root/'tracker_latency_breakdown.png',facecolor='white');plt.close(fig)
(root/'tracker_latency_breakdown.json').write_text(json.dumps(dict(mean_total_ms=total,stages=rows),ensure_ascii=False,indent=2),encoding='utf-8')
print(str(root/'tracker_latency_breakdown.png'))
