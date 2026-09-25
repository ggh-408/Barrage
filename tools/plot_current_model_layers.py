"""Draw current module and leaf-layer timings without double counting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory.resolve()
    report=json.loads((root/'window.json').read_text(encoding='utf-8'))
    data=json.loads((root/'latency_breakdown.json').read_text(encoding='utf-8'))
    inventory={r['name']:r for r in data['model_structure']}
    ranks={r['stage']:r for r in data['ranking']}
    colors={'Linear':'#2878B5','GELU':'#46A887','LayerNorm':'#9474BA','MultiheadAttention':'#DF9252'}
    leaves=[];uncalled=[]
    for name,module in inventory.items():
        kind=module['type']
        if kind=='NonDynamicallyQuantizableLinear':kind='Linear'
        if kind not in colors:continue
        row=ranks.get('Neural module: '+name)
        if row is None:
            uncalled.append(dict(name=name,type=kind,reason='包含在 MultiheadAttention 内部计算中' if '.attention.out_proj' in name else '动作查询缓存复用'))
        else:
            leaves.append(dict(name=name,type=kind,mean_ms=row['exclusive_mean_per_decision_ms'],p95_ms=row['exclusive_p95_ms']))
    leaves.sort(key=lambda r:r['mean_ms'],reverse=True)
    components=[]
    labels={
        'Model geometry: prepare_action_geometry':'共享几何准备（已优化）',
        'Model geometry: action_geometry_features_from_shared':'几何统计特征',
        'Action query cache':'动作查询缓存',
        'Neural module: object_encoder':'对象编码器',
        'Neural module: global_encoder':'全局编码器',
        'Neural module: geometry_encoder':'几何编码器',
        'Neural module: cross_attention.0':'交叉注意力块 0（含 FFN / Norm）',
        'Neural module: cross_attention.1':'交叉注意力块 1（含 FFN / Norm）',
        'Neural module: policy_head':'策略输出头',
        'Neural module: teacher_cost_head':'教师成本输出头',
    }
    for key,label in labels.items():
        if key in ranks:components.append(dict(name=label,mean_ms=ranks[key]['inclusive_mean_per_decision_ms']))
    components.sort(key=lambda r:r['mean_ms'],reverse=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'legend.frameon':False})
    fig,axes=plt.subplots(2,1,figsize=(13,14),gridspec_kw={'height_ratios':[len(components)+2,len(leaves)+2]})
    fig.subplots_adjust(left=.38,right=.92,top=.91,bottom=.12,hspace=.28)
    model_ms=ranks['Model forward']['inclusive_mean_per_decision_ms']
    fig.suptitle('当前模型逐层耗时',fontsize=22,fontweight='bold',y=.99)
    fig.text(.5,.943,f"v53 / round1 · CPU · 384 槽位 · 192 维 · 2 层 / 4 头\n前台实测 {report['duration_seconds']:.0f} s / {data['decisions']} 次决策  |  模型均值 {model_ms:.3f} ms（含共享几何）",ha='center',fontsize=11,color='#4A5664')
    for ax,rows,title in ((axes[0],components,'模块整体耗时 · 包含子层'),(axes[1],leaves,'具体层耗时 · 独占时间，按平均耗时排序')):
        y=list(range(len(rows)))
        names=[r['name']+(f"  [{r['type']}]" if 'type' in r else '') for r in rows]
        values=[r['mean_ms'] for r in rows]
        bars=ax.barh(y,values,height=.68,color=[colors.get(r.get('type'),'#647F99') for r in rows])
        ax.set_yticks(y,names,fontsize=9)
        ax.invert_yaxis();ax.set_xlim(0,max(values)*1.22)
        ax.bar_label(bars,labels=[f'{v:.4f}' for v in values],padding=5,fontsize=9)
        ax.set_title(title,loc='left',fontsize=13,pad=13,fontweight='bold')
        ax.set_xlabel('平均耗时（ms / 决策）',fontsize=10)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
    axes[1].legend(handles=[Patch(color=c,label=t) for t,c in colors.items()],frameon=False,
                   loc='lower right',fontsize=9)
    fig.text(.04,.075,'口径：上图与下图不可相加；模块整体时间包含其子层。计时包含插桩开销。\nMultiheadAttention 包含 Q/K/V 与输出投影；未独立调用的子层不填 0。',fontsize=10,color='#4A5664')
    fig.text(.04,.027,'缓存复用：action_vector_encoder.0 / .1 / .2；action_embedding。\n内部调用：cross_attention.0.attention.out_proj、cross_attention.1.attention.out_proj。',fontsize=9,color='#647080')
    fig.savefig(root/'current_model_layers.png',dpi=180,facecolor='white')
    plt.close(fig)
    (root/'current_model_layers.json').write_text(json.dumps(dict(components=components,layers=leaves,not_separately_timed=uncalled,model_mean_ms=model_ms),indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# 当前模型逐层计时图','',f"{data['decisions']} 次决策，模型含共享几何平均 {model_ms:.4f} ms。",'',
           '| 层 | 类型 | 平均独占 ms | P95 独占 ms |','|---|---|---:|---:|']
    for r in leaves:lines.append(f"| {r['name']} | {r['type']} | {r['mean_ms']:.6f} | {r['p95_ms']:.6f} |")
    lines+=['','缓存复用或内部调用的层缺少独立计时，不赋予虚构的零耗时。模块整体耗时与子层不可相加。计时包含逐层插桩开销。',
            f"本轮逐层插桩测试出现 {report['decision_over_budget_count']} 次决策超预算，最大决策耗时 {report['ai_decision']['max_ms']:.2f} ms；所有样本均保留在统计中。", '',
            f"![逐层耗时]({root.as_posix()}/current_model_layers.png)"]
    (root/'current_model_layers.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(image=str(root/'current_model_layers.png'),timed_layers=len(leaves),not_separately_timed=uncalled,model_mean_ms=model_ms),ensure_ascii=False))


if __name__=='__main__':main()
