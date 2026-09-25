"""Measure the real visible Barrage window with the receding image guard."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--threads',type=int,default=10)
    parser.add_argument('--seed',type=int,default=2131058160)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.seconds<=0 or args.threads<=0: parser.error('seconds and threads must be positive')
    out=args.output or ROOT/'diagnostics'/f'visible_window_refined_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(parents=True,exist_ok=False)
    os.environ['SDL_VIDEODRIVER']='windows'
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
    import pygame
    import torch
    import numpy as np
    from barrage_rl.live_screen import LiveVisualController
    from barrage_rl.artifacts import sha256_file
    from tools.pixel_guard_receding import install_receding_guard
    checkpoint=ROOT/'diagnostics/v46_recovery_threshold_candidate/threshold_0.18.pt'
    original_init=LiveVisualController.__init__
    original_observe=LiveVisualController.observe_due_surface
    rows=[]
    inference=[]
    guards=[]
    controller_metadata={}
    def initialize(self,*a,**kw):
        torch.set_num_threads(args.threads)
        original_init(self,str(checkpoint),**kw)
        if self.action_delay_steps!=0: raise RuntimeError('This measurement requires the synchronous checkpoint')
        guards.append(install_receding_guard(self.agent))
        controller_metadata.update(device='cpu',threads=torch.get_num_threads(),action_delay_steps=self.action_delay_steps)
        original_act=self.agent.act_features
        def act(*x,**y):
            started=time.perf_counter()
            result=original_act(*x,**y)
            inference.append((time.perf_counter()-started)*1000)
            return result
        self.agent.act_features=act
    def observe(self,surface):
        started=time.perf_counter()
        result=original_observe(self,surface)
        rows.append((started,(time.perf_counter()-started)*1000,inference[-1],int(result)))
        return result
    sys.argv=[str(ROOT/'Barrage.py'),'--ai','--bullets','300','--targeted-probability','0.10','--seed',str(args.seed),'--no-music','--latency-test-seconds',str(args.seconds),'--latency-output',str(out/'window.json')]
    with patch.object(LiveVisualController,'__init__',initialize),patch.object(LiveVisualController,'observe_due_surface',observe):
        runpy.run_path(str(ROOT/'Barrage.py'),run_name='__main__')
    if not rows: raise RuntimeError('No visible-window decisions recorded')
    with (out/'decisions.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['elapsed_seconds','image_decision_ms','model_guard_ms','action'])
        writer.writerows((t-rows[0][0],d,m,a) for t,d,m,a in rows)
    report=json.loads((out/'window.json').read_text())
    report.update(checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint),controller=controller_metadata,guard=guards[0].manifest(),seed=args.seed,video_driver='windows',note='Decision timestamps start at first scheduled decision; collision immunity is enabled for latency measurement.')
    (out/'measurement.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (out/'window.frames.csv').open() as stream: frames=list(csv.DictReader(stream))
    t=np.array([r[0]-rows[0][0] for r in rows]);d=np.array([r[1] for r in rows]);m=np.array([r[2] for r in rows])
    ft=np.array([float(r['elapsed_seconds']) for r in frames]);fp=np.array([float(r['frame_processing_ms']) for r in frames]);sim=np.cumsum([int(r['physics_steps']) for r in frames])/120
    fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    axes[0].plot(t,d,lw=.55,alpha=.65,label='RGB capture + detection + model + guard')
    axes[0].plot(t,m,lw=.55,alpha=.65,label='Model + guard')
    axes[0].axhline(1000/30,color='red',ls='--',lw=1,label='33.33 ms decision budget')
    axes[0].set(ylabel='Decision latency (ms)',title='Visible Barrage window | 300 bullets | best checkpoint + receding guard | CPU '+str(args.threads)+' threads')
    axes[1].plot(ft,fp,lw=.5,color='#228866',label='Frame processing (includes frame pacing)')
    axes[1].set(ylabel='Frame latency (ms)')
    axes[2].plot(ft,sim,label='Simulated game time');axes[2].plot(ft,ft,'--',label='Wall-clock reference')
    axes[2].set(xlabel='Elapsed wall-clock time (s)',ylabel='Time (s)')
    for i,ax in enumerate(axes):
        ax.grid(alpha=.2)
        ax.legend(loc='upper right' if i<2 else 'lower right')
    fig.savefig(out/'latency_curve.png',dpi=160)
    plt.close(fig)
    print('OUTPUT:',out,flush=True)


if __name__=='__main__': main()
