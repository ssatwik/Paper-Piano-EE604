"""Run deterministic engine variants over labeled MP4 clips."""
import argparse,csv,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'paper-piano'))
from config import preset
from engine import PaperPianoEngine,BaselineEngine
from metrics import summarize,match_events,hold_errors
from scipy.stats import spearmanr

def evaluate_clip(video,gt_path,variant):
    gt=json.loads(gt_path.read_text()); cfg=preset(variant); cfg.mirror=True; cfg.auto_freeze=(variant=='V0')
    engine=BaselineEngine(cfg) if variant=='V0' else PaperPianoEngine(cfg)
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or gt.get('fps',30); predictions=[]; frame_i=0; times=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            t=frame_i/fps; result=engine.process([frame],t); times.append(result.timings_ms.get('total',0))
            if not engine.frozen and frame_i>=30: engine.freeze()
            predictions.extend(result.events); frame_i+=1
    finally: cap.release(); engine.close()
    duration=frame_i/fps; events=gt.get('events',[]); m=summarize(events,predictions,duration)
    match=match_events(events,predictions,.150)
    gt_on=sorted([e for e in events if e.get('type','on')=='on'],key=lambda e:e.get('t_on',e.get('t')))
    pred_on=sorted([e for e in predictions if e.type=='on'],key=lambda e:e.t)
    latencies=[]
    for gi,pi in match.pairs:
        onset=gt_on[gi].get('t_on',gt_on[gi].get('t'))
        latencies.append(pred_on[pi].t-onset)
    m.update(mean_ms=float(np.mean(times)) if times else 0,p95_ms=float(np.percentile(times,95)) if times else 0,
             fps=1000/max(float(np.mean(times)),1e-9),latency_median_ms=float(np.median(latencies)*1000) if latencies else 0,
             latency_p90_ms=float(np.percentile(latencies,90)*1000) if latencies else 0)
    for tol in (.10,.25):
        tm=summarize(events,predictions,duration,tol)
        label=int(tol*1000); m[f'f1_{label}ms']=tm['f1'];m[f'precision_{label}ms']=tm['precision'];m[f'recall_{label}ms']=tm['recall']
    herrors=hold_errors(events,predictions,match)
    m['hold_abs_error_ms']=float(np.median(np.abs(herrors))*1000) if herrors else 0.0
    vel_pairs=[(gt_on[gi]['vel_class'],pred_on[pi].velocity) for gi,pi in match.pairs if gt_on[gi].get('vel_class') is not None]
    m['velocity_spearman']=float(spearmanr([p[0] for p in vel_pairs],[p[1] for p in vel_pairs]).statistic) if len(vel_pairs)>=2 else 0.0
    return m,predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--variants',default='V0,V1,V2,V3');p.add_argument('--split',default='test');p.add_argument('--out',default='results');a=p.parse_args()
    root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True); rows=[]
    for gtpath in sorted(root.rglob('*.gt.json')):
        gt=json.loads(gtpath.read_text())
        if gt.get('split','test')!=a.split: continue
        video=gtpath.with_name(gtpath.name.replace('.gt.json','.mp4'))
        if not video.exists(): continue
        for variant in a.variants.split(','):
            try: metrics,_=evaluate_clip(video,gtpath,variant)
            except RuntimeError as e: print(f'{variant} {video}: {e}',file=sys.stderr); continue
            rows.append({"clip":str(video),"variant":variant,"scenario":gt.get('meta',{}).get('scenario',gtpath.parent.name),
                         "height_cm":gt.get('meta',{}).get('height_cm',''),**metrics})
    fields=['clip','variant','scenario','height_cm','precision','recall','f1','f1_100ms','f1_250ms','fp_per_min','tp','fp','fn','substitutions','latency_median_ms','latency_p90_ms','hold_abs_error_ms','velocity_spearman','mean_ms','p95_ms','fps']
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    lines=['# Evaluation summary','',f'Split: **{a.split}**. Results require real recorded clips and are not generated from synthetic scenes.','', '| Variant | Scenario | Clips | F1 (95% CI) | FP/min (95% CI) | Mean ms |','|---|---|---:|---:|---:|---:|']
    groups={}
    for r in rows: groups.setdefault((r['variant'],r['scenario']),[]).append(r)
    rng=np.random.default_rng(0)
    for (v,s),rs in sorted(groups.items()):
        f1=np.asarray([r['f1'] for r in rs]); fp=np.asarray([r['fp_per_min'] for r in rs])
        f1_boot=np.asarray([np.mean(rng.choice(f1,len(f1),replace=True)) for _ in range(1000)])
        fp_boot=np.asarray([np.mean(rng.choice(fp,len(fp),replace=True)) for _ in range(1000)])
        fci=np.percentile(f1_boot,[2.5,97.5]); pCI=np.percentile(fp_boot,[2.5,97.5])
        lines.append(f"| {v} | {s} | {len(rs)} | {np.mean(f1):.3f} [{fci[0]:.3f}, {fci[1]:.3f}] | {np.mean(fp):.2f} [{pCI[0]:.2f}, {pCI[1]:.2f}] | {np.mean([r['mean_ms'] for r in rs]):.2f} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    if rows:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        scenarios=sorted({r['scenario'] for r in rows}); variants=a.variants.split(',')
        fig,ax=plt.subplots(figsize=(max(6,len(scenarios)*1.5),4)); x=np.arange(len(scenarios)); width=.8/max(1,len(variants))
        for i,v in enumerate(variants): ax.bar(x+(i-(len(variants)-1)/2)*width,[np.mean([r['f1'] for r in rows if r['variant']==v and r['scenario']==s]) if any(r['variant']==v and r['scenario']==s for r in rows) else 0 for s in scenarios],width,label=v)
        ax.set_xticks(x,scenarios);ax.set_ylim(0,1);ax.set_ylabel('F1');ax.legend();fig.tight_layout();fig.savefig(out/'f1_by_scenario.png');plt.close(fig)
        no_touch=[s for s in scenarios if s in ('S3','S4')]
        if no_touch:
            fig,ax=plt.subplots(figsize=(max(5,len(no_touch)*2),4)); nx=np.arange(len(no_touch)); nw=.8/max(1,len(variants))
            for i,v in enumerate(variants):
                vals=[np.mean([r['fp_per_min'] for r in rows if r['variant']==v and r['scenario']==s]) if any(r['variant']==v and r['scenario']==s for r in rows) else 0 for s in no_touch]
                ax.bar(nx+(i-(len(variants)-1)/2)*nw,vals,nw,label=v)
            ax.set_xticks(nx,no_touch);ax.set_ylabel('False triggers / minute');ax.legend();fig.tight_layout();fig.savefig(out/'false_triggers_no_touch.png');plt.close(fig)
        latency=[[r['latency_median_ms'] for r in rows if r['variant']==v and r['tp']>0] for v in variants]
        nonempty=[(v,d) for v,d in zip(variants,latency) if d]
        if nonempty:
            fig,ax=plt.subplots(figsize=(max(5,len(variants)*1.3),4))
            ax.boxplot([d for _,d in nonempty],labels=[v for v,_ in nonempty]);ax.set_ylabel('Median onset latency (ms)');fig.tight_layout();fig.savefig(out/'latency_by_variant.png');plt.close(fig)
        s9=[r for r in rows if r['scenario']=='S9' and r['height_cm']!='']
        if s9:
            fig,ax=plt.subplots(figsize=(5,4))
            for v in variants:
                selected=[r for r in s9 if r['variant']==v]
                if selected: ax.scatter([float(r['height_cm']) for r in selected],[r['fp_per_min'] for r in selected],label=v)
            ax.set_xlabel('Hover height (cm)');ax.set_ylabel('False triggers / minute');ax.legend();fig.tight_layout();fig.savefig(out/'s9_fp_vs_height.png');plt.close(fig)
if __name__=='__main__': main()
