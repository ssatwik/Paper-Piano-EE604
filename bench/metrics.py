"""Deterministic event matching and clip-level piano metrics."""
from dataclasses import dataclass
import numpy as np


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    substitutions: int
    pairs: list
    precision: float
    recall: float
    f1: float


def _get(obj, key): return obj[key] if isinstance(obj,dict) else getattr(obj,key)


def match_events(ground_truth, predictions, tol=.150):
    gt=sorted([e for e in ground_truth if e.get("type","on")=="on"],key=lambda e:e["t_on"] if "t_on" in e else e["t"])
    pred=sorted([e for e in predictions if _get(e,"type")=="on"],key=lambda e:_get(e,"t"))
    subs=0
    unmatched=set(range(len(pred))); pairs=[]
    for gi,g in enumerate(gt):
        gt_t=g.get("t_on",g.get("t")); key=g["key"]
        cand=[(abs(_get(pred[pi],"t")-gt_t),pi) for pi in unmatched if _get(pred[pi],"key")==key and abs(_get(pred[pi],"t")-gt_t)<=tol]
        if cand:
            _,pi=min(cand); unmatched.remove(pi); pairs.append((gi,pi))
    tp=len(pairs); fn=len(gt)-tp; fp=len(pred)-tp
    fn_g=[i for i,g in enumerate(gt) if all(x[0]!=i for x in pairs)]
    fp_p=list(unmatched)
    for gi in fn_g:
        gt_t=gt[gi].get("t_on",gt[gi].get("t"))
        candidates=[pi for pi in fp_p if _get(pred[pi],"key")!=gt[gi]["key"] and abs(_get(pred[pi],"t")-gt_t)<=tol]
        if candidates:
            pi=min(candidates,key=lambda p:abs(_get(pred[p],"t")-gt_t)); fp_p.remove(pi); subs+=1
    precision=tp/(tp+fp) if tp+fp else (1.0 if not gt else 0.0)
    recall=tp/(tp+fn) if tp+fn else 1.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return MatchResult(tp,fp,fn,subs,pairs,precision,recall,f1)


def hold_errors(ground_truth, predictions, matches):
    gt=sorted([e for e in ground_truth if e.get("type","on")=="on"],key=lambda e:e.get("t_on",e.get("t")))
    pred_on=sorted([e for e in predictions if _get(e,"type")=="on"],key=lambda e:_get(e,"t"))
    errors=[]
    for gi,pi in matches.pairs:
        g=gt[gi]; p=pred_on[pi]
        if g.get("t_off") is None: continue
        ons=[e for e in predictions if _get(e,"type")=="on" and _get(e,"key")==_get(p,"key")]
        if not ons: continue
        on=min(ons,key=lambda e:abs(_get(e,"t")-g.get("t_on",g.get("t"))))
        offs=[e for e in predictions if _get(e,"type")=="off" and _get(e,"key")==_get(p,"key") and _get(e,"t")>=_get(on,"t")]
        if offs:
            gt_hold=g["t_off"]-g.get("t_on",g.get("t")); pr_hold=min(offs,key=lambda e:_get(e,"t"))
            errors.append((_get(pr_hold,"t")-_get(on,"t"))-gt_hold)
    return errors


def summarize(gt,pred,duration_s,tol=.150):
    m=match_events(gt,pred,tol)
    return {"tp":m.tp,"fp":m.fp,"fn":m.fn,"substitutions":m.substitutions,
        "precision":m.precision,"recall":m.recall,"f1":m.f1,
        "fp_per_min":m.fp/(duration_s/60) if duration_s>0 else 0.0}
