"""Record a camera clip and timestamp each frame; cue protocols are JSON."""
import argparse, json, time
from pathlib import Path
import cv2

def main():
    p=argparse.ArgumentParser(); p.add_argument('--protocol'); p.add_argument('--out',required=True); p.add_argument('--camera',type=int,default=0)
    a=p.parse_args(); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path(a.protocol).read_text()) if a.protocol else {"cues":[]}
    cap=cv2.VideoCapture(a.camera); writer=None; stamps=[]; cues=[]; cue_i=0; start=None
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if writer is None: writer=cv2.VideoWriter(str(out.with_suffix('.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),30,(frame.shape[1],frame.shape[0]))
            now=time.perf_counter(); start=start or now; stamps.append(now-start); writer.write(frame)
            text='q: stop'
            if cue_i<len(protocol.get('cues',[])):
                cue=protocol['cues'][cue_i]; due=float(cue.get('at',cue_i*2))
                text=f"{cue.get('label','CUE')} in {max(0,due-(now-start)):.1f}s"
                if now-start>=due: cues.append({**cue,"t":now-start}); cue_i+=1
            cv2.putText(frame,text,(20,40),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,255),2); cv2.imshow('Recorder',frame)
            if cv2.waitKey(1)&0xff==ord('q'): break
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
    out.with_suffix('.ts.json').write_text(json.dumps({"timestamps":stamps},indent=2))
    out.with_suffix('.cues.json').write_text(json.dumps({"cues":cues},indent=2))
if __name__=='__main__': main()
