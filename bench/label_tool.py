"""Frame-stepped video labeling tool for onset and release annotations."""
import argparse,json
from pathlib import Path
import cv2

def main():
    p=argparse.ArgumentParser(); p.add_argument('video'); p.add_argument('--out'); p.add_argument('--keys',type=int,default=8); a=p.parse_args()
    cap=cv2.VideoCapture(a.video); fps=cap.get(cv2.CAP_PROP_FPS) or 30; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frame_i=0; playing=False; labels=[]; key=0
    cue_path=Path(a.video).with_suffix('.cues.json')
    cues=json.loads(cue_path.read_text()).get('cues',[]) if cue_path.exists() else []
    def get(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,min(n-1,i))); ok,f=cap.read(); return f if ok else None
    frame=get(frame_i)
    while frame is not None:
        view=frame.copy()
        for j in range(a.keys): cv2.putText(view,str(j),(20+j*45,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        cv2.putText(view,f'frame {frame_i}/{n} t={frame_i/fps:.3f}s key={key} | a/d step, space play, o onset, r release, u undo, s save, q quit',(10,60),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
        cv2.imshow('Label',view); k=cv2.waitKey(30 if playing else 0)&0xff
        if k==ord('q'): break
        elif k==ord('a'): frame_i-=1
        elif k==ord('d'): frame_i+=1
        elif k==ord(' '): playing=not playing
        elif k==ord('['):
            prev=[int(c['t']*fps) for c in cues if c.get('t',0)*fps<frame_i]
            if prev: frame_i=max(prev)
        elif k==ord(']'):
            nxt=[int(c['t']*fps) for c in cues if c.get('t',0)*fps>frame_i]
            if nxt: frame_i=min(nxt)
        elif k in [ord(str(i)) for i in range(10)]: key=k-ord('0')
        elif k==ord('o'): labels.append({"key":key,"t_on":frame_i/fps,"frame_on":frame_i})
        elif k==ord('r'):
            found=next((e for e in reversed(labels) if e['key']==key and 't_off' not in e),None)
            if found: found.update(t_off=frame_i/fps,frame_off=frame_i)
        elif k==ord('u') and labels: labels.pop()
        elif k==ord('s'):
            path=Path(a.out or str(Path(a.video).with_suffix('.gt.json')))
            path.write_text(json.dumps({"fps":fps,"n_keys":a.keys,"events":labels},indent=2))
        if playing: frame_i+=1
        frame=get(frame_i)
    cap.release(); cv2.destroyAllWindows()
if __name__=='__main__': main()
