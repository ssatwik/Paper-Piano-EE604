"""Live camera UI for the headless Paper Piano engines."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
from config import Config, preset
from engine import PaperPianoEngine, BaselineEngine
from output import SampleSink, MidiSink, CompositeSink, NullSink
from control_panel import ControlPanel


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera",type=int,default=0); p.add_argument("--camera-b",type=int)
    p.add_argument("--config",default="configs/full.json"); p.add_argument("--variant",choices=["V0","V1","V2","V3","V4","FULL"])
    p.add_argument("--audio",choices=["samples","midi","both","none"],default="samples")
    p.add_argument("--midi-port"); p.add_argument("--list-midi-ports",action="store_true")
    p.add_argument("--scale",choices=["major","chromatic","original_wholetone"]); p.add_argument("--root",type=int)
    p.add_argument("--auto-freeze",action="store_true"); p.add_argument("--record")
    p.add_argument("--debug-overlay",action="store_true")
    p.add_argument("--control-panel",action="store_true",help="show the Tkinter live parameter panel")
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    if args.list_midi_ports:
        print("\n".join(MidiSink.list_ports())); return 0
    if args.variant: cfg=preset(args.variant)
    elif Path(args.config).exists(): cfg=Config.from_json(args.config)
    else: cfg=Config()
    if args.auto_freeze: cfg.auto_freeze=True
    if args.camera_b is not None: cfg.dual_camera=cfg.use_parallax=True
    if args.scale: cfg.scale=args.scale
    if args.root is not None: cfg.root=args.root
    engine=BaselineEngine(cfg) if args.variant=="V0" else PaperPianoEngine(cfg)
    sounds=Path(__file__).resolve().parent.parent/"sounds"
    sinks=[]
    try:
        if args.audio in ("samples","both"): sinks.append(SampleSink(sounds))
        if args.audio in ("midi","both"): sinks.append(MidiSink(args.midi_port,cfg.scale,cfg.root,cfg.midi_channel))
    except Exception:
        engine.close()
        for s in sinks: s.close()
        raise
    sink=CompositeSink(*sinks) if sinks else NullSink()
    cap=cv2.VideoCapture(args.camera); cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap_b=cv2.VideoCapture(args.camera_b) if args.camera_b is not None else None
    controls=ControlPanel() if args.control_panel else None
    writer=None
    record_timestamps=[]
    record_started=None
    if args.record: Path(args.record).parent.mkdir(parents=True,exist_ok=True)
    frozen=False
    try:
        while True:
            if controls is not None:
                controls.update()
                if controls.is_closed: break
                cfg.blur=controls.blur.get()
                cfg.canny1=controls.canny_ths1.get(); cfg.canny2=controls.canny_ths2.get()
                cfg.ncc_on=controls.ncc_on.get()/100; cfg.ncc_off=controls.ncc_off.get()/100
                cfg.gate_frac=controls.gate_frac.get()/100; cfg.v_sweep=controls.v_sweep.get()/10
                cfg.velocity_min=controls.velocity_min.get(); cfg.velocity_max=max(controls.velocity_max.get(),cfg.velocity_min+1)
            ok,frame=cap.read()
            if not ok: break
            timestamp=time.perf_counter()
            if args.record:
                record_started=record_started or timestamp
                record_timestamps.append(timestamp-record_started)
            frames=[frame]
            if cap_b:
                ok_b,frame_b=cap_b.read()
                if not ok_b: break
                if time.perf_counter()-timestamp > .080: continue
                frames.append(frame_b)
            result=engine.process(frames,timestamp)
            key_count = len(engine.scorer.dots) if hasattr(engine,"scorer") and engine.frozen else len(getattr(engine.det,"piano_points",[])) if hasattr(engine,"det") else 0
            for out_sink in sinks:
                if hasattr(out_sink,"set_key_count") and key_count: out_sink.set_key_count(key_count)
            for event in result.events:
                (sink.note_on(event.key,event.velocity) if event.type=="on" else sink.note_off(event.key))
            view=cv2.flip(frame,1) if cfg.mirror else frame.copy()
            if engine.last and engine.last[2].corners_cam is not None:
                cv2.polylines(view,[engine.last[2].corners_cam.astype("int32")],True,(0,255,0),2)
            if hasattr(engine,"det"):
                engine.det.draw_points(view)
            frozen=engine.frozen
            if controls is not None:
                if controls.freeze_points.get() and not engine.frozen: engine.freeze()
                elif not controls.freeze_points.get() and engine.frozen: sink.all_off(); engine.unfreeze()
            cv2.putText(view,f"{result.status} | {'FROZEN' if frozen else 'SPACE: freeze'} | q: quit",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,220,20),2)
            if result.per_key:
                for i,row in enumerate(result.per_key):
                    color=(0,0,255) if row["state"]=="HELD" else (255,100,0)
                    cv2.putText(view,f"{i}:{row['ncc']:.2f}",(10+i*65,50),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
            if args.debug_overlay and result.overlays.get("warped") is not None:
                cv2.imshow("Canonical",result.overlays["warped"])
            cv2.imshow("Paper Piano",view)
            if args.record:
                if writer is None: writer=cv2.VideoWriter(args.record,cv2.VideoWriter_fourcc(*"mp4v"),30,(frame.shape[1],frame.shape[0]))
                writer.write(frame)
            key=cv2.waitKey(1)&0xff
            if key==ord('q'): break
            if key==ord(' '):
                if engine.frozen:
                    sink.all_off(); engine.unfreeze()
                else: engine.freeze()
    finally:
        sink.all_off(); engine.close(); sink.close(); cap.release()
        if cap_b: cap_b.release()
        if writer: writer.release()
        if args.record:
            Path(args.record).with_suffix('.ts.json').write_text(json.dumps({"timestamps":record_timestamps},indent=2))
        if controls is not None: controls.destroy()
        cv2.destroyAllWindows()
    return 0

if __name__=="__main__": raise SystemExit(main())
