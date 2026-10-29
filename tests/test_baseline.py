from config import Config
from engine import BaselineEngine
from synth import scene

def test_baseline_engine_keeps_original_occlusion_press_edge():
    engine=BaselineEngine(Config(mirror=False))
    frame,_=scene()
    engine.process([frame],0.0)
    assert engine.freeze()
    covered,_=scene(cover=0)
    result=engine.process([covered],1/30)
    assert [(event.type,event.key) for event in result.events]==[("on",0)]
