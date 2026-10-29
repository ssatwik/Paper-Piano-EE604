import cv2
import numpy as np
from config import Config
from tracking import PaperTracker
from synth import scene

def test_tracker_canonical_coordinates_survive_translation():
    tracker=PaperTracker(Config())
    canonical=[]
    for i in range(8):
        frame,_=scene(shift=(i*2,0))
        result=tracker.update(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),i/30)
        if result.H_cam2canon is not None:
            point=cv2.perspectiveTransform(np.array([[[160+i*2,230]]],np.float32),result.H_cam2canon)[0,0]
            canonical.append(point)
    assert max(np.linalg.norm(p-canonical[0]) for p in canonical)<1.5
