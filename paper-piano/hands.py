"""Optional MediaPipe Hands adapter (dependency loaded only when enabled)."""
from collections import deque
from dataclasses import dataclass, field
import cv2
import numpy as np
from tracking import OneEuroFilter


@dataclass
class Fingertip:
    hand: str
    finger: int
    p: np.ndarray
    dip: np.ndarray
    scale: float
    v: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    speed_norm: float = 0.0
    hist: deque = field(default_factory=lambda: deque(maxlen=32))


class HandTracker:
    TIP = (4, 8, 12, 16, 20)
    DIP = (3, 7, 11, 15, 19)

    def __init__(self, model_complexity=0, enabled=True):
        self.enabled = enabled
        self.hands = None
        self.filters = {}
        self.previous = {}
        self.velocity = {}
        if enabled:
            try:
                import mediapipe as mp
                self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                    model_complexity=model_complexity, min_detection_confidence=.5, min_tracking_confidence=.5)
            except (ImportError, AttributeError) as exc:
                raise RuntimeError("Hand tracking requires mediapipe==0.10.14") from exc

    def process(self, frame, t):
        if self.hands is None:
            return [], []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.hands.process(rgb)
        h, w = frame.shape[:2]
        tips, all_landmarks = [], []
        for hi, hand_lm in enumerate(res.multi_hand_landmarks or []):
            label = "unknown"
            if res.multi_handedness and hi < len(res.multi_handedness):
                label = res.multi_handedness[hi].classification[0].label
            pts = np.asarray([(p.x*w, p.y*h) for p in hand_lm.landmark], np.float32)
            all_landmarks.extend(pts)
            scale = max(float(np.linalg.norm(pts[0]-pts[9])), 1.0)
            for finger, (tip_id, dip_id) in enumerate(zip(self.TIP, self.DIP)):
                key = (label, finger)
                fp = self.filters.setdefault(key, (OneEuroFilter(1.5,.02), OneEuroFilter(1.5,.02)))
                p = fp[0](pts[tip_id], t); dip = fp[1](pts[dip_id], t)
                prev, pv = self.previous.get(key), self.velocity.get(key, [])
                vel = (p-prev[0])/max(t-prev[1], 1e-6) if prev else np.zeros(2, np.float32)
                pv = (pv + [vel])[-3:]
                v = np.mean(pv, axis=0)
                speed = float(np.linalg.norm(v)/scale)
                hist = self.velocity.setdefault((key, "hist"), deque(maxlen=32))
                hist.append((t, speed))
                self.previous[key] = (p.copy(), t)
                self.velocity[key] = pv
                tips.append(Fingertip(label, finger, p, dip, scale, v, speed, hist))
        return tips, all_landmarks

    def close(self):
        if self.hands is not None:
            self.hands.close()
