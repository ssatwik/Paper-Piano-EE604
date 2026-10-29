"""Headless deterministic engines for live use and offline evaluation."""
from dataclasses import dataclass, field
import time
import cv2
import numpy as np
import detection, validation
from tracking import PaperTracker, TrackResult, extract_dots, CANON_W
from occlusion import OcclusionScorer
from hands import HandTracker
from touch import TouchFusion, NoteEvent


@dataclass
class FrameResult:
    events: list[NoteEvent]
    status: str
    per_key: list[dict] = field(default_factory=list)
    overlays: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)


class PaperPianoEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracker = PaperTracker(cfg)
        self.tracker_b = PaperTracker(cfg) if cfg.dual_camera and cfg.use_parallax else None
        self.scorer = OcclusionScorer(cfg)
        self.legacy_scorer = validation.TouchValidator()
        self.hands = HandTracker(cfg.model_complexity, enabled=cfg.use_fingertip_gate)
        self.fusion = TouchFusion(cfg)
        self.frozen = False
        self.dot_cam = np.empty((0, 2), np.float32)
        self.canon_size = (CANON_W, 480)
        self.last = None
        self.stable = 0
        self.previous_dots = None
        self.hand_drop_since = None
        self.last_b = None
        self.b_rotation = 0
        self.H_fixed = None
        self.corners_fixed = None

    def _prepare(self, frame):
        work = cv2.flip(frame, 1) if self.cfg.mirror else frame.copy()
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        return work, gray

    def process(self, frames, t):
        started = time.perf_counter()
        work, gray = self._prepare(frames[0] if isinstance(frames, (list, tuple)) else frames)
        track = self.tracker.update(gray, t)
        if self.frozen and not self.cfg.use_tracking and self.H_fixed is not None:
            track = TrackResult(self.H_fixed.copy(), self.corners_fixed.copy(), 1.0, "tracking")
        self.last = (work, gray, track)
        track_b = None
        if self.tracker_b is not None:
            if not isinstance(frames, (list, tuple)) or len(frames) < 2:
                return FrameResult([], "searching", timings_ms={"track":track_ms,"total":(time.perf_counter()-started)*1000})
            work_b, gray_b = self._prepare(frames[1])
            track_b = self.tracker_b.update(gray_b, t)
            self.last_b = (work_b, gray_b, track_b)
        track_ms = (time.perf_counter()-started)*1000
        if not self.frozen:
            if self.cfg.auto_freeze and track.H_cam2canon is not None:
                warped = cv2.warpPerspective(work, track.H_cam2canon, self.canon_size)
                dots, _ = extract_dots(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
                if len(dots) >= 2 and self.previous_dots is not None and len(dots)==len(self.previous_dots) and np.mean(np.linalg.norm(dots-self.previous_dots,axis=1)) < 2:
                    self.stable += 1
                else: self.stable = 0
                self.previous_dots = dots
                if self.stable >= 30: self.freeze()
            return FrameResult([], track.status, timings_ms={"track":track_ms,"total":(time.perf_counter()-started)*1000})
        if track.status == "lost" or track.H_cam2canon is None or (track_b is not None and (track_b.status == "lost" or track_b.corners_cam is None)):
            events = self.fusion.all_off(t)
            return FrameResult(events, "lost", timings_ms={"track":track_ms,"total":(time.perf_counter()-started)*1000})
        warp_started=time.perf_counter()
        warped = cv2.warpPerspective(work, track.H_cam2canon, self.canon_size)
        warp_ms=(time.perf_counter()-warp_started)*1000
        hands_started = time.perf_counter()
        tips, landmarks = self.hands.process(work, t)
        hands_ms = (time.perf_counter()-hands_started)*1000
        inv = np.linalg.inv(track.H_cam2canon)
        dot_cam = cv2.perspectiveTransform(self.scorer.dots.reshape(-1,1,2),inv).reshape(-1,2)
        update_mask=[]
        for i, point in enumerate(dot_cam):
            idle = self.fusion.states[i]["state"] == "IDLE" and t-self.fusion.states[i]["idle_since"] >= .5
            radius = 2.5*self.cfg.gate_frac*max([f.scale for f in tips], default=1.0)
            clear = not landmarks or min(np.linalg.norm(np.asarray(landmarks)-point,axis=1)) > radius
            update_mask.append(idle and clear and track.confidence == 1)
        if self.cfg.use_canonical_ncc:
            ncc, offsets = self.scorer.score(warped, allow_update=track.confidence == 1, update_mask=update_mask)
        else:
            ncc=np.asarray(self.legacy_scorer.scores(work),np.float32); offsets=[]
        if track_b is not None and self.b_rotation is not None:
            corners_b = np.roll(track_b.corners_cam, self.b_rotation, axis=0)
            h_b = PaperTracker.transform(corners_b, *self.canon_size)
            warped_b = cv2.warpPerspective(self.last_b[0], h_b, self.canon_size)
            ncc_b, _ = self.scorer.score(warped_b)
            # Joint appearance must be occluded in both views; either clear view releases.
            ncc = np.maximum(ncc, ncc_b)
        ncc_ms=(time.perf_counter()-hands_started-hands_ms/1000)*1000
        fusion_started=time.perf_counter()
        events, info = self.fusion.update(t,ncc,dot_cam,tips,track.status, fallback=not bool(tips))
        if not tips:
            if self.hand_drop_since is None: self.hand_drop_since=t
            if t-self.hand_drop_since > self.cfg.grace_hand_lost and np.any(ncc >= self.cfg.ncc_on):
                events.extend(self.fusion.all_off(t))
        else: self.hand_drop_since=None
        fusion_ms=(time.perf_counter()-fusion_started)*1000
        return FrameResult(events,track.status,info,{"warped":warped},
            {"track":track_ms,"warp":warp_ms,"ncc":ncc_ms,"hands":hands_ms,"fusion":fusion_ms,
             "total":(time.perf_counter()-started)*1000})

    def freeze(self):
        if self.last is None: return False
        frame, gray, track = self.last
        if track.H_cam2canon is None or track.corners_cam is None: return False
        aspect = (np.linalg.norm(track.corners_cam[0]-track.corners_cam[3])+np.linalg.norm(track.corners_cam[1]-track.corners_cam[2]))/max(
            np.linalg.norm(track.corners_cam[0]-track.corners_cam[1])+np.linalg.norm(track.corners_cam[3]-track.corners_cam[2]),1)
        self.canon_size=(CANON_W,max(80,int(round(CANON_W*aspect))))
        warped=cv2.warpPerspective(frame,track.H_cam2canon,self.canon_size)
        dots,boxes=extract_dots(cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY))
        if len(dots)<2: return False
        self.scorer.initialize(warped,dots,boxes)
        if self.tracker_b is not None:
            if self.last_b is None or self.last_b[2].corners_cam is None:
                return False
            work_b, _, track_b = self.last_b
            best = (-float("inf"), None)
            for rotation in range(4):
                corners_b = np.roll(track_b.corners_cam, rotation, axis=0)
                h_b = PaperTracker.transform(corners_b, *self.canon_size)
                candidate = cv2.warpPerspective(work_b, h_b, self.canon_size)
                scores, _ = self.scorer.score(candidate)
                quality = float(np.mean(scores)) if len(scores) else -1.0
                if quality > best[0]: best = (quality, rotation)
            if best[0] < .5:
                self.scorer.ready = False
                raise RuntimeError("Camera B does not show the same paper clearly enough (mean NCC < 0.5)")
            self.b_rotation = best[1]
        self.dot_cam=cv2.perspectiveTransform(dots.reshape(-1,1,2),np.linalg.inv(track.H_cam2canon)).reshape(-1,2)
        if not self.cfg.use_canonical_ncc:
            self.legacy_scorer.initialize_points_images(frame,self.dot_cam.tolist())
        self.H_fixed=track.H_cam2canon.copy(); self.corners_fixed=track.corners_cam.copy()
        self.fusion.reset(len(dots)); self.frozen=True
        return True

    def unfreeze(self):
        self.fusion.all_off(0.0)
        self.frozen=False; self.scorer.ready=False
        self.H_fixed=None; self.corners_fixed=None

    def close(self):
        self.fusion.all_off(0); self.hands.close()


class BaselineEngine:
    """Wrapper that preserves the original frame-level detection and validator."""
    def __init__(self,cfg=None):
        self.cfg=cfg
        self.det=detection.PianoDetection(); self.validator=validation.TouchValidator()
        self.frozen=False; self.warmup=0; self.last_frame=None

    def process(self,frames,t):
        started=time.perf_counter()
        frame=frames[0] if isinstance(frames,(list,tuple)) else frames
        frame=cv2.flip(frame,1) if getattr(self.cfg,"mirror",True) else frame.copy()
        detect_started=time.perf_counter()
        if not self.frozen:
            self.det.process(frame, getattr(self.cfg,"blur",10)*2+1,getattr(self.cfg,"canny1",40),getattr(self.cfg,"canny2",40))
            self.warmup+=1
            if getattr(self.cfg,"auto_freeze",False) and self.warmup>=30 and len(self.det.piano_points)>=2: self.freeze()
        self.last_frame=frame
        detect_ms=(time.perf_counter()-detect_started)*1000
        events=[]
        if self.frozen:
            for key in self.validator.process(frame,getattr(self.cfg,"ncc_on",.6)):
                events.append(NoteEvent("on",key,t,100))
        return FrameResult(events,"tracking" if self.det.piano_points else "searching",
                           timings_ms={"track":detect_ms,"total":(time.perf_counter()-started)*1000})

    def freeze(self):
        if self.last_frame is None or len(self.det.piano_points)<2: return False
        self.validator.initialize_points_images(self.last_frame,self.det.piano_points); self.frozen=True; return True
    def unfreeze(self): self.frozen=False
    def close(self): pass
