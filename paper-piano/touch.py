"""Fingertip and occlusion fusion with debounced note lifecycle."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class NoteEvent:
    type: str
    key: int
    t: float
    velocity: int = 0
    debug: dict = field(default_factory=dict)


def point_segment_dist(points, a, b):
    ab = b-a
    u = np.clip(((points-a) @ ab) / (ab @ ab + 1e-9), 0, 1)
    return np.linalg.norm(points-(a+u[:, None]*ab), axis=1)


class TouchFusion:
    def __init__(self, cfg):
        self.cfg = cfg
        self.states = []
        self.velocity_samples = []

    def reset(self, n):
        self.states = [dict(state="IDLE", on_count=0, cond_t=None, idle_since=0.0, off_count=0, last_near=-1e9,
                            refractory=0.0, t_on=0.0, claimer=None, fallback=0) for _ in range(n)]

    def _velocity(self, finger, t):
        speeds = [s for ts, s in getattr(finger, "hist", []) if t-.15 <= ts <= t]
        s = max(speeds, default=getattr(finger, "speed_norm", 0.0))
        lo, hi = self.cfg.velocity_s_lo, self.cfg.velocity_s_hi
        if self.cfg.adaptive_velocity and len(self.velocity_samples) >= 10:
            lo, hi = np.percentile(self.velocity_samples[-30:], [10, 90])
        u = np.clip((s-lo)/max(hi-lo, 1e-6), 0, 1) ** self.cfg.velocity_gamma
        vel = int(round(self.cfg.velocity_min+(self.cfg.velocity_max-self.cfg.velocity_min)*u))
        self.velocity_samples.append(float(s))
        return vel

    def update(self, t, ncc, dot_cam, fingertips, track_status="tracking", fallback=False):
        n = len(dot_cam)
        if len(self.states) != n: self.reset(n)
        claims = {}
        if self.cfg.use_wta:
            for f in fingertips:
                d = point_segment_dist(np.asarray(dot_cam, np.float32), np.asarray(f.dip), np.asarray(f.p))
                key = int(np.argmin(d)) if len(d) else -1
                if key >= 0 and d[key] < self.cfg.gate_frac*max(f.scale, 1):
                    if key not in claims or d[key] < claims[key][0]: claims[key] = (float(d[key]), f)
        else:
            for key, point in enumerate(np.asarray(dot_cam, np.float32)):
                candidates = [(float(point_segment_dist(np.asarray([point]), np.asarray(f.dip), np.asarray(f.p))[0]), f) for f in fingertips]
                if candidates:
                    distance, finger = min(candidates, key=lambda item: item[0])
                    if distance < self.cfg.gate_frac*max(finger.scale, 1): claims[key] = (distance, finger)
        events, info = [], []
        can_press = track_status == "tracking"
        for i, state in enumerate(self.states):
            claim = claims.get(i, (None, None))[1]
            if not self.cfg.use_fingertip_gate and ncc[i] < self.cfg.ncc_on:
                claim = object()
            if claim is None and fallback and self.cfg.occlusion_only_fallback and ncc[i] < self.cfg.ncc_on:
                state["fallback"] += 1
                if state["fallback"] >= 3: claim = object()
            else: state["fallback"] = 0
            near = claim is not None
            cond = ncc[i] < self.cfg.ncc_on and near and (not self.cfg.use_kinematics or getattr(claim, "speed_norm", 0) <= self.cfg.v_sweep)
            if state["state"] == "IDLE":
                if cond and state["on_count"] == 0:
                    state["cond_t"], state["cond_claimer"] = t, claim
                state["on_count"] = state["on_count"]+1 if cond and can_press and t >= state["refractory"] else 0
                if state["on_count"] >= self.cfg.n_on:
                    first_t = state["cond_t"] if state["cond_t"] is not None else t
                    first_claim = state.get("cond_claimer", claim)
                    state["state"], state["t_on"], state["claimer"] = "HELD", first_t, first_claim
                    events.append(NoteEvent("on", i, first_t, self._velocity(first_claim, first_t), {"ncc": float(ncc[i])}))
                    state["on_count"], state["cond_t"], state["cond_claimer"] = 0, None, None
            else:
                if near: state["last_near"] = t
                release = (ncc[i] > self.cfg.ncc_off) or (not near and t-state["last_near"] > self.cfg.grace_hand_lost)
                state["off_count"] = state["off_count"]+1 if release else 0
                if not can_press: state["off_count"] = self.cfg.n_off
                if state["off_count"] >= self.cfg.n_off:
                    events.append(NoteEvent("off", i, t, 0, {"held_duration": t-state["t_on"]}))
                    state.update(state="IDLE", off_count=0, on_count=0, refractory=t+self.cfg.refractory,
                                 idle_since=t, claimer=None)
            info.append({"ncc": float(ncc[i]), "claimed_by": getattr(claim, "finger", None), "state": state["state"]})
        return events, info

    def all_off(self, t):
        events = []
        for i, state in enumerate(self.states):
            if state["state"] == "HELD":
                events.append(NoteEvent("off", i, t, debug={"held_duration": t-state["t_on"], "forced": True}))
            state.update(state="IDLE", on_count=0, off_count=0, claimer=None)
        return events
