"""Serializable settings and ablation presets for Paper Piano."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Config:
    mirror: bool = True
    auto_freeze: bool = False
    dual_camera: bool = False
    use_tracking: bool = True
    use_canonical_ncc: bool = True
    use_search_window: bool = True
    use_adaptive_template: bool = True
    use_fingertip_gate: bool = True
    use_wta: bool = True
    use_kinematics: bool = True
    occlusion_only_fallback: bool = False
    use_parallax: bool = False
    ncc_on: float = 0.55
    ncc_off: float = 0.70
    template_margin: int = 4
    search_margin: int = 4
    alpha: float = 0.03
    n_on: int = 2
    n_off: int = 2
    gate_frac: float = 0.25
    v_sweep: float = 2.0
    grace_hand_lost: float = 0.30
    refractory: float = 0.08
    velocity_min: int = 30
    velocity_max: int = 127
    velocity_s_lo: float = 0.3
    velocity_s_hi: float = 4.0
    velocity_gamma: float = 0.7
    adaptive_velocity: bool = False
    model_complexity: int = 0
    blur: int = 10
    canny1: int = 40
    canny2: int = 40
    hold_frames: int = 10
    pedal_key: int | None = None
    scale: str = "major"
    root: int = 60
    midi_channel: int = 0

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def preset(name: str) -> Config:
    name = name.upper()
    cfg = Config()
    if name == "V0":
        cfg.use_tracking = cfg.use_fingertip_gate = cfg.use_adaptive_template = False
        cfg.ncc_on = cfg.ncc_off = 0.60
    elif name == "V1":
        cfg.use_fingertip_gate = cfg.use_wta = cfg.use_kinematics = False
        cfg.alpha = 0.01
    elif name == "V2":
        cfg.use_kinematics = False
    elif name == "V3" or name == "FULL":
        pass
    elif name == "V4":
        cfg.dual_camera = cfg.use_parallax = True
    else:
        raise ValueError(f"Unknown config preset: {name}")
    return cfg
