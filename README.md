# Paper Piano

Paper Piano detects dots drawn on a sheet of paper and turns them into a touch keyboard. The original OpenCV detector is retained as the V0 baseline. The extended engine adds canonical paper tracking, local appearance templates, fingertip and occlusion fusion, note release and velocity, optional audio/MIDI output, and tools for recording and labeling benchmark clips.

## Install and run

Use Python 3.10–3.12 and install the pinned computer-vision stack:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run the full live engine (MediaPipe hands enabled by the full config):

```bash
python paper-piano/main.py --config configs/full.json --camera 0 --audio samples
```

Press Space to freeze or unfreeze the detected row of dots; press `q` to quit. `--auto-freeze` freezes after the layout is stable. `--debug-overlay` shows the canonical view. Use `--variant V0` to run the original detector/occlusion baseline.

Add `--control-panel` to open the optional Tkinter sliders for paper detection, NCC thresholds, fingertip gate, sweep speed, and velocity range.

Audio and MIDI options:

```bash
python paper-piano/main.py --audio midi --midi-port "Your MIDI output" --scale major --root 60
python paper-piano/main.py --audio both --midi-port "Your MIDI output"
python paper-piano/main.py --list-midi-ports
```

MIDI scales are `major`, `chromatic`, and `original_wholetone`. The original sample set spans 28 whole-tone notes: file index `i` maps to MIDI note `31 + 2*i`. Windows users need a virtual MIDI driver such as loopMIDI to create a software MIDI port. Sample output works without MIDI configuration.

The Tkinter control panel from the original project remains available as a separate module; the live CLI provides keyboard controls and status overlays. The paper should be roughly upright, with one row of dark dots clearly inside its border.

## Configurations and engines

`configs/v0.json` through `configs/v4.json` are the ablation presets; `configs/full.json` is the full single-camera setup. Configuration files are plain JSON and can be loaded by the engine or CLI. The headless API is in `paper-piano/engine.py`:

```python
from config import Config
from engine import PaperPianoEngine

engine = PaperPianoEngine(Config())
result = engine.process([raw_bgr_frame], t=frame_index / fps)
if not engine.frozen:
    engine.freeze()
for event in result.events:
    ...  # note_on or note_off
```

Pass raw, unmirrored BGR frames. The engine applies mirroring according to config; it does not access a camera, GUI, or output device. Call `unfreeze()` to rescan the layout. Each event has a key index from left to right, timestamp, velocity, and debug data. `BaselineEngine` wraps the original `PianoDetection` and `TouchValidator` for V0.

## Collecting benchmark data

The tools build the evaluation pipeline; real results require human-recorded clips. Use at least two participants and keep tuning and test sessions separate. Record raw unflipped 640x480 video at 30 fps, leave auto-exposure enabled, and note camera and lighting details. Label key indices left-to-right based on the frozen layout.

```bash
python bench/record.py --protocol bench/protocols/S1.json --out data/S1/take01
python bench/label_tool.py data/S1/take01.mp4 --keys 8
```

The recorder writes `.mp4`, `.ts.json` frame timestamps, and `.cues.json` cue times. Cues help annotation but are not ground truth. The label tool supports `a`/`d` frame stepping, Space play/pause, number keys to select a dot, `o` onset, `r` release, `u` undo, and `s` save. Add clip metadata and split (`tuning` or `test`) to its `.gt.json` file. Recommended scenarios include scripted scales and chords, hand sweeps, palm contact without fingertip touch, lighting changes, paper/camera movement, soft-to-loud strikes, and held notes.

Run evaluation on labeled clips:

```bash
python bench/evaluate.py --data data/ --variants V0,V1,V2,V3 --split test --out results/
```

This writes per-clip metrics to `results/metrics.csv`, an aggregate `results/summary.md`, and an F1 plot. `bench/metrics.py` exposes event matching at the default 150 ms tolerance and reports key substitutions separately. Do not report synthetic scenes as benchmark results or tune thresholds on the test split.

## Design notes and limitations

The layout supports one roughly upright row of dots. Single-camera occlusion plus fingertip location cannot distinguish a hover and a touch when both lie on the same camera ray; fingertip fusion instead rejects many palm, sweep, and multi-key artifacts. V4 requires `--camera-b` and two cameras viewing the same paper; its calibrated evaluation requires synchronized recordings. Image-space strike speed is only a proxy for force. Report measured performance, tracker dropouts, participant count, and these limits before making robustness claims.

The sounds are sourced from the [University of Iowa Electronic Music Studios](http://theremin.music.uiowa.edu/) and modified for the project. The project is distributed under the MIT License; see [LICENSE](LICENSE).
