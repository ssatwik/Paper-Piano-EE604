# Paper Piano extension report draft

## Contribution

This implementation preserves the original fixed-camera dot-occlusion detector as V0 and adds a headless engine with a paper-plane coordinate frame, searchable local appearance templates, optional fingertip/occlusion fusion, debounced note-on and note-off events, strike-speed velocity, sample and MIDI sinks, and clip recording/labeling/evaluation utilities. V4 includes a second-view consensus path that requires both views to report a dot occluded.

## Related work

Paper Piano and related projects already demonstrate camera-based paper keyboard interaction. The original repository detects the paper and treats a hidden dot as a key press ([BTifmmp/paper-piano](https://github.com/BTifmmp/paper-piano)); an Android virtual-piano project combines paper-key localization with fingertip detection and movement-based presses ([alatanlc/virtual-piano](https://github.com/alantanlc/virtual-piano)). Other systems use printed fiducials, homographies, and MediaPipe hand tracking to play a paper piano ([Paper Piano case study](https://www.larchi.dev/projects/paperpiano)). Multiple oblique cameras have also been used for touch interaction on planar surfaces ([Holz, *3D From 2D Touch*](https://www.christianholz.net/2013-thesis-ChristianHolz-3D_from_2D_Touch-PhD_Dissertation.pdf)). Thus fingertip tracking, paper pianos, and multi-view surface-touch sensing are prior art. This project’s intended engineering contribution is the ablation of paper tracking, appearance matching, fingertip claims, lifecycle logic, and two-view occlusion consensus against the preserved baseline. No novelty or performance claim is made until evaluation on human-recorded test clips.

## Evaluation status

The evaluation scripts and synthetic unit tests are included, but no camera clips or ground-truth annotations were present in the workspace. Dataset collection, tuning/test separation, measured precision/recall/F1, false triggers per minute, latency, velocity correlation, and confidence intervals remain pending. Synthetic scenes are only for implementation checks and must not be used as headline results.

## Limitations

The layout assumes one roughly upright row of dots. A single RGB view cannot geometrically distinguish a hover from a touch when the fingertip lies on the same camera ray as the dot. The second-view path requires both cameras to see the same paper and still needs S9 measurements to quantify useful hover-height rejection. Image-plane fingertip speed is a pose-dependent proxy for force. Results must report MediaPipe dropouts, camera arrangement, and participant/session counts.
