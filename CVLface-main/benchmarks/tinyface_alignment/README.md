# TinyFace alignment benchmark on Modal

This is a deterministic smoke benchmark for alignment choices feeding the
CVLFace ViT-KPRPE recognizer. TinyFace has no landmark ground truth, so the
reported metrics are downstream Rank-1/mAP, operational coverage by original
face-crop size, and end-to-end latency. They are not landmark NME/PCK/AUC.

## Upload TinyFace once

Run from the workspace root (`VSF`):

```bash
.venv-modal/bin/modal volume create cvlface-tinyface-data
.venv-modal/bin/modal volume put cvlface-tinyface-data \
  CVLface-main/tinyface /tinyface
```

Confirm the expected tree exists:

```bash
.venv-modal/bin/modal volume ls cvlface-tinyface-data /tinyface/Testing_Set
```

## Run

Run every adapter on an L4 GPU:

```bash
.venv-modal/bin/modal run \
  CVLface-main/benchmarks/tinyface_alignment/modal_app.py
```

Run a cheap first pass before all models:

```bash
.venv-modal/bin/modal run \
  CVLface-main/benchmarks/tinyface_alignment/modal_app.py \
  --probes-per-bin 1 --distractors 10 \
  --pipelines square,dfa_mobilenet,scrfd10g
```

The result is returned to `last_modal_result.json` locally and persisted as
`tinyface_alignment_smoke.json` in the `cvlface-tinyface-results` Volume.

Download the persisted result with:

```bash
.venv-modal/bin/modal volume get cvlface-tinyface-results \
  tinyface_alignment_smoke.json \
  CVLface-main/benchmarks/tinyface_alignment/tinyface_alignment_smoke.json
```

## Compared pipelines

- `square`: no-landmark control.
- `dfa_mobilenet`: CVLFace DFA MobileNet.
- `dfa_resnet50`: CVLFace DFA ResNet-50.
- `scrfd10g`: InsightFace SCRFD-10GF-KPS from `buffalo_l`.
- `mediapipe`: MediaPipe FaceMesh converted to ArcFace-style five points.

The first cloud run builds the image and downloads public model weights. Later
runs reuse Modal image caching and the `cvlface-model-cache` Volume.
