"""Modal smoke benchmark for TinyFace alignment pipelines used by CVLFace KPRPE.

TinyFace has no landmark ground truth. This benchmark therefore measures
downstream identification utility, coverage, and latency; it does not report
landmark NME/PCK/AUC.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "cvlface-tinyface-alignment"
DATA_VOLUME_NAME = "cvlface-tinyface-data"
RESULT_VOLUME_NAME = "cvlface-tinyface-results"
MODEL_VOLUME_NAME = "cvlface-model-cache"

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
result_volume = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=True)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.1.2",
        "torchvision==0.16.2",
        "transformers==4.34.1",
        "huggingface-hub==0.17.3",
        "omegaconf==2.3.0",
        "scikit-image==0.22.0",
        "opencv-python-headless==4.9.0.80",
        "onnxruntime-gpu==1.17.1",
        "insightface==0.7.3",
        "mediapipe==0.10.14",
        "pandas==2.2.2",
        "pyrootutils==1.0.4",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/mk-minchul/CVLface.git /opt/CVLface",
    )
    .env(
        {
            "PYTHONPATH": "/opt/CVLface/cvlface",
            "HF_HOME": "/models/huggingface",
            "CVLFACE_CACHE": "/models/cvlface",
        }
    )
)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={
        "/data": data_volume,
        "/results": result_volume,
        "/models": model_volume,
    },
)
def benchmark(
    probes_per_bin: int = 5,
    distractors: int = 100,
    pipelines: str = "square,dfa_mobilenet,dfa_resnet50,scrfd10g,mediapipe",
) -> dict:
    import os
    import platform
    import time
    from collections import defaultdict

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    from general_utils.huggingface_model_utils import load_model_from_local_path

    size_bins = (
        ("S1 (<=16 px)", 0, 16),
        ("S2 (17-20 px)", 17, 20),
        ("S3 (21-24 px)", 21, 24),
        ("S4 (25-28 px)", 25, 28),
        ("S5 (29-32 px)", 29, 32),
    )
    pipeline_names = [name.strip() for name in pipelines.split(",") if name.strip()]
    allowed = {"square", "dfa_mobilenet", "dfa_resnet50", "scrfd10g", "mediapipe"}
    unknown = set(pipeline_names) - allowed
    if unknown:
        raise ValueError(f"Unknown pipelines: {sorted(unknown)}")

    tiny_root = Path("/data/tinyface")
    if not (tiny_root / "Testing_Set/Probe").is_dir():
        raise FileNotFoundError(
            "TinyFace is missing at /data/tinyface. Run the upload command in README.md."
        )

    def label(path: Path) -> int:
        return int(path.stem.split("_")[0])

    def path_key(path: Path):
        bits = path.stem.split("_", 1)
        return int(bits[0]), int(bits[1]) if len(bits) > 1 and bits[1].isdigit() else -1

    def hw(path: Path):
        with Image.open(path) as im:
            return im.height, im.width

    def bin_name(side: int) -> str:
        for name, lo, hi in size_bins:
            if lo <= side <= hi:
                return name
        return "S6 (>32 px)"

    selected = {name: [] for name, _, _ in size_bins}
    used_ids = set()
    candidates = sorted((tiny_root / "Testing_Set/Probe").glob("*.jpg"), key=path_key)
    for path in candidates:
        group = bin_name(min(hw(path)))
        if group not in selected or len(selected[group]) >= probes_per_bin or label(path) in used_ids:
            continue
        selected[group].append(path)
        used_ids.add(label(path))
        if all(len(rows) == probes_per_bin for rows in selected.values()):
            break
    probes = [p for name, _, _ in size_bins for p in selected[name]]
    ids = {label(p) for p in probes}
    matches = [
        p
        for p in sorted((tiny_root / "Testing_Set/Gallery_Match").glob("*.jpg"), key=path_key)
        if label(p) in ids
    ]
    distractor_paths = sorted((tiny_root / "Testing_Set/Gallery_Distractor").glob("*.jpg"))[
        :distractors
    ]
    items = (
        [(p, "probe") for p in probes]
        + [(p, "gallery_match") for p in matches]
        + [(p, "distractor") for p in distractor_paths]
    )
    if not probes or not matches:
        raise RuntimeError("Deterministic TinyFace sample could not be created")

    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    def download_model(repo_id: str) -> Path:
        target = Path("/models/cvlface") / repo_id.replace("/", "__")
        if not (target / "pretrained_model/model.pt").exists():
            snapshot_download(repo_id=repo_id, local_dir=target, local_dir_use_symlinks=False)
            model_volume.commit()
        return target

    recognizer_id = "minchul/cvlface_adaface_vit_base_kprpe_webface4m"
    recognizer = load_model_from_local_path(str(download_model(recognizer_id))).to(device).eval()
    dfa_models = {}
    for name, repo_id in {
        "dfa_mobilenet": "minchul/cvlface_DFA_mobilenet",
        "dfa_resnet50": "minchul/cvlface_DFA_resnet50",
    }.items():
        if name in pipeline_names:
            dfa_models[name] = load_model_from_local_path(str(download_model(repo_id))).to(device).eval()

    scrfd = None
    if "scrfd10g" in pipeline_names:
        from insightface.app import FaceAnalysis

        scrfd = FaceAnalysis(
            name="buffalo_l",
            root="/models/insightface",
            allowed_modules=["detection"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        scrfd.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))
        model_volume.commit()

    face_mesh = None
    if "mediapipe" in pipeline_names:
        import mediapipe as mp

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.3,
        )

    arc_template = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32,
    )

    def square_rgb(path: Path):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        side = max(h, w)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        top, left = (side - h) // 2, (side - w) // 2
        canvas[top:top + h, left:left + w] = rgb
        return canvas

    def tensor_from_rgb(rgb: np.ndarray):
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(127.5).sub(1.0)
        return x.unsqueeze(0).to(device)

    def warp_from_five(rgb: np.ndarray, kps: np.ndarray):
        transform, _ = cv2.estimateAffinePartial2D(
            kps.astype(np.float32), arc_template, method=cv2.LMEDS
        )
        if transform is None:
            raise RuntimeError("similarity transform failed")
        aligned = cv2.warpAffine(rgb, transform, (112, 112), borderValue=0)
        homogeneous = np.concatenate([kps, np.ones((5, 1), np.float32)], axis=1)
        aligned_kps = homogeneous @ transform.T
        return tensor_from_rgb(aligned), torch.from_numpy(aligned_kps / 112.0).float()[None].to(device)

    def infer(path: Path, name: str):
        rgb = square_rgb(path)
        confidence = None
        if name == "square":
            aligned = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
            # A neutral canonical template is only a control; it is not a landmark prediction.
            kps = torch.from_numpy(arc_template / 112.0).float()[None].to(device)
            return tensor_from_rgb(aligned), kps, 1.0
        if name in dfa_models:
            outputs = dfa_models[name](tensor_from_rgb(rgb))
            confidence = float(outputs[3].item())
            return outputs[0], outputs[2], confidence
        if name == "scrfd10g":
            faces = scrfd.get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not faces:
                raise RuntimeError("no face")
            face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
            return (*warp_from_five(rgb, np.asarray(face.kps, np.float32)), float(face.det_score))
        if name == "mediapipe":
            enlarged = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_CUBIC)
            result = face_mesh.process(enlarged)
            if not result.multi_face_landmarks:
                raise RuntimeError("no face mesh")
            points = result.multi_face_landmarks[0].landmark
            def mean(indices):
                return np.mean([[points[i].x * rgb.shape[1], points[i].y * rgb.shape[0]] for i in indices], axis=0)
            five = np.asarray(
                [mean([33, 133]), mean([362, 263]), mean([1]), mean([61]), mean([291])],
                dtype=np.float32,
            )
            return (*warp_from_five(rgb, five), 1.0)
        raise AssertionError(name)

    def embedding(aligned, kps):
        value = recognizer(aligned, kps)
        value = F.normalize(value.float(), dim=1)
        return value[0].detach().cpu().numpy()

    def summarize(records):
        count = len(records)
        successes = [r for r in records if r["success"]]
        latencies = [r["latency_ms"] for r in records]
        return {
            "count": count,
            "coverage": len(successes) / count if count else None,
            "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        }

    def retrieval(records):
        probe_rows = [r for r in records if r["split"] == "probe"]
        gallery = [r for r in records if r["split"] != "probe"]
        rank1, aps = [], []
        for probe in probe_rows:
            valid_gallery = [g for g in gallery if g.get("embedding") is not None]
            if probe.get("embedding") is None or not valid_gallery:
                rank1.append(0.0); aps.append(0.0); continue
            scores = np.asarray([float(g["embedding"] @ probe["embedding"]) for g in valid_gallery])
            order = np.argsort(-scores, kind="stable")
            relevant = np.asarray([g["label"] == probe["label"] for g in valid_gallery])[order]
            rank1.append(float(relevant[0]))
            positions = np.flatnonzero(relevant) + 1
            aps.append(float(np.mean(np.arange(1, len(positions) + 1) / positions)) if len(positions) else 0.0)
        return {"rank1": float(np.mean(rank1)), "mAP": float(np.mean(aps)), "probe_count": len(probe_rows)}

    output = {
        "protocol": {
            "official_tinyface": False,
            "landmark_ground_truth": False,
            "recognizer": recognizer_id,
            "probes_per_bin": probes_per_bin,
            "distractors": distractors,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "sample": {"probes": len(probes), "matches": len(matches), "distractors": len(distractor_paths)},
        "pipelines": {},
    }

    with torch.inference_mode():
        for name in pipeline_names:
            print(f"Benchmarking {name} on {len(items)} images", flush=True)
            records = []
            for path, split in items:
                start = time.perf_counter()
                row = {
                    "name": path.name,
                    "label": label(path) if split != "distractor" else -100,
                    "split": split,
                    "size_bin": bin_name(min(hw(path))),
                    "success": False,
                    "error": "",
                    "embedding": None,
                }
                try:
                    aligned, keypoints, confidence = infer(path, name)
                    row["confidence"] = confidence
                    row["embedding"] = embedding(aligned, keypoints)
                    row["success"] = confidence >= 0.3
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                torch.cuda.synchronize()
                row["latency_ms"] = (time.perf_counter() - start) * 1000
                records.append(row)
            operational = [r for r in records if r["split"] != "distractor"]
            by_size = {
                group: summarize([r for r in operational if r["size_bin"] == group])
                for group, _, _ in size_bins
            }
            output["pipelines"][name] = {
                "summary": summarize(records),
                "coverage_by_size": by_size,
                "retrieval": retrieval(records),
                "failures": [
                    {k: r[k] for k in ("name", "split", "size_bin", "error")}
                    for r in records if not r["success"]
                ],
            }

    result_path = Path("/results/tinyface_alignment_smoke.json")
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    result_volume.commit()
    print(json.dumps(output, indent=2), flush=True)
    return output


@app.local_entrypoint()
def main(
    probes_per_bin: int = 5,
    distractors: int = 100,
    pipelines: str = "square,dfa_mobilenet,dfa_resnet50,scrfd10g,mediapipe",
):
    result = benchmark.remote(probes_per_bin, distractors, pipelines)
    local = Path(__file__).with_name("last_modal_result.json")
    local.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved returned result to {local}")
