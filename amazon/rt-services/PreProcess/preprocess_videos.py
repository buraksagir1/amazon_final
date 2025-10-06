#!/usr/bin/env python3
"""
Preprocess ASL videos → MediaPipe landmark tensors (.npz) ready for skeleton-space
augmentation / model training.

Improvements over the original script
─────────────────────────────────────
1. **FPS standardisation** – resamples every video to a common target_fps (default = 30) so every
   sample has a comparable temporal resolution.  # ADDED
2. **Landmark normalisation** – each frame is root-centred (mid-hip) and scaled so the average
   bone length ≈ 1 (unit-norm), making models camera-distance invariant.  # ADDED
3. **Visibility channel** – stores a 4-vector (x,y,z,v) per joint; `v` combines MediaPipe’s
   visibility / presence or defaults to 0.  # ADDED
4. **Missing‐frame handling** – uses forward-fill (or zeros if first frame) instead of padding
   with zeros only, which stabilises temporal derivatives.  # ADDED
5. **Multiprocessing** – optional `--num_workers` speeds up large datasets via
   `ProcessPoolExecutor`.  # ADDED
6. **Manifest logging** – each processed clip is recorded to a `.jsonl` manifest for future
   reproducibility / debugging.  # ADDED

Usage
─────
python preprocess_videos.py \
    --input_root data/ \
    --output_root dataset_root/ \
    --target_fps 30 \
    --num_workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp

# --------------------------- LANDMARK INDICES ---------------------------
POSE_IDS: List[int] = list(range(0, 33))                # 0–32 pose
HAND_L_IDS: List[int] = list(range(0, 21))              # 0–20 left hand
HAND_R_IDS: List[int] = list(range(0, 21))              # 0–20 right hand
FACE_IDS: List[int] = [  # 15 salient facial points
    1, 33, 263, 61, 291, 199, 234, 454, 127,
    356, 152, 168, 94, 323, 93,
]

# Final joint ordering: pose(33) + left-hand(21) + right-hand(21) + face(15)
V = 33 + 21 + 21 + 15  # = 90

# --------------------------- MEDIAPIPE SETUP ---------------------------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# --------------------------- UTILS ---------------------------

# (T,V,4) -> (T,V,4)
def interpolate_missing(seq):
    xyz = seq[..., :3]
    vis = np.any(xyz != 0, axis=-1)  # (T,V)
    for v in range(xyz.shape[1]):
        valid = vis[:, v]
        if valid.any():
            t = np.arange(xyz.shape[0])
            for c in range(3):
                xyz[~valid, v, c] = np.interp(t[~valid], t[valid], xyz[valid, v, c])
        else:
            # Tümü yoksa sıfır bırak
            pass
    seq[..., :3] = xyz
    return seq


def normalise_sequence(seq: np.ndarray) -> np.ndarray:
    """Root-centre and scale a (T, V, C) landmark array.

    Root = mid-hip ≈ mean(pose[23], pose[24]) in MediaPipe indexing.
    Scale = average distance from root to both shoulders; fallback to 1.
    Returns the same shaped array.
    """
    if seq.ndim != 3:
        raise ValueError("seq must be (T,V,C)")

    T, _, C = seq.shape
    seq_norm = seq.copy()

    # -- root centre per frame
    left_hip, right_hip = 23, 24  # MediaPipe pose ids
    root = (seq_norm[:, left_hip] + seq_norm[:, right_hip]) / 2  # (T,3)
    seq_norm -= root[:, None, :]

    # -- scale per sequence (use shoulder span averaged over frames)
    l_sh, r_sh = 11, 12
    valid = np.linalg.norm(seq_norm[:, l_sh] - seq_norm[:, r_sh], axis=1) > 0
    if valid.any():
        scale = np.mean(np.linalg.norm(seq_norm[valid][:, l_sh] - seq_norm[valid][:, r_sh], axis=1))
    else:
        scale = 1.0
    if scale < 1e-6:
        scale = 1.0
    seq_norm /= scale
    return seq_norm


def resample_indices(n_frames, src_fps, target_fps):
    if src_fps <= 0 or target_fps <= 0:
        return list(range(n_frames))
    duration = n_frames / src_fps
    t = np.arange(0, duration, 1/target_fps)
    idx = np.clip((t * src_fps).astype(int), 0, n_frames - 1)
    # ardışık tekrarları kaldır (ama sırayı koru)
    return [idx[0], *[i for i in idx[1:] if i != i-1]]

# --------------------------- EXTRACTOR ---------------------------

def extract_landmarks(
    res_pose: mp.framework.formats.landmark_pb2.NormalizedLandmarkList | None,
    res_hands: mp.framework.formats.landmark_pb2.NormalizedLandmarkList | None,
    res_face: mp.framework.formats.landmark_pb2.NormalizedLandmarkList | None,
) -> np.ndarray:
    """Extract (V,4) array [x,y,z,visibility] from MediaPipe results."""
    lm = np.zeros((V, 4), dtype=np.float32)  # default zeros = not visible

    # Pose
    if res_pose and res_pose.pose_landmarks:
        for i, p in enumerate(POSE_IDS):
            l = res_pose.pose_landmarks.landmark[p]
            lm[i, :3] = (l.x, l.y, l.z)
            lm[i, 3] = l.visibility

    # Hands
    if res_hands and res_hands.multi_hand_landmarks and res_hands.multi_handedness:
        for h_lmks, handedness in zip(res_hands.multi_hand_landmarks, res_hands.multi_handedness):
            offset = 33 if handedness.classification[0].label == "Left" else 33 + 21
            for j in range(21):
                l = h_lmks.landmark[j]
                lm[offset + j, :3] = (l.x, l.y, l.z)
                lm[offset + j, 3] = 1.0  # MediaPipe hands lacks explicit visibility

    # Face (first face only)
    if res_face and res_face.multi_face_landmarks:
        for k, idx in enumerate(FACE_IDS):
            l = res_face.multi_face_landmarks[0].landmark[idx]
            lm[33 + 42 + k, :3] = (l.x, l.y, l.z)
            lm[33 + 42 + k, 3] = 1.0  # assume visible if detected

    return lm

# --------------------------- REAL-TIME: SINGLE-FRAME → (V,4) ---------------------------
# Kalıcı (lazy) MediaPipe objeleri: kare başına yeniden oluşturmayalım
_rt_hands = None
_rt_face = None
_rt_pose = None

def _ensure_rt_solutions():
    """Lazy init of MediaPipe solutions for real-time single-frame extraction."""
    global _rt_hands, _rt_face, _rt_pose
    if _rt_hands is None:
        _rt_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    if _rt_face is None:
        _rt_face = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    if _rt_pose is None:
        _rt_pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

def frame_to_v4(
    frame_bgr: np.ndarray,
    normalise: bool = True,
    V_target: int | None = None,
) -> np.ndarray:
    """
    Tek kare → (V,4) float32 döndürür (V = 90: pose33 + Lhand21 + Rhand21 + face15).
    - Eğitim pipeline'ıyla tutarlı olacak şekilde aynı `extract_landmarks` düzenini kullanır.
    - `normalise=True` iken, tek karelik dizi üzerinde `normalise_sequence` (root-centre + scale) uygular.
    - `V_target` verilirse V boyutunu pad/trim ile eşler (ör. modelin beklediği V = F/4).
    """
    _ensure_rt_solutions()

    # MediaPipe RGB ister
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    res_hands = _rt_hands.process(rgb)
    res_face = _rt_face.process(rgb)
    res_pose = _rt_pose.process(rgb)

    lm = extract_landmarks(res_pose, res_hands, res_face)  # (V,4)

    if normalise:
        # (1,V,3) üzerinde eğitimdeki normalizasyonu uygula
        seq = lm[None, :, :3]
        seq = normalise_sequence(seq)
        lm[:, :3] = seq[0]

    # V hedefine eşle (gerekliyse)
    if V_target is not None and int(V_target) != lm.shape[0]:
        Vt = int(V_target)
        if lm.shape[0] > Vt:
            lm = lm[:Vt]
        else:
            pad = np.zeros((Vt - lm.shape[0], 4), dtype=np.float32)
            lm = np.vstack([lm, pad])

    return lm.astype(np.float32)

# --------------------------- VIDEO PROCESS ---------------------------

def process_video(
    vid_path: Path,
    target_fps: int = 30,
    normalise: bool = True,
) -> Tuple[np.ndarray, float]:
    """Return (T,V,4) float32 array and src_fps."""

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV failed to open video: " + str(vid_path))

    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idx = set(resample_indices(total_frames, src_fps, target_fps))  # set for O(1) test

    frames: List[np.ndarray] = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands, mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx in sample_idx:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res_hands = hands.process(rgb)
                res_face = face.process(rgb)
                res_pose = pose.process(rgb)

                lm = extract_landmarks(res_pose, res_hands, res_face)
                frames.append(lm)
            f_idx += 1
        cap.release()

    if not frames:
        raise RuntimeError(f"{vid_path}: No frames were processed.")

    seq = np.stack(frames)  # (T,V,4)

    # -- forward fill missing frames  # ADDED
    mask_valid = np.any(seq[:, :, :3] != 0, axis=(1, 2))
    for i in range(1, len(seq)):
        if not mask_valid[i]:
            seq[i] = seq[i - 1]
    seq = interpolate_missing(seq)
    # -- optional normalisation  # ADDED
    if normalise:
        seq[:, :, :3] = normalise_sequence(seq[:, :, :3])

    return seq.astype(np.float32), src_fps

# --------------------------- WORKER ---------------------------

def _worker(
    args: Tuple[Path, Path, str, int, bool],
):
    vid_path, out_dir, label, target_fps, normalise = args
    out_file = out_dir / f"{vid_path.stem}.npz"
    if out_file.exists():
        return None  # skip existing
    try:
        arr, src_fps = process_video(vid_path, target_fps, normalise)
        np.savez_compressed(out_file, data=arr)
        return {
            "file": str(out_file),
            "label": label,
            "n_frames": int(arr.shape[0]),
            "src_fps": src_fps,
        }
    except Exception as e:
        return {"error": str(e), "video": str(vid_path)}

# --------------------------- MAIN ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--target_fps", type=int, default=30, help="Resample videos to this FPS")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel workers (0=main proc)")
    parser.add_argument("--normalise", action="store_true", help="Apply root+scale normalisation")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")

    args = parser.parse_args()
    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # -- gather labels
    labels = sorted([p.name for p in in_root.iterdir() if p.is_dir()])
    label_map = {lbl: idx for idx, lbl in enumerate(labels)}
    with open(out_root / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print("label_map.json written to", out_root / "label_map.json")

    manifest_path = out_root / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        tasks = []
        for lbl in labels:
            in_dir = in_root / lbl
            out_dir = out_root / lbl
            out_dir.mkdir(exist_ok=True)
            for vid in in_dir.glob("*.mp4"):
                out_file = out_dir / f"{vid.stem}.npz"
                if out_file.exists() and not args.force:
                    continue
                tasks.append((vid, out_dir, lbl, args.target_fps, args.normalise))

        print(f"{len(tasks)} videos to process…")

        if args.num_workers > 1:
            with futures.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
                for res in pool.map(_worker, tasks):
                    if res is None:
                        continue
                    manifest_file.write(json.dumps(res) + "\n")
        else:
            for t in tasks:
                res = _worker(t)
                if res is None:
                    continue
                manifest_file.write(json.dumps(res) + "\n")

    print("Dataset extraction finished →", out_root)
    print("Manifest saved →", manifest_path)

if __name__ == "__main__":
    main()
