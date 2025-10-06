#!/usr/bin/env python3
"""
Extract selected WLASL videos into per-gloss folders.

Usage
-----
python extract_glosses.py \
    --json WLASL_v0.3.json \
    --videos_dir Videos \
    --output_dir SelectedVideos \
    --glosses "hello,yes,no,thank you,please,car,food,stop,beautiful,ugly"
"""

import argparse
import json
import os
import shutil
from typing import List, Dict

# ---------- helpers -----------------------------------------------------------
def normalize(text: str) -> str:
    """Lower-case, strip, and normalise separators so
    'THANK-YOU', 'thank_you', 'thank you' → 'thank you'."""
    return (
        text.strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )

def load_json(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_ids(data: List[Dict], wanted: List[str]) -> Dict[str, List[str]]:
    """Return {gloss: [video_id, ...]} for the requested glosses."""
    wanted_norm = {normalize(g): g for g in wanted}        # map norm→original
    by_gloss: Dict[str, List[str]] = {g: [] for g in wanted}

    for entry in data:
        gloss_norm = normalize(entry["gloss"])
        if gloss_norm in wanted_norm:
            orig_key = wanted_norm[gloss_norm]
            for inst in entry["instances"]:
                by_gloss[orig_key].append(inst["video_id"])
    return by_gloss

def copy_videos(mapping: Dict[str, List[str]], videos_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for gloss, ids in mapping.items():
        dest_gloss_dir = os.path.join(out_dir, gloss.replace(" ", "_"))
        os.makedirs(dest_gloss_dir, exist_ok=True)

        for vid in ids:
            src = os.path.join(videos_dir, f"{vid}.mp4")
            if not os.path.exists(src):
                print(f"⚠️  Video dosyası bulunamadı: {src}")
                continue

            dst = os.path.join(dest_gloss_dir, f"{vid}.mp4")
            # Çakışma olursa üstüne yazma
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

# ---------- cli ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="WLASL_v0.3.json yolu")
    parser.add_argument("--videos_dir", required=True, help="Videos klasörü")
    parser.add_argument("--output_dir", default="SelectedVideos",
                        help="Çıktı ana klasörü")
    parser.add_argument("--glosses", required=True,
                        help="Virgülle ayrılmış gloss listesi")

    args = parser.parse_args()

    wanted = [g.strip() for g in args.glosses.split(",") if g.strip()]
    data = load_json(args.json)
    mapping = collect_ids(data, wanted)
    copy_videos(mapping, args.videos_dir, args.output_dir)

    print("✅ Tamamlandı!")

if __name__ == "__main__":
    main()
