#!/usr/bin/env python3
"""
organize_lsa_videos.py
──────────────────────
LSA videolarını ID → gloss eşlemesine göre klasörlere ayırır.

Kullanım
--------
python organize_lsa_videos.py \
    --videos_dir LSA_Videos \
    --output_dir LSA_Split

• --move   : Kopyalamak yerine taşır (isteğe bağlı)
"""

import argparse, shutil
from pathlib import Path

# ---------- ID → Gloss eşlemesi (ilk 16) ----------
ID2GLOSS = {
    "001": "Opaque",
    "002": "Red",
    "003": "Green",
    "004": "Yellow",
    "005": "Bright",
    "006": "Light-blue",
    "007": "Colors",
    "008": "Pink",
    "009": "Women",
    "010": "Enemy",
    "011": "Son",
    "012": "Man",
    "013": "Away",
    "014": "Drawer",
    "015": "Born",
    "016": "Learn",
}

# ---------- ana fonksiyon ----------
def main(videos_dir: Path, output_dir: Path, move: bool):
    output_dir.mkdir(parents=True, exist_ok=True)

    for vid in videos_dir.glob("*.mp4"):
        vid_id = vid.stem.split("_")[0]          # '001' kısmı
        if vid_id not in ID2GLOSS:
            continue                             # ilk 16 dışında → atla
        gloss = ID2GLOSS[vid_id].replace(" ", "_")
        dest_gloss_dir = output_dir / gloss
        dest_gloss_dir.mkdir(exist_ok=True)

        dest_file = dest_gloss_dir / vid.name
        if dest_file.exists():
            continue

        if move:
            shutil.move(vid, dest_file)
            action = "TAŞINDI"
        else:
            shutil.copy2(vid, dest_file)
            action = "KOPYALANDI"

        print(f"{action}: {vid.name}  →  {dest_gloss_dir}")

    print("✅ Tamamlandı.")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True, type=Path,
                    help="Tüm .mp4 dosyalarının bulunduğu klasör")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Çıktı ana klasörü (oluşturulur)")
    ap.add_argument("--move", action="store_true",
                    help="Dosyaları kopyalamak yerine taşır")
    args = ap.parse_args()
    main(args.videos_dir, args.output_dir, args.move)
