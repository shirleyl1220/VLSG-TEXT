import os
import subprocess

# ----- CONFIG -----
ROOT = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan"
OUT_DIR = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/scene_graphs"
MAX_SCENES = 100   # change to any number
SCRIPT = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/utils/build_scene_graph_from_raw.py"
# -------------------

os.makedirs(OUT_DIR, exist_ok=True)

scenes = sorted(os.listdir(ROOT))
count = 0


for scene_id in scenes:
    scene_path = os.path.join(ROOT, scene_id)
    if not os.path.isdir(scene_path):
        continue

    ply = os.path.join(scene_path, "labels.instances.annotated.v2.ply")
    semseg = os.path.join(scene_path, "semseg.v2.json")

    if not (os.path.exists(ply) and os.path.exists(semseg)):
        print(f"[SKIP] Missing files for {scene_id}")
        continue

    out_path = os.path.join(OUT_DIR, f"{scene_id}.json")

    print(f"\n[RUNNING] {scene_id}")
    cmd = [
        "python", SCRIPT,
        "--ply", ply,
        "--semseg", semseg,
        "--out", out_path
    ]
    subprocess.run(cmd)

    count += 1
    if count >= MAX_SCENES:
        break

print(f"\n[DONE] Generated {count} scene graphs.")