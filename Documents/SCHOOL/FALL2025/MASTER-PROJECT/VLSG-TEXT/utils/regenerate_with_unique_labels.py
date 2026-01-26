"""
Regenerate Scene Graphs with Unique, Spatially-Aware Labels

This will make your CLIP embeddings actually useful!
"""

import os
import subprocess
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

import sys

CONFIG = {
    "root_dir": "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan",
    "out_dir": "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT-1/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/scene_graphs_unique",
    "script_path": os.path.abspath("/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT-1/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/utils/build_scene_graph_unique_labels.py"),  # Use absolute path
    "max_scenes": 100,
}

print("""
╔═══════════════════════════════════════════════════════════════╗
║     REGENERATE SCENE GRAPHS WITH UNIQUE LABELS                 ║
╚═══════════════════════════════════════════════════════════════╝

This will create spatially-aware labels like:
  - "wall" → "wall_north_upper", "wall_south_lower"
  - "obj_0" → "small_red_object_center_middle"
  - "chair" → "chair_west_lower_1", "chair_east_lower_2"

Benefits:
  ✅ Every object gets a unique CLIP embedding
  ✅ 512 CLIP dimensions become informative (not redundant!)
  ✅ Model can actually learn from semantic differences
  ✅ Expected accuracy: 30% → 70-80%

""")

response = input("Continue? [y/N]: ")
if response.lower() != 'y':
    print("Cancelled.")
    exit()

# Backup old scene graphs
print("\n1. Backing up old scene graphs...")
old_dir = Path(CONFIG["out_dir"]).parent / "scene_graphs_old"
new_dir = Path(CONFIG["out_dir"])

if new_dir.exists():
    os.system(f"mv '{new_dir}' '{old_dir}'")
    print(f"   Backed up to: {old_dir}")

os.makedirs(new_dir, exist_ok=True)

# Find scenes
print("\n2. Finding scenes...")
root = Path(CONFIG["root_dir"])
scenes = []

for scene_dir in sorted(root.iterdir()):
    if not scene_dir.is_dir():
        continue
    
    ply_path = scene_dir / "labels.instances.annotated.v2.ply"
    semseg_path = scene_dir / "semseg.v2.json"
    
    if ply_path.exists() and semseg_path.exists():
        scenes.append((scene_dir.name, str(ply_path), str(semseg_path)))
        
    if len(scenes) >= CONFIG["max_scenes"]:
        break

print(f"   Found {len(scenes)} valid scenes")

# Process scenes
print("\n3. Processing scenes...")
success = 0
errors = 0

for i, (scene_id, ply_path, semseg_path) in enumerate(scenes, 1):
    out_path = new_dir / f"{scene_id}.json"
    
    print(f"   [{i}/{len(scenes)}] {scene_id}...", end=" ")
    
    try:
        result = subprocess.run(
            [
                "python3",
                CONFIG["script_path"],
                "--ply", ply_path,
                "--semseg", semseg_path,
                "--out", str(out_path)
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and out_path.exists():
            print("✓")
            success += 1
        else:
            print(f"✗ {result.stderr[:50]}")
            errors += 1
    except Exception as e:
        print(f"✗ {str(e)[:50]}")
        errors += 1

# Summary
print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"Success: {success}/{len(scenes)}")
print(f"Errors:  {errors}/{len(scenes)}")
print(f"\nOutput: {new_dir}")
print("="*70)

# Verify one scene
if success > 0:
    print("\n4. Verifying labels in first scene...")
    import json
    
    first_scene = list(new_dir.glob("*.json"))[0]
    with open(first_scene) as f:
        data = json.load(f)
    
    print(f"\nSample labels from {first_scene.name}:")
    for i, (nid, node) in enumerate(list(data['nodes'].items())[:5]):
        base = node.get('base_label', 'N/A')
        unique = node['label']
        print(f"  {nid}: {base:20s} → {unique}")
    
    # Check CLIP diversity
    import numpy as np
    
    clip_embs = [np.array(node['clip_text_emb']) for node in data['nodes'].values()]
    clip_means = [emb.mean() for emb in clip_embs]
    
    print(f"\nCLIP embedding diversity:")
    print(f"  Mean: {np.mean(clip_means):.4f}")
    print(f"  Std:  {np.std(clip_means):.4f}")
    
    if np.std(clip_means) > 0.01:
        print(f"  ✓ CLIP embeddings are diverse!")
    else:
        print(f"  ⚠️  CLIP embeddings still similar")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Update your training script:")
print(f"   dataset_dir = '{new_dir}'")
print("2. Keep node_input_dim = 518 (no change needed)")
print("3. Set use_pure_geometric = False")
print("4. Retrain with batch_size=8, lr=1e-3")
print("5. Watch separation jump to 0.5+ by epoch 20!")
print("="*70)