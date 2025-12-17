
"""
Batch process multiple 3RScan scenes to generate scene graphs.

Usage:
    python batch_build_scene_graphs.py
    
    # Or with custom parameters:
    python batch_build_scene_graphs.py --max_scenes 50 --skip_existing
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple
import time

# Import the scene graph builder
# Make sure build_scene_graph.py is in the same directory or adjust the import
from build_scene_graph import build_scene_graph


# ============================================================
# Configuration
# ============================================================

DEFAULT_CONFIG = {
    "root_dir": "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan",
    "out_dir": "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/scene_graphs",
    "objects_json": "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/meta_files/objects.json",
    "max_scenes": 100,
    "skip_existing": True,  # Skip scenes that already have output files
}


# ============================================================
# Scene Discovery
# ============================================================

def find_valid_scenes(root_dir: str, max_scenes: int = None) -> List[Tuple[str, str]]:
    """
    Find all valid scene directories with required files.
    
    Returns:
        List of (scene_id, scene_path) tuples
    """
    root = Path(root_dir)
    valid_scenes = []
    
    print("Scanning for valid scenes...")
    
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        scene_id = scene_dir.name
        
        # Check for required files
        ply_path = scene_dir / "labels.instances.annotated.v2.ply"
        
        if not ply_path.exists():
            print(f"  ⊗ {scene_id}: Missing PLY file")
            continue
        
        # Optional: Check for description file
        desc_path = scene_dir / "output" / "descriptions" / "all_descriptions.json"
        if not desc_path.exists():
            print(f"  ⚠ {scene_id}: No descriptions (will process anyway)")
        
        valid_scenes.append((scene_id, str(scene_dir)))
        
        if max_scenes and len(valid_scenes) >= max_scenes:
            break
    
    print(f"\nFound {len(valid_scenes)} valid scenes")
    return valid_scenes


# ============================================================
# Batch Processing
# ============================================================

def process_scene(
    scene_id: str,
    scene_path: str,
    out_dir: str,
    objects_json: str,
    skip_existing: bool = True
) -> Tuple[bool, str]:
    """
    Process a single scene and save the scene graph.
    
    Returns:
        (success: bool, message: str)
    """
    out_path = Path(out_dir) / f"{scene_id}.json"
    
    # Check if already processed
    if skip_existing and out_path.exists():
        return True, "Already exists (skipped)"
    
    try:
        # Build scene graph
        scene_graph = build_scene_graph(scene_path, objects_json)
        
        # Save to file
        with open(out_path, 'w') as f:
            json.dump(scene_graph, f, indent=2)
        
        # Return success with stats
        n_nodes = len(scene_graph['nodes'])
        n_edges_geom = len(scene_graph['edges_geometric'])
        n_edges_text = len(scene_graph['edges_text'])
        
        msg = f"✓ {n_nodes} nodes, {n_edges_geom} geom edges, {n_edges_text} text edges"
        return True, msg
    
    except FileNotFoundError as e:
        return False, f"✗ Missing file: {e}"
    
    except Exception as e:
        return False, f"✗ Error: {str(e)}"


def batch_process_scenes(
    root_dir: str,
    out_dir: str,
    objects_json: str,
    max_scenes: int = None,
    skip_existing: bool = True
):
    """
    Process multiple scenes in batch.
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Find valid scenes
    valid_scenes = find_valid_scenes(root_dir, max_scenes)
    
    if not valid_scenes:
        print("No valid scenes found!")
        return
    
    # Process each scene
    print("\n" + "="*70)
    print("Starting batch processing...")
    print("="*70 + "\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    start_time = time.time()
    
    for i, (scene_id, scene_path) in enumerate(valid_scenes, 1):
        print(f"[{i}/{len(valid_scenes)}] Processing: {scene_id}")
        
        success, message = process_scene(
            scene_id,
            scene_path,
            out_dir,
            objects_json,
            skip_existing
        )
        
        print(f"    {message}")
        
        if success:
            if "skipped" in message.lower():
                skip_count += 1
            else:
                success_count += 1
        else:
            error_count += 1
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("Batch Processing Complete")
    print("="*70)
    print(f"Total scenes processed: {len(valid_scenes)}")
    print(f"  ✓ Successfully created: {success_count}")
    print(f"  ⊙ Skipped (existing):   {skip_count}")
    print(f"  ✗ Errors:               {error_count}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/len(valid_scenes):.1f}s per scene)")
    print(f"\nOutput directory: {out_dir}")
    print("="*70)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch generate scene graphs from 3RScan data"
    )
    
    parser.add_argument(
        "--root_dir",
        type=str,
        default=DEFAULT_CONFIG["root_dir"],
        help="Root directory containing scene folders"
    )
    
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_CONFIG["out_dir"],
        help="Output directory for scene graph JSONs"
    )
    
    parser.add_argument(
        "--objects_json",
        type=str,
        default=DEFAULT_CONFIG["objects_json"],
        help="Path to objects.json metadata file"
    )
    
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=DEFAULT_CONFIG["max_scenes"],
        help="Maximum number of scenes to process (default: 100)"
    )
    
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=DEFAULT_CONFIG["skip_existing"],
        help="Skip scenes that already have output files"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (opposite of --skip_existing)"
    )
    
    args = parser.parse_args()
    
    # Handle skip_existing vs overwrite
    skip_existing = args.skip_existing and not args.overwrite
    
    # Print configuration
    print("\n" + "="*70)
    print("Batch Scene Graph Generator")
    print("="*70)
    print(f"Root directory:    {args.root_dir}")
    print(f"Output directory:  {args.out_dir}")
    print(f"Objects JSON:      {args.objects_json}")
    print(f"Max scenes:        {args.max_scenes}")
    print(f"Skip existing:     {skip_existing}")
    print("="*70 + "\n")
    
    # Check if paths exist
    if not os.path.exists(args.root_dir):
        print(f"Error: Root directory not found: {args.root_dir}")
        return
    
    if not os.path.exists(args.objects_json):
        print(f"Warning: objects.json not found: {args.objects_json}")
        print("Scene graphs will have generic labels.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            return
    
    # Run batch processing
    batch_process_scenes(
        args.root_dir,
        args.out_dir,
        args.objects_json,
        args.max_scenes,
        skip_existing
    )


if __name__ == "__main__":
    main()
    # import os
# import subprocess

# # ----- CONFIG -----
# ROOT = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan"
# OUT_DIR = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/scene_graphs"
# MAX_SCENES = 100   # change to any number
# SCRIPT = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/utils/build_scene_graph_from_raw.py"
# # -------------------

# os.makedirs(OUT_DIR, exist_ok=True)

# scenes = sorted(os.listdir(ROOT))
# count = 0


# for scene_id in scenes:
#     scene_path = os.path.join(ROOT, scene_id)
#     if not os.path.isdir(scene_path):
#         continue

#     ply = os.path.join(scene_path, "labels.instances.annotated.v2.ply")
#     semseg = os.path.join(scene_path, "semseg.v2.json")

#     if not (os.path.exists(ply) and os.path.exists(semseg)):
#         print(f"[SKIP] Missing files for {scene_id}")
#         continue

#     out_path = os.path.join(OUT_DIR, f"{scene_id}.json")

#     print(f"\n[RUNNING] {scene_id}")
#     cmd = [
#         "python", SCRIPT,
#         "--ply", ply,
#         "--semseg", semseg,
#         "--out", out_path
#     ]
#     subprocess.run(cmd)

#     count += 1
#     if count >= MAX_SCENES:
#         break

# print(f"\n[DONE] Generated {count} scene graphs.")