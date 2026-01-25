"""
Batch process multiple 3RScan scenes to generate scene graphs.

Usage:
    python batch_build_scene_graphs.py
    
    # Or with custom parameters:
    python batch_build_scene_graphs.py --max_scenes 100 --overwrite
"""

import os
import subprocess
import argparse
from pathlib import Path
import time


# ============================================================
# Configuration
# ============================================================

DEFAULT_CONFIG = {
    "root_dir": "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan",
    "out_dir": "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT-1/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/scene_graphs",
    "script_path": "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT-1/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/utils/build_scene_graph_from_raw.py",
    "max_scenes": 100,
}


# ============================================================
# Scene Discovery
# ============================================================

def find_valid_scenes(root_dir: str, max_scenes: int = None):
    """
    Find all valid scene directories with required files.
    
    Returns:
        List of (scene_id, ply_path, semseg_path) tuples
    """
    root = Path(root_dir)
    valid_scenes = []
    
    print("\n" + "="*70)
    print("Scanning for valid scenes...")
    print("="*70)
    
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        scene_id = scene_dir.name
        
        # Check for required files
        ply_path = scene_dir / "labels.instances.annotated.v2.ply"
        semseg_path = scene_dir / "semseg.v2.json"
        
        if not ply_path.exists():
            print(f"  ⊗ {scene_id}: Missing PLY file")
            continue
        
        if not semseg_path.exists():
            print(f"  ⊗ {scene_id}: Missing semseg file")
            continue
        
        valid_scenes.append((scene_id, str(ply_path), str(semseg_path)))
        print(f"  ✓ {scene_id}")
        
        if max_scenes and len(valid_scenes) >= max_scenes:
            print(f"\n  Reached max_scenes limit ({max_scenes})")
            break
    
    print(f"\nFound {len(valid_scenes)} valid scenes")
    print("="*70 + "\n")
    
    return valid_scenes


# ============================================================
# Batch Processing
# ============================================================

def process_scene(
    scene_id: str,
    ply_path: str,
    semseg_path: str,
    out_dir: str,
    script_path: str,
    skip_existing: bool = True
):
    """
    Process a single scene by calling the build_scene_graph script.
    
    Returns:
        (success: bool, message: str)
    """
    out_path = Path(out_dir) / f"{scene_id}.json"
    
    # Check if already processed
    if skip_existing and out_path.exists():
        return True, "Already exists (skipped)"
    
    try:
        # Call the build_scene_graph script
        cmd = [
            "python3",
            script_path,
            "--ply", ply_path,
            "--semseg", semseg_path,
            "--out", str(out_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout per scene
        )
        
        if result.returncode != 0:
            # Script failed
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return False, f"✗ Script failed: {error_msg[:100]}"
        
        # Check if output file was created
        if not out_path.exists():
            return False, "✗ Output file not created"
        
        # Parse the output to get stats (optional)
        try:
            import json
            with open(out_path, 'r') as f:
                data = json.load(f)
            
            n_nodes = len(data.get('nodes', {}))
            n_edges = len(data.get('edges_text', []))
            
            msg = f"✓ {n_nodes} nodes, {n_edges} edges"
        except:
            msg = "✓ Created"
        
        return True, msg
    
    except subprocess.TimeoutExpired:
        return False, "✗ Timeout (>60s)"
    
    except Exception as e:
        return False, f"✗ Error: {str(e)}"


def batch_process_scenes(
    root_dir: str,
    out_dir: str,
    script_path: str,
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
    errors = []
    
    start_time = time.time()
    
    for i, (scene_id, ply_path, semseg_path) in enumerate(valid_scenes, 1):
        print(f"[{i}/{len(valid_scenes)}] Processing: {scene_id}")
        
        success, message = process_scene(
            scene_id,
            ply_path,
            semseg_path,
            out_dir,
            script_path,
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
            errors.append((scene_id, message))
    
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
    
    if errors:
        print(f"\nErrors encountered:")
        for scene_id, msg in errors[:10]:  # Show first 10 errors
            print(f"  - {scene_id}: {msg}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
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
        "--script_path",
        type=str,
        default=DEFAULT_CONFIG["script_path"],
        help="Path to build_scene_graph_from_raw.py"
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
        default=True,
        help="Skip scenes that already have output files (default: True)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    
    args = parser.parse_args()
    
    # Handle skip_existing vs overwrite
    skip_existing = not args.overwrite
    
    # Print configuration
    print("\n" + "="*70)
    print("Batch Scene Graph Generator")
    print("="*70)
    print(f"Root directory:    {args.root_dir}")
    print(f"Output directory:  {args.out_dir}")
    print(f"Script path:       {args.script_path}")
    print(f"Max scenes:        {args.max_scenes}")
    print(f"Skip existing:     {skip_existing}")
    print("="*70)
    
    # Check if paths exist
    if not os.path.exists(args.root_dir):
        print(f"\n❌ Error: Root directory not found: {args.root_dir}")
        return
    
    if not os.path.exists(args.script_path):
        print(f"\n❌ Error: Script not found: {args.script_path}")
        return
    
    # Confirmation prompt
    if args.overwrite:
        response = input("\n⚠️  This will OVERWRITE existing scene graphs. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Run batch processing
    batch_process_scenes(
        args.root_dir,
        args.out_dir,
        args.script_path,
        args.max_scenes,
        skip_existing
    )


if __name__ == "__main__":
    main()