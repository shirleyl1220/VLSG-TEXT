import os
import json
import numpy as np
import torch
from plyfile import PlyData
from collections import defaultdict
from typing import Dict, List, Optional


# ============================================================
# Load semantic labels from objects.json
# ============================================================

def load_object_labels(objects_json_path: str) -> Dict[int, dict]:
    """
    Load semantic labels from 3RScan objects.json metadata.
    
    Args:
        objects_json_path: Path to objects.json
    
    Returns:
        Dict mapping global_id -> {label, ply_color, nyu40id, ...}
    """
    with open(objects_json_path, 'r') as f:
        data = json.load(f)
    
    id_to_label = {}
    for scan in data.get('scans', []):
            for obj in scan.get('objects', []):
                global_id = obj.get('global_id')
                if global_id is not None:
                    id_to_label[int(global_id)] = {
                        'label': obj.get('label', 'unknown'),
                        'ply_color': obj.get('ply_color', '#000000'),
                        'nyu40id': obj.get('nyu40'),  # Note: field is 'nyu40', not 'nyu40id'
                        'eigen13id': obj.get('eigen13'),  # Note: field is 'eigen13'
                        'rio27id': obj.get('rio27')  # Note: field is 'rio27'
                    }
    
    return id_to_label


# ============================================================
# Load relationships from relationships.json
# ============================================================

def load_relationships(relationships_json_path: str) -> List[dict]:
    """
    Load spatial relationships from relationships.json.
    
    Args:
        relationships_json_path: Path to relationships.json
    
    Returns:
        List of relationship dictionaries with subject, object, relation
    """
    if not os.path.exists(relationships_json_path):
        print(f"Warning: relationships.json not found at {relationships_json_path}")
        return []
    
    with open(relationships_json_path, 'r') as f:
        data = json.load(f)
    

    
    # Convert to our format
    text_edges = []
    for scan in data.get('scans', []):
        for rel in scan.get('relationships', []):
            # rel is a list: [subject_id, object_id, relation_id, relation_text]
            if len(rel) >= 4:
                text_edges.append({
                    'subject': str(rel[0]),      # first element
                    'object': str(rel[1]),       # second element  
                    'relation': rel[3].lower().strip()  # fourth element (text)
                })
    
    print(f"Loaded {len(text_edges)} relationships from relationships.json")
        
    return text_edges


# ============================================================
# Geometric Feature Extraction
# ============================================================

def compute_geometric_features(vertices, normalize: bool = True):
    """
    Extract rich geometric features from point cloud vertices.
    """
    xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
    
    # Basic statistics
    centroid = xyz.mean(axis=0).astype(float)
    std_dev = xyz.std(axis=0).astype(float)
    
    # Color features
    if 'red' in vertices.dtype.names:
        rgb = np.stack([vertices['red'], vertices['green'], vertices['blue']], axis=1)
        mean_color = rgb.mean(axis=0).astype(float)
        std_color = rgb.std(axis=0).astype(float)
    else:
        mean_color = np.array([0., 0., 0.])
        std_color = np.array([0., 0., 0.])
    
    # Shape features
    centered = xyz - centroid
    distances = np.linalg.norm(centered, axis=1)
    radius = distances.mean().astype(float)
    
    # PCA for orientation and extent
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    
    # Sort eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    
    # Shape descriptors
    extent = np.sqrt(eigenvalues)
    l1, l2, l3 = eigenvalues
    linearity = (l1 - l2) / (l1 + 1e-8)
    planarity = (l2 - l3) / (l1 + 1e-8)
    sphericity = l3 / (l1 + 1e-8)
    
    # Bounding box
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    # Normalize colors to [0, 1]
    if normalize:
        mean_color = mean_color / 255.0
        std_color = std_color / 255.0
    
    return {
        "centroid": centroid.tolist(),
        "mean_color": mean_color.tolist(),
        "radius": float(radius),
        
        # Additional geometric features
        "geometric_features": {
            "std_dev": std_dev.tolist(),
            "std_color": std_color.tolist(),
            "extent": extent.tolist(),
            "linearity": float(linearity),
            "planarity": float(planarity),
            "sphericity": float(sphericity),
            "bbox_size": bbox_size.tolist(),
            "num_points": int(len(vertices))
        }
    }


def compute_object_stats(vertices, label_info: Optional[dict] = None):
    """
    Compute object statistics with semantic labels.
    """
    features = compute_geometric_features(vertices, normalize=False)
    
    output = {
        "label": label_info['label'] if label_info else "unknown",
        "centroid": features["centroid"],
        "mean_color": features["mean_color"],
        "radius": features["radius"]
    }
    
    if label_info:
        output["nyu40id"] = label_info.get('nyu40id')
        output["eigen13id"] = label_info.get('eigen13id')
        output["ply_color"] = label_info.get('ply_color')
    
    output["geometric_features"] = features["geometric_features"]
    
    return output


# ============================================================
# Geometric Relations (keeping original format)
# ============================================================

def compute_pairwise_relations(object_stats: Dict[str, dict]) -> List[dict]:
    """
    Build pairwise geometric relationships.
    """
    obj_ids = list(object_stats.keys())
    edges = []

    for i in range(len(obj_ids)):
        for j in range(i+1, len(obj_ids)):
            id1 = obj_ids[i]
            id2 = obj_ids[j]
            
            c1 = np.array(object_stats[id1]["centroid"])
            c2 = np.array(object_stats[id2]["centroid"])

            direction = c2 - c1
            distance = float(np.linalg.norm(direction))
            
            if distance > 1e-6:
                direction_norm = (direction / distance).tolist()
            else:
                direction_norm = [0., 0., 0.]

            # 8-d edge attributes
            diff = c2 - c1
            dx, dy, dz = diff.tolist()
            dist = float(distance)
            n1 = float(np.linalg.norm(c1))
            n2 = float(np.linalg.norm(c2))
            
            if n1 > 1e-6 and n2 > 1e-6:
                cosang = float(np.dot(c1, c2) / (n1 * n2))
            else:
                cosang = 1.0
            sinang = float(np.sqrt(max(0.0, 1.0 - cosang*cosang)))

            # Forward edge
            edges.append({
                "subject": str(id1),
                "object": str(id2),
                "distance": dist,
                "direction": direction_norm,
                "edge_attr": [dx, dy, dz, dist, n1, n2, cosang, sinang]
            })
            
            # Reverse edge
            edges.append({
                "subject": str(id2),
                "object": str(id1),
                "distance": dist,
                "direction": [-direction_norm[0], -direction_norm[1], -direction_norm[2]],
                "edge_attr": [-dx, -dy, -dz, dist, n2, n1, cosang, sinang]
            })

    return edges


# ============================================================
# Main Scene Graph Builder
# ============================================================

def build_scene_graph(
    scene_dir: str,
    objects_json_path: str = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/meta_files/objects.json"
):
    """
    Build scene graph from 3RScan data with proper semantic labels and relationships.
    
    Args:
        scene_dir: Path to scene folder
        objects_json_path: Path to objects.json metadata
    """
    ply_path = os.path.join(scene_dir, "labels.instances.annotated.v2.ply")
    relationships_path = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/meta_files/relationships.json"
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    # Load semantic labels
    print(f"Loading labels from: {objects_json_path}")
    if os.path.exists(objects_json_path):
        id_to_label = load_object_labels(objects_json_path)
        print(f"Loaded {len(id_to_label)} object labels")
    else:
        print(f"Warning: objects.json not found at {objects_json_path}")
        id_to_label = {}

    # Load text relationships
    print(f"Loading relationships from: {relationships_path}")
    text_edges = load_relationships(relationships_path)

    # Load PLY file
    print(f"Loading PLY: {ply_path}")
    with open(ply_path, "rb") as f:
        ply = PlyData.read(f)
    
    ply_data = ply["vertex"].data
    obj_ids = ply_data["objectId"]

    # Group points by objectId
    object_points = defaultdict(list)
    for idx, oid in enumerate(obj_ids):
        object_points[int(oid)].append(idx)

    print(f"Found {len(object_points)} objects in scene")

    # Compute per-object statistics
    object_stats = {}
    for oid, idxs in object_points.items():
        vertices = ply_data[idxs]
        
        # Get semantic label
        label_info = id_to_label.get(oid, None)
        
        # Compute features
        stats = compute_object_stats(vertices, label_info)
        object_stats[str(oid)] = stats

    # Compute geometric relationships
    geometric_edges = compute_pairwise_relations(object_stats)
    print(f"Created {len(geometric_edges)} geometric edges")

    # Filter text edges to only include objects that exist
    valid_text_edges = []
    for edge in text_edges:
        if edge['subject'] in object_stats and edge['object'] in object_stats:
            valid_text_edges.append(edge)
    
    print(f"Filtered text edges: {len(text_edges)} -> {len(valid_text_edges)} (kept only existing objects)")

    # Build final scene graph
    scene_graph = {
        "scene_id": os.path.basename(scene_dir),
        "nodes": object_stats,
        "edges_text": valid_text_edges,  # ← TEXT RELATIONS FROM relationships.json!
        "edges_geometric": geometric_edges
    }

    return scene_graph


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build scene graph with semantic labels and relationships"
    )
    parser.add_argument(
        "--scene_path",
        type=str,
        required=True,
        help="Path to 3RScan scene folder"
    )
    parser.add_argument(
        "--objects_json",
        type=str,
        default="/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/meta_files/objects.json",
        help="Path to objects.json metadata file"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save output JSON"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Building Scene Graph with Relationships")
    print("=" * 70)

    sg = build_scene_graph(args.scene_path, args.objects_json)

    print("\n✅ Scene Graph Built Successfully")
    print("-" * 70)
    print(f"Scene ID:         {sg['scene_id']}")
    print(f"Objects:          {len(sg['nodes'])}")
    print(f"Geometric edges:  {len(sg['edges_geometric'])}")
    print(f"Text relations:   {len(sg['edges_text'])}")
    
    # Show sample text relations
    if sg['edges_text']:
        print(f"\nSample text relations:")
        for rel in sg['edges_text'][:5]:
            subj_label = sg['nodes'][rel['subject']]['label']
            obj_label = sg['nodes'][rel['object']]['label']
            print(f"  {subj_label} --[{rel['relation']}]-> {obj_label}")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(sg, f, indent=2)
        print(f"\n💾 Saved to: {args.save}")
        print("=" * 70)


if __name__ == "__main__":
    main()