import os
import json
import numpy as np
from plyfile import PlyData
from collections import defaultdict

# ============================================================
# Utility functions
# ============================================================

def load_ply(path):
    """Load PLY file and return vertex data"""
    with open(path, "rb") as f:
        ply = PlyData.read(f)
    v = ply["vertex"].data
    return v


def compute_object_stats(vertices):
    """
    vertices: structured numpy array for this object
    Returns: dict of centroid, mean_color, radius
    """
    xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
    centroid = xyz.mean(axis=0).astype(float)

    if 'red' in vertices.dtype.names:
        rgb = np.stack([vertices['red'], vertices['green'], vertices['blue']], axis=1)
        mean_color = rgb.mean(axis=0).astype(float)
    else:
        mean_color = np.array([0., 0., 0.])  # fallback

    # radius: mean distance to centroid
    radius = np.linalg.norm(xyz - centroid[None, :], axis=1).mean().astype(float)

    return {
        "centroid": centroid.tolist(),
        "mean_color": mean_color.tolist(),
        "radius": float(radius)
    }


def compute_pairwise_relations(object_stats):
    """
    Build pairwise geometric relationships between objects.
    object_stats: dict {id: stats}
    returns: list of edges
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

            # compute 8-d edge attributes
            diff = c2 - c1
            dx, dy, dz = diff.tolist()
            dist = float(distance)
            n1 = float(np.linalg.norm(c1))
            n2 = float(np.linalg.norm(c2))
            # angle between vectors from origin
            if n1 > 1e-6 and n2 > 1e-6:
                cosang = float(np.dot(c1, c2) / (n1 * n2))
            else:
                cosang = 1.0
            sinang = float(np.sqrt(max(0.0, 1.0 - cosang*cosang)))

            edges.append({
                "subject": str(id1),
                "object": str(id2),
                "distance": dist,
                "direction": direction_norm,
                "edge_attr": [dx, dy, dz, dist, n1, n2, cosang, sinang]
            })
            # add reverse edge (undirected graph)
            edges.append({
                "subject": str(id2),
                "object": str(id1),
                "distance": dist,
                "direction": (-direction_norm[0], -direction_norm[1], -direction_norm[2]),
                "edge_attr": [-dx, -dy, -dz, dist, n2, n1, cosang, sinang]
            })

    return edges


def merge_text_relations(all_frames):
    """
    all_frames: list[frame_dict]
    Each frame contains:
        "spatial_relations": [{subject, object, relation}]
    Returns: list of unique relations
    """
    merged = []
    seen = set()

    for fr in all_frames:
        for r in fr.get("spatial_relations", []):
            key = (r["subject"], r["object"], r["relation"])
            if key not in seen:
                seen.add(key)
                merged.append(r)

    return merged


def merge_text_descriptions(all_frames):
    """
    Concatenate descriptions but remove duplicates and low-information parts.
    """
    descs = []
    seen = set()

    for fr in all_frames:
        d = fr.get("description", "").strip()
        if d and d not in seen:
            seen.add(d)
            descs.append(d)

    return " ".join(descs)


# ============================================================
# Main processing function
# ============================================================

def build_scene_graph(scene_dir):
    """
    scene_dir = /path/to/3RScan/<scene_id>/
    Must contain:
        - labels.instances.annotated.v2.ply
        - output/descriptions/all_descriptions.json
    """
    ply_path = os.path.join(scene_dir, "labels.instances.annotated.v2.ply")
    desc_path = os.path.join(scene_dir, "output/descriptions/all_descriptions.json")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    if not os.path.exists(desc_path):
        raise FileNotFoundError(f"Descriptions not found: {desc_path}")

    # Load text annotations
    with open(desc_path, "r") as f:
        all_descriptions = json.load(f)

    # Load PLY
    ply = load_ply(ply_path)
    obj_ids = ply["objectId"]

    # Group points by objectId
    object_points = defaultdict(list)
    for idx, oid in enumerate(obj_ids):
        object_points[int(oid)].append(idx)

    # Compute per-object statistics
    object_stats = {}
    for oid, idxs in object_points.items():
        v = ply[idxs]
        object_stats[str(oid)] = compute_object_stats(v)

    # Compute geometric relationships
    geometric_edges = compute_pairwise_relations(object_stats)

    # Merge text relations
    text_edges = merge_text_relations(all_descriptions)

    # Merge scene description
    merged_description = merge_text_descriptions(all_descriptions)

    # Build final scene graph
    scene_graph = {
        "scene_id": os.path.basename(scene_dir),
        "description": merged_description,
        "nodes": object_stats,              # per-object features
        "edges_text": text_edges,           # text relations
        "edges_geometric": geometric_edges  # geometric relations
    }

    return scene_graph


# ============================================================
# A simple CLI for testing
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_path",
        type=str,
        required=True,
        help="Path to a 3RScan scene folder, e.g. 3RScan/dbeb4d09-faf9-2324-9b85-dabd70dba4d0/"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional: path to save scene graph JSON"
    )

    args = parser.parse_args()

    sg = build_scene_graph(args.scene_path)

    print("✅ Scene Graph Built — Summary:")
    print(f"Scene ID: {sg['scene_id']}")
    print(f"#objects: {len(sg['nodes'])}")
    print(f"#text edges: {len(sg['edges_text'])}")
    print(f"#geom edges: {len(sg['edges_geometric'])}")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(sg, f, indent=2)
        print(f"Saved → {args.save}")


if __name__ == "__main__":
    main()