import json
import numpy as np
from plyfile import PlyData
import argparse

import clip
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# -----------------------------------------------------
# Threshold constants
# -----------------------------------------------------
DIR_THRESHOLD = 0.20
HEIGHT_THRESHOLD = 0.15
NEAR_THRESHOLD = 0.80


# -----------------------------------------------------
# Load PLY and group points by instanceId
# -----------------------------------------------------
def load_ply_instances(ply_path):
    print("\n[PLY] Reading:", ply_path)
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T
    rgb = np.vstack([v["red"], v["green"], v["blue"]]).T
    inst = v["objectId"]

    instance_ids = np.unique(inst)
    print(f"[PLY] Found {len(instance_ids)} unique object IDs")

    instance_points = {}
    for iid in instance_ids:
        mask = inst == iid
        pts = xyz[mask]
        colors = rgb[mask]

        print(f"  - Object {iid}: {pts.shape[0]} points")
        instance_points[str(iid)] = (pts, colors)

    return instance_points


# -----------------------------------------------------
# Load semseg labels
# -----------------------------------------------------
def load_semseg(semseg_path):
    print("\n[SEMSEG] Loading:", semseg_path)
    with open(semseg_path, "r") as f:
        data = json.load(f)

    id_to_label = {str(o["objectId"]): o["label"].lower()
                   for o in data["segGroups"]}

    print(f"[SEMSEG] Loaded {len(id_to_label)} labels.")
    return id_to_label


# -----------------------------------------------------
# Compute node attributes
# -----------------------------------------------------
def compute_node_attributes(points, colors):
    centroid = points.mean(axis=0)

    # Bounding-sphere radius
    d = np.linalg.norm(points - centroid, axis=1)
    radius = float(min(d.max(), 0.40))  # clamp

    mean_color = colors.mean(axis=0).tolist()

    return centroid.tolist(), mean_color, radius


# -----------------------------------------------------
# Relation helpers
# -----------------------------------------------------
def directional_rel(ci, cj):
    dx, dy, dz = cj - ci
    out = []

    if abs(dx) > DIR_THRESHOLD:
        out.append("right_of" if dx > 0 else "left_of")

    if abs(dz) > DIR_THRESHOLD:
        out.append("in_front_of" if dz > 0 else "behind")

    if abs(dy) > HEIGHT_THRESHOLD:
        out.append("above" if dy > 0 else "below")

    return out


def distance_rel(ci, cj, ri, rj):
    d = np.linalg.norm(cj - ci)
    if d <= (ri + rj) * 0.65:
        return ["touching"]
    if d < NEAR_THRESHOLD:
        return ["near"]
    return []


def symmetric_rel(sub, obj, rel):
    if rel == "left_of": return obj, sub, "right_of"
    if rel == "right_of": return obj, sub, "left_of"
    if rel == "in_front_of": return obj, sub, "behind"
    if rel == "behind": return obj, sub, "in_front_of"
    if rel == "above": return obj, sub, "below"
    if rel == "below": return obj, sub, "above"
    if rel == "near": return obj, sub, "near"
    return None


# -----------------------------------------------------
# Main Scene Graph Builder
# -----------------------------------------------------
def build_scene_graph(ply_path, semseg_path, output_path):

    inst_points = load_ply_instances(ply_path)
    id_to_label = load_semseg(semseg_path)

    # ------------------------------------------
    # Build nodes
    # ------------------------------------------
    nodes = {}
    print("\n[NODES] Computing centroids + radii + CLIP embeddings...")

    clip_cache = {}   # prevents recomputing same label many times

    for iid, (pts, colors) in inst_points.items():

        # -------- geometry features --------
        centroid, mean_color, radius = compute_node_attributes(pts, colors)
        label = id_to_label.get(iid, f"obj_{iid}")

        # -------- CLIP text embedding --------
        if label not in clip_cache:
            tokens = clip.tokenize(label).to(device)
            with torch.no_grad():
                clip_emb = clip_model.encode_text(tokens)[0]
                clip_emb = clip_emb / clip_emb.norm()  # normalize
            clip_cache[label] = clip_emb.cpu().tolist()

        nodes[iid] = {
            "label": label,
            "centroid": centroid,
            "mean_color": mean_color,
            "radius": radius,
            "clip_text_emb": clip_cache[label]   # <--- NEW FIELD
        }

        print(f"  Node {iid}: label={label}, centroid={centroid}, radius={radius:.3f}")

    # ------------------------------------------
    # Build RELATIONS using K-NN (K = 5)
    # ------------------------------------------
    print("\n[RELATIONS] Selecting K-nearest neighbors for each object...")

    K = 5
    obj_ids = list(nodes.keys())
    N = len(obj_ids)

    # Precompute centroids array (N, 3)
    centroids = np.array([nodes[obj]["centroid"] for obj in obj_ids])

    # Compute full pairwise distance matrix (N×N)
    dmat = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :], axis=2
    )

    # For each object, find indices of its K nearest neighbors (ignore itself)
    knn_idx = np.argsort(dmat, axis=1)[:, 1:K+1]   # skip index 0 (self)

    print(f"[RELATIONS] K = {K}, building relations only for nearest neighbors.")
    print("Debug nearest neighbors:")
    for i, oi in enumerate(obj_ids):
        neigh = [obj_ids[j] for j in knn_idx[i]]
        print(f"  - {oi} → {neigh}")

    edges = []
    seen = set()

    # ------------------------------------------
    # Build relations ONLY for K-nearest neighbors
    # ------------------------------------------
    for i, oi in enumerate(obj_ids):
        ci = np.array(nodes[oi]["centroid"])
        ri = nodes[oi]["radius"]

        # iterate over nearest neighbors
        for j_idx in knn_idx[i]:
            oj = obj_ids[j_idx]

            cj = np.array(nodes[oj]["centroid"])
            rj = nodes[oj]["radius"]

            # directional & distance relations
            dirs = directional_rel(ci, cj)
            dist = distance_rel(ci, cj, ri, rj)

            # touching → keep only vertical relations
            if "touching" in dist:
                dirs = [r for r in dirs if r in ["above", "below"]]

            rels = dirs + dist

            # forward edges
            for r in rels:
                key = (oi, oj, r)
                if key not in seen:
                    edges.append({"subject": oi, "object": oj, "relation": r})
                    seen.add(key)

                # symmetric
                sym = symmetric_rel(oi, oj, r)
                if sym:
                    sub, obj, rr = sym
                    skey = (sub, obj, rr)
                    if skey not in seen:
                        edges.append({"subject": sub, "object": obj, "relation": rr})
                        seen.add(skey)

    print(f"[RELATIONS] Final number of relations: {len(edges)}")

    # ------------------------------------------
    # Save output JSON
    # ------------------------------------------
    out = {
        "scene_id": ply_path.split("/")[-2],
        "nodes": nodes,
        "edges_text": edges
    }

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n[OK] Scene graph saved:", output_path)

# -----------------------------------------------------
# CLI
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Path to labels.instances.annotated.v2.ply")
    parser.add_argument("--semseg", required=True, help="Path to semseg.v2.json")
    parser.add_argument("--out", required=True, help="Output scene_graph JSON path")

    args = parser.parse_args()

    print("\n[MAIN] Starting scene graph generation...")
    print("PLY:", args.ply)
    print("SEMSEG:", args.semseg)
    print("OUT:", args.out)

    build_scene_graph(args.ply, args.semseg, args.out)