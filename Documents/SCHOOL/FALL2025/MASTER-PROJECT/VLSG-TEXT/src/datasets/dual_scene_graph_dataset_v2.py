"""
Modified Dataset - Adds geometric features to make node embeddings more distinctive.

Key changes:
- Node features: 3 (centroid) + 3 (color) + 16 (geometric) + 512 (CLIP) = 534 dims
- Adds bbox_size, shape descriptors, extent, std_dev as additional features
- Makes 22 dimensions vary by geometry instead of just 3!

IMPORTANT: After using this, update your model's node_input_dim from 518 to 534!
"""

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset


def build_node_features_with_geometry(node_dict):
    """
    Build 534-dim features with CLIP + rich geometric information.
    
    Format: [centroid(3) + color(3) + geometric(16) + CLIP(512)] = 534 dims
    
    This makes 22 dimensions vary by object geometry (instead of just 3),
    giving the model much more signal to learn from!
    """
    # Basic features (same as before)
    centroid = np.array(node_dict["centroid"], dtype=np.float32)
    color = np.array(node_dict["mean_color"], dtype=np.float32) / 255.0
    
    # CLIP embedding
    clip_vec = np.array(
        node_dict.get("clip_text_emb", np.zeros(512)),
        dtype=np.float32
    )
    
    # ===== NEW: Extract geometric features (16 dimensions) =====
    geom_features = np.zeros(16, dtype=np.float32)
    
    if "geometric_features" in node_dict:
        geom = node_dict["geometric_features"]
        idx = 0
        
        # 1. Standard deviation (3 dims) - spread of points
        std_dev = np.array(geom.get("std_dev", [0, 0, 0]), dtype=np.float32)
        geom_features[idx:idx+3] = std_dev
        idx += 3
        
        # 2. Extent (3 dims) - principal component lengths
        extent = np.array(geom.get("extent", [0, 0, 0]), dtype=np.float32)
        geom_features[idx:idx+3] = extent
        idx += 3
        
        # 3. Shape descriptors (3 dims) - object shape
        geom_features[idx] = geom.get("linearity", 0.0)      # How line-like
        geom_features[idx+1] = geom.get("planarity", 0.0)    # How plane-like
        geom_features[idx+2] = geom.get("sphericity", 0.0)   # How sphere-like
        idx += 3
        
        # 4. Bounding box size (3 dims) - overall size
        bbox_size = np.array(geom.get("bbox_size", [0, 0, 0]), dtype=np.float32)
        geom_features[idx:idx+3] = bbox_size
        idx += 3
        
        # 5. Number of points (1 dim) - normalized
        num_points = float(geom.get("num_points", 1000)) / 10000.0
        geom_features[idx] = num_points
        idx += 1
        
        # 6. Radius (1 dim)
        radius = float(node_dict.get("radius", 0.5))
        geom_features[idx] = radius
        idx += 1
        
        # 7. Color std dev (2 dims) - color variation
        std_color = np.array(geom.get("std_color", [0, 0, 0]), dtype=np.float32)[:2] / 255.0
        geom_features[idx:idx+2] = std_color
        idx += 2
        
    else:
        # Fallback: just use radius if no geometric_features
        geom_features[0] = float(node_dict.get("radius", 0.5))
    
    # Concatenate: centroid + color + geometric + CLIP
    return torch.cat([
        torch.from_numpy(centroid),      # 3 dims
        torch.from_numpy(color),         # 3 dims
        torch.from_numpy(geom_features), # 16 dims (NEW!)
        torch.from_numpy(clip_vec)       # 512 dims
    ])  # Total: 534 dims


def extract_centroids_and_radii(nodes):
    obj_ids = list(nodes.keys())
    centroids = np.array([nodes[o]["centroid"] for o in obj_ids], dtype=float)
    radii = np.array([nodes[o]["radius"] for o in obj_ids], dtype=float)
    return obj_ids, centroids, radii


def build_geometric_edges_knn(nodes, K=5):
    obj_ids, centroids, radii = extract_centroids_and_radii(nodes)
    N = len(obj_ids)

    dmat = np.linalg.norm(centroids[:,None,:] - centroids[None,:,:], axis=2)
    np.fill_diagonal(dmat, np.inf)
    knn_idx = np.argsort(dmat, axis=1)[:, :K]

    edge_index = []
    edge_attr = []

    for i in range(N):
        ci, ri = centroids[i], radii[i]
        for j in knn_idx[i]:
            cj, rj = centroids[j], radii[j]
            vec = cj - ci
            dist = float(np.linalg.norm(vec))
            feat = np.array([vec[0], vec[1], vec[2], dist, ri, rj, 0.0, 0.0], dtype=np.float32)
            edge_index.append([i, j])
            edge_attr.append(feat)

    if not edge_index:
        return torch.zeros((2,0),dtype=torch.long), torch.zeros((0,8),dtype=torch.float32)

    return (
        torch.tensor(edge_index, dtype=torch.long).t(),
        torch.tensor(edge_attr, dtype=torch.float32)
    )


def build_text_edges(relations, rel2id, id_to_idx):
    if len(relations) > 1000:
        relations = relations[:500]
        
    edge_index = []
    rel_ids = []

    for r in relations:
        subj = str(r.get("subject", ""))
        obj = str(r.get("object", ""))
        
        s = id_to_idx.get(subj)
        o = id_to_idx.get(obj)
        
        if s is None or o is None:
            continue

        rel_name = r.get("relation", "").lower().strip()
        
        if rel_name not in rel2id:
            rel_name = "none"
        
        edge_index.append([s, o])
        rel_ids.append(rel2id[rel_name])

    if not edge_index:
        return (
            torch.zeros((2,0),dtype=torch.long),
            torch.zeros((0,1),dtype=torch.long)
        )

    return (
        torch.tensor(edge_index, dtype=torch.long).t(),
        torch.tensor(rel_ids, dtype=torch.long).unsqueeze(-1)
    )


def mask_node_features(feats, ratio=0.1):
    """Modified: Don't mask CLIP embeddings (last 512 dims)"""
    if ratio <= 0:
        return feats
    N, D = feats.shape
    
    # Only mask first 22 dimensions (centroid + color + geometric)
    # Don't mask CLIP (dimensions 22-534)
    mask = torch.zeros((N, D), dtype=torch.bool, device=feats.device)
    mask[:, :22] = torch.rand((N, 22), device=feats.device) < ratio
    
    feats = feats.clone()
    feats[mask] = 0.0
    return feats


def dropout_edges(edge_index, edge_attr, drop_ratio=0.1):
    if edge_index.size(1) == 0 or drop_ratio <= 0:
        return edge_index, edge_attr
    E = edge_index.size(1)
    keep = int(E * (1 - drop_ratio))
    idx = torch.randperm(E)[:keep]
    return edge_index[:, idx], edge_attr[idx]


class DualSceneGraphDataset(Dataset):
    """
    Dataset with enhanced geometric features.
    
    Node features: 534 dims instead of 518
    - 22 dims vary by geometry (centroid + color + geometric features)
    - 512 dims CLIP embedding
    """
    
    def __init__(
        self, 
        dataset_dir, 
        metadata_path,
        generate_text_edges=True,
        use_pure_geometric=False,  # Set to False to use CLIP!
        augment_ratio=0.0  # Disabled by default
    ):
        self.dataset_dir = dataset_dir
        self.use_pure_geometric = use_pure_geometric
        self.augment_ratio = augment_ratio

        all_scenes = sorted([
            f.replace(".json","") for f in os.listdir(dataset_dir)
            if f.endswith(".json")
        ])
        self.scene_files = [os.path.join(dataset_dir, f + ".json") for f in all_scenes]

        print(f"[DATASET] Found {len(self.scene_files)} scene graphs")
        print(f"[DATASET] Use CLIP features: {not use_pure_geometric}")
        print(f"[DATASET] Augmentation ratio: {augment_ratio}")
        print(f"[DATASET] Node feature dims: 534 (22 geometric + 512 CLIP)")

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        self.scene_to_group = {}
        self.group_to_scenes = {}

        for entry in meta:
            group_id = entry["reference"]
            if group_id not in self.group_to_scenes:
                self.group_to_scenes[group_id] = []

            if group_id in all_scenes:
                self.group_to_scenes[group_id].append(group_id)
                self.scene_to_group[group_id] = group_id

            for scan_entry in entry["scans"]:
                sid = scan_entry["reference"]
                if sid in all_scenes:
                    self.group_to_scenes[group_id].append(sid)
                    self.scene_to_group[sid] = group_id

        # Build relation vocabulary
        print("[DATASET] Building relation vocabulary...")
        self.rel2id = {"none": 0}
        next_id = 1

        for scene_path in self.scene_files:
            with open(scene_path, 'r') as f:
                data = json.load(f)
            
            text_relations = data.get("edges_text", [])
            
            for r in text_relations:
                rel_name = r.get("relation", "").lower().strip()
                if rel_name and rel_name not in self.rel2id:
                    self.rel2id[rel_name] = next_id
                    next_id += 1

        common_relations = ["above", "below", "left_of", "right_of", "in_front_of", "behind"]
        for rel in common_relations:
            if rel not in self.rel2id:
                self.rel2id[rel] = next_id
                next_id += 1

        print(f"[DATASET] Relation vocab size: {len(self.rel2id)}")

    def _sample_pos_or_neg(self, idx):
        src_path = self.scene_files[idx]
        sid = os.path.basename(src_path).replace(".json","")
        g = self.scene_to_group.get(sid)

        if g is None:
            pool = list(range(len(self.scene_files)))
            pool.remove(idx)
            return random.choice(pool)

        positives = []
        for i, path in enumerate(self.scene_files):
            sid2 = os.path.basename(path).replace(".json","")
            if sid2 != sid and self.scene_to_group.get(sid2) == g:
                positives.append(i)

        if positives:
            return random.choice(positives)

        pool = list(range(len(self.scene_files)))
        pool.remove(idx)
        return random.choice(pool)

    def _load_scene(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        nodes = data["nodes"]
        text_relations = data.get("edges_text", [])
        
        node_ids = list(nodes.keys())
        id2idx = {str(nid): i for i, nid in enumerate(node_ids)}

        # Build node features with geometric info
        feats = []
        for nid in node_ids:
            n = nodes[nid]
            feat = build_node_features_with_geometry(n)
            feats.append(feat)

        node_feats = torch.stack(feats, dim=0)

        # Build edges
        geom_edges, geom_attr = build_geometric_edges_knn(nodes)
        text_edges, text_attr = build_text_edges(text_relations, self.rel2id, id2idx)

        # Augmentation (only masks first 22 dims, not CLIP)
        node_feats = mask_node_features(node_feats, ratio=self.augment_ratio)
        geom_edges, geom_attr = dropout_edges(geom_edges, geom_attr, drop_ratio=self.augment_ratio)
        text_edges, text_attr = dropout_edges(text_edges, text_attr, drop_ratio=self.augment_ratio)

        return node_feats, geom_edges, geom_attr, text_edges, text_attr

    def __getitem__(self, idx):
        src_path = self.scene_files[idx]
        ref_path = self.scene_files[self._sample_pos_or_neg(idx)]

        src = self._load_scene(src_path)
        ref = self._load_scene(ref_path)

        return {
            "node_feats_src": src[0],
            "geom_edges_src": src[1],
            "geom_attr_src": src[2],
            "text_edges_src": src[3],
            "text_attr_src": src[4],

            "node_feats_ref": ref[0],
            "geom_edges_ref": ref[1],
            "geom_attr_ref": ref[2],
            "text_edges_ref": ref[3],
            "text_attr_ref": ref[4],
        }

    def __len__(self):
        return len(self.scene_files)


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         DATASET WITH GEOMETRIC FEATURES (534 dims)             ║
╚═══════════════════════════════════════════════════════════════╝

Changes from original (518 dims):
  ✓ Added 16 geometric feature dimensions
  ✓ Now 22 dims vary by object (instead of 3!)
  ✓ CLIP embeddings still included (512 dims)
  ✓ Total: 534 dimensions

Geometric features added:
  - std_dev (3): Point cloud spread
  - extent (3): Principal component lengths
  - shape descriptors (3): linearity, planarity, sphericity
  - bbox_size (3): Bounding box dimensions
  - num_points (1): Number of points in cloud
  - radius (1): Bounding sphere radius
  - std_color (2): Color variation

IMPORTANT: Update your model config!
  model = DualSceneAligner(
      node_input_dim=534,  # Changed from 518!
      ...
  )
    """)