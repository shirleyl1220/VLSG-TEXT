"""
Fixed Dataset - Loads REAL text edges from your scene graphs.
Only generates synthetic edges as FALLBACK if none exist.
"""

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset


# ============================================================
# Fallback: Generate Text Edges ONLY if missing
# ============================================================

def generate_text_edges_fallback(nodes, k_neighbors=5):
    """
    FALLBACK: Only called if scene has no text edges.
    Generates basic spatial relations based on geometry.
    """
    node_ids = list(nodes.keys())
    if len(node_ids) < 2:
        return []
    
    centroids = {nid: np.array(nodes[nid]["centroid"]) for nid in node_ids}
    text_edges = []
    
    for nid1 in node_ids:
        c1 = centroids[nid1]
        
        distances = []
        for nid2 in node_ids:
            if nid1 == nid2:
                continue
            c2 = centroids[nid2]
            dist = np.linalg.norm(c2 - c1)
            distances.append((dist, nid2, c2))
        
        distances.sort(key=lambda x: x[0])
        
        for dist, nid2, c2 in distances[:k_neighbors]:
            diff = c2 - c1
            dx, dy, dz = diff
            
            abs_vals = [abs(dx), abs(dy), abs(dz)]
            max_idx = abs_vals.index(max(abs_vals))
            
            if max_idx == 2:
                relation = "above" if dz > 0 else "below"
            elif max_idx == 0:
                relation = "right_of" if dx > 0 else "left_of"
            else:
                relation = "in_front_of" if dy > 0 else "behind"
            
            text_edges.append({
                "subject": str(nid1),
                "object": str(nid2),
                "relation": relation
            })
    
    return text_edges


# ============================================================
# Build Node Features WITHOUT CLIP
# ============================================================

def build_geometric_node_features(node_dict):
    """
    Build 518-dim features using ONLY geometry + random noise.
    NO CLIP EMBEDDINGS - they cause collapse!
    """
    centroid = np.array(node_dict["centroid"], dtype=np.float32)
    color = np.array(node_dict["mean_color"], dtype=np.float32) / 255.0
    radius = float(node_dict.get("radius", 0.5))
    
    features = np.zeros(512, dtype=np.float32)
    
    if "geometric_features" in node_dict:
        geom = node_dict["geometric_features"]
        
        offset = 0
        
        # Std dev (3)
        std_dev = np.array(geom.get("std_dev", [0, 0, 0]), dtype=np.float32)
        features[offset:offset+3] = std_dev
        offset += 3
        
        # Std color (3)
        std_color = np.array(geom.get("std_color", [0, 0, 0]), dtype=np.float32) / 255.0
        features[offset:offset+3] = std_color
        offset += 3
        
        # Extent (3)
        extent = np.array(geom.get("extent", [0, 0, 0]), dtype=np.float32)
        features[offset:offset+3] = extent
        offset += 3
        
        # Shape descriptors (3)
        features[offset] = geom.get("linearity", 0.0)
        features[offset+1] = geom.get("planarity", 0.0)
        features[offset+2] = geom.get("sphericity", 0.0)
        offset += 3
        
        # Bbox size (3)
        bbox = np.array(geom.get("bbox_size", [0, 0, 0]), dtype=np.float32)
        features[offset:offset+3] = bbox
        offset += 3
        
        # Num points (1)
        num_points = float(geom.get("num_points", 1000)) / 10000.0
        features[offset] = num_points
        offset += 1
        
        features[offset] = radius
        offset += 1
        
        # CRITICAL: Add diverse random features
        remaining = 512 - offset
        features[offset:] = np.random.randn(remaining).astype(np.float32) * 0.2
    else:
        # All random with some structure
        features[:6] = np.concatenate([centroid, color])
        features[6] = radius
        features[7:] = np.random.randn(505).astype(np.float32) * 0.2
    
    return torch.cat([
        torch.from_numpy(centroid),
        torch.from_numpy(color),
        torch.from_numpy(features)
    ])


# ============================================================
# Utility Functions
# ============================================================

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
    """
    Build text edges from relations.
    This properly handles your scene graph format!
    """
    edge_index = []
    rel_ids = []

    for r in relations:
        # Get subject and object - handle both string and int
        subj = str(r.get("subject", ""))
        obj = str(r.get("object", ""))
        
        # Map to indices
        s = id_to_idx.get(subj)
        o = id_to_idx.get(obj)
        
        if s is None or o is None:
            continue

        # Get relation name
        rel_name = r.get("relation", "").lower().strip()
        
        if rel_name not in rel2id:
            # If relation not in vocab, skip or use "none"
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
    if ratio <= 0:
        return feats
    N, D = feats.shape
    mask = torch.rand((N, D), device=feats.device) < ratio
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


# ============================================================
# FIXED Dataset - Loads YOUR Text Edges
# ============================================================

class DualSceneGraphDataset(Dataset):
    def __init__(
        self, 
        dataset_dir, 
        metadata_path,
        generate_text_edges=True,
        use_pure_geometric=True,
        augment_ratio=0.1,
        fallback_generate_text_edges=True  # Only generate if missing
    ):
        """
        Args:
            dataset_dir: Directory with scene graph JSONs
            metadata_path: Path to sequence.json
            use_pure_geometric: Use geometric features instead of CLIP
            augment_ratio: Data augmentation ratio
            fallback_generate_text_edges: Generate synthetic edges only if missing
        """
        self.dataset_dir = dataset_dir
        self.use_pure_geometric = use_pure_geometric
        self.augment_ratio = augment_ratio
        self.fallback_generate = fallback_generate_text_edges

        # Load scene files
        all_scenes = sorted([
            f.replace(".json","") for f in os.listdir(dataset_dir)
            if f.endswith(".json")
        ])
        self.scene_files = [os.path.join(dataset_dir, f + ".json") for f in all_scenes]

        print(f"[DATASET] Found {len(self.scene_files)} scene graphs")
        print(f"[DATASET] Use pure geometric features: {use_pure_geometric}")
        print(f"[DATASET] Fallback text edge generation: {fallback_generate_text_edges}")

        # Load metadata
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

        # Build relation vocabulary from ACTUAL data
        print("[DATASET] Building relation vocabulary from data...")
        self.rel2id = {"none": 0}
        next_id = 1

        for scene_path in self.scene_files:
            with open(scene_path, 'r') as f:
                data = json.load(f)
            
            # Look for text relations in the data
            text_relations = data.get("edges_text", [])
            
            for r in text_relations:
                rel_name = r.get("relation", "").lower().strip()
                if rel_name and rel_name not in self.rel2id:
                    self.rel2id[rel_name] = next_id
                    next_id += 1

        # Add common spatial relations if not present
        common_relations = ["above", "below", "left_of", "right_of", "in_front_of", "behind"]
        for rel in common_relations:
            if rel not in self.rel2id:
                self.rel2id[rel] = next_id
                next_id += 1

        print(f"[DATASET] Relation vocab size: {len(self.rel2id)}")
        print(f"[DATASET] Relations: {list(self.rel2id.keys())}")

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
        """
        Load scene and use REAL text edges from your data.
        Only generate synthetic edges as fallback if missing.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        nodes = data["nodes"]
        
        # TRY TO LOAD REAL TEXT EDGES FIRST
        text_relations = data.get("edges_text", [])
        
        # FALLBACK: Generate only if missing
        if not text_relations and self.fallback_generate:
            print(f"[WARNING] No text edges in {os.path.basename(json_path)}, generating fallback")
            text_relations = generate_text_edges_fallback(nodes, k_neighbors=5)
        
        if not text_relations:
            print(f"[INFO] {os.path.basename(json_path)}: {len(text_relations)} text relations")
        
        # Build node ID mapping
        node_ids = list(nodes.keys())
        id2idx = {str(nid): i for i, nid in enumerate(node_ids)}

        # Build node features
        feats = []
        for nid in node_ids:
            n = nodes[nid]
            if self.use_pure_geometric:
                feat = build_geometric_node_features(n)
            else:
                # Fallback to CLIP if available (not recommended)
                clip_vec = np.array(
                    n.get("clip_text_emb", np.random.randn(512) * 0.1),
                    dtype=np.float32
                )
                feat = torch.cat([
                    torch.from_numpy(np.array(n["centroid"], dtype=np.float32)),
                    torch.from_numpy(np.array(n["mean_color"], dtype=np.float32)),
                    torch.from_numpy(clip_vec)
                ])
            feats.append(feat)

        node_feats = torch.stack(feats, dim=0)

        # Build edges
        geom_edges, geom_attr = build_geometric_edges_knn(nodes)
        text_edges, text_attr = build_text_edges(text_relations, self.rel2id, id2idx)

        # Augmentation
        node_feats = mask_node_features(node_feats, ratio=self.augment_ratio)
        geom_edges, geom_attr = dropout_edges(geom_edges, geom_attr, drop_ratio=self.augment_ratio)
        text_edges, text_attr = dropout_edges(text_edges, text_attr, drop_ratio=self.augment_ratio)

        return node_feats, geom_edges, geom_attr, text_edges, text_attr

    def __getitem__(self, idx):
        """
        Returns the same format as before.
        """
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


# ============================================================
# Usage
# ============================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║             DATASET - LOADS YOUR REAL TEXT EDGES               ║
╚═══════════════════════════════════════════════════════════════╝

This dataset will:

✅ Load your REAL text edges from scene graphs (edges_text)
✅ Use geometric features instead of broken CLIP
✅ Generate synthetic edges ONLY as fallback if missing
✅ Build relation vocabulary from your actual data

Your scene graph format:
{
  "nodes": {...},
  "edges_text": [
    {"subject": "0", "object": "14", "relation": "right_of"},
    {"subject": "14", "object": "0", "relation": "left_of"},
    ...
  ],
  "edges_geometric": [...]
}

USAGE:
------
dataset = DualSceneGraphDataset(
    dataset_dir="scene_graphs/",
    metadata_path="3RScan.json",
    use_pure_geometric=True,
    augment_ratio=0.1,
    fallback_generate_text_edges=True  # Only if missing
)

The dataset will:
1. Try to load edges_text from your JSON
2. Build relation vocab from your data
3. Only generate synthetic if edges_text is empty
    """)