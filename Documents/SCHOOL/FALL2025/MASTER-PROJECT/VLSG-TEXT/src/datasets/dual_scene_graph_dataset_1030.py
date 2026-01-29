"""
Dataset with Scene-Level CLIP - Uses 1030-dim features

Features: centroid(3) + color(3) + node_CLIP(512) + scene_CLIP(512) = 1030 dims

This combines:
- Object-level semantics (node CLIP): "desk", "chair", etc.
- Room-level semantics (scene CLIP): "office", "bedroom", etc.
"""

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset


def build_node_features_with_scene_clip(node_dict, scene_clip_emb):
    """
    Build 1030-dim features with BOTH node-level and scene-level CLIP.
    
    Args:
        node_dict: Single node data
        scene_clip_emb: Scene-level CLIP (512D) - shared by all nodes in scene
    """
    centroid = np.array(node_dict["centroid"], dtype=np.float32)
    color = np.array(node_dict["mean_color"], dtype=np.float32) / 255.0
    
    # Node-level CLIP (object-specific: "desk_center_middle")
    node_clip = np.array(
        node_dict.get("clip_text_emb", np.zeros(512)),
        dtype=np.float32
    )
    
    # Scene-level CLIP (room-type: "office with desk, chair, monitor...")
    scene_clip = np.array(scene_clip_emb, dtype=np.float32)
    
    return torch.cat([
        torch.from_numpy(centroid),    # 3
        torch.from_numpy(color),       # 3
        torch.from_numpy(node_clip),   # 512 (object)
        torch.from_numpy(scene_clip)   # 512 (room)
    ])  # Total: 1030 dims


def extract_centroids_and_radii(nodes):
    obj_ids = list(nodes.keys())
    centroids = np.array([nodes[o]["centroid"] for o in obj_ids], dtype=float)
    radii = np.array([nodes[o]["radius"] for o in obj_ids], dtype=float)
    return obj_ids, centroids, radii


def build_geometric_edges_knn(nodes, K=5):
    obj_ids, centroids, radii = extract_centroids_and_radii(nodes)
    N = len(obj_ids)

    if N <= 1:
        return torch.zeros((2,0),dtype=torch.long), torch.zeros((0,8),dtype=torch.float32)

    dmat = np.linalg.norm(centroids[:,None,:] - centroids[None,:,:], axis=2)
    np.fill_diagonal(dmat, np.inf)
    knn_idx = np.argsort(dmat, axis=1)[:, :min(K, N-1)]

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


class DualSceneGraphDataset(Dataset):
    """
    Dataset that loads scene graphs with scene-level CLIP embeddings.
    
    Node features: 1030 dims (centroid + color + node_CLIP + scene_CLIP)
    """
    
    def __init__(
        self, 
        dataset_dir, 
        metadata_path,
        augment_ratio=0.0
    ):
        self.dataset_dir = dataset_dir
        self.augment_ratio = augment_ratio

        all_scenes = sorted([
            f.replace(".json","") for f in os.listdir(dataset_dir)
            if f.endswith(".json")
            and not f.startswith("metadata")
            and not f.startswith("training_splits")
        ])
        self.scene_files = [os.path.join(dataset_dir, f + ".json") for f in all_scenes]

        print(f"[DATASET] Found {len(self.scene_files)} scene graphs")
        print(f"[DATASET] Node features: 1030 dims (centroid + color + node_CLIP + scene_CLIP)")
        print(f"[DATASET] Augmentation: {augment_ratio}")

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

            for scan_entry in entry.get("scans", []):
                sid = scan_entry["reference"]
                if sid in all_scenes:
                    self.group_to_scenes[group_id].append(sid)
                    self.scene_to_group[sid] = group_id

        # Build relation vocabulary
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

        common_relations = ["above", "below", "left_of", "right_of", "in_front_of", "behind", "near", "touching"]
        for rel in common_relations:
            if rel not in self.rel2id:
                self.rel2id[rel] = next_id
                next_id += 1

        print(f"[DATASET] Relation vocab size: {len(self.rel2id)}")

    def _load_scene(self, json_path):
        """Load scene with scene-level CLIP"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        nodes = data["nodes"]
        text_relations = data.get("edges_text", [])
        
        # Get scene-level CLIP (512D)
        scene_clip_emb = data.get("scene_clip_emb", [0.0] * 512)
        
        node_ids = list(nodes.keys())
        id2idx = {str(nid): i for i, nid in enumerate(node_ids)}

        # Build node features with scene CLIP
        feats = []
        for nid in node_ids:
            n = nodes[nid]
            # Pass scene_clip_emb to every node!
            feat = build_node_features_with_scene_clip(n, scene_clip_emb)  # 1030 dims
            feats.append(feat)

        node_feats = torch.stack(feats, dim=0)

        # Build edges
        geom_edges, geom_attr = build_geometric_edges_knn(nodes)
        text_edges, text_attr = build_text_edges(text_relations, self.rel2id, id2idx)

        return node_feats, geom_edges, geom_attr, text_edges, text_attr

    def __getitem__(self, idx):
        src_path = self.scene_files[idx]
        ref_idx = (idx + 1) % len(self.scene_files)
        ref_path = self.scene_files[ref_idx]

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