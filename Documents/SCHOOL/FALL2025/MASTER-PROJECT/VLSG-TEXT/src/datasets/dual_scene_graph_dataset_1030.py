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
    """Build 1030D with scene CLIP in every node."""
    centroid = np.array(node_dict["centroid"], dtype=np.float32)
    color = np.array(node_dict["mean_color"], dtype=np.float32) / 255.0
    node_clip = np.array(node_dict.get("clip_text_emb", np.zeros(512)), dtype=np.float32)
    scene_clip = np.array(scene_clip_emb, dtype=np.float32)  # ← Add back!
    
    return torch.cat([
        torch.from_numpy(centroid),    # 3
        torch.from_numpy(color),       # 3  
        torch.from_numpy(node_clip),   # 512
        torch.from_numpy(scene_clip)   # 512 ← Include scene CLIP!
    ])  # Total: 1030 dims

# def build_node_features_with_scene_clip(node_dict, scene_clip_emb):
#     """
#     Build 518-dim node features WITHOUT scene CLIP in nodes.
    
#     Scene CLIP will be added AFTER GNN, not duplicated in every node!
    
#     Args:
#         node_dict: Single node data
#         scene_clip_emb: Scene-level CLIP (NOT USED - kept for API compatibility)
    
#     Returns: centroid(3) + color(3) + node_CLIP(512) = 518 dims
#     """
#     centroid = np.array(node_dict["centroid"], dtype=np.float32)
#     color = np.array(node_dict["mean_color"], dtype=np.float32) / 255.0
    
#     # Node-level CLIP only (object-specific: "desk_center_middle")
#     node_clip = np.array(
#         node_dict.get("clip_text_emb", np.zeros(512)),
#         dtype=np.float32
#     )
    
#     # DON'T concatenate scene_clip_emb here!
#     # It will be added after GNN processing
#     return torch.cat([
#         torch.from_numpy(centroid),    # 3
#         torch.from_numpy(color),       # 3  
#         torch.from_numpy(node_clip)    # 512 (object only)
#     ])  # Total: 518 dims (no scene CLIP!)


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
        return self._load_scene_from_data(data)
    
    def _load_scene_from_data(self, data):
        """Load scene from data dict (allows subgraph augmentation)."""
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
    
    def _create_subgraph(self, data, ratio=None):
        """
        Create subgraph by keeping 40-70% of nodes.
        Simulates ScanScribe partial queries!
        """
        nodes = data['nodes']
        edges = data.get('edges_text', [])
        
        if ratio is None:
            ratio = random.uniform(0.4, 0.7)  # Like ScanScribe
        
        num_nodes = len(nodes)
        num_keep = max(3, int(num_nodes * ratio))
        
        # Random sampling
        all_node_ids = list(nodes.keys())
        keep_node_ids = set(random.sample(all_node_ids, num_keep))
        
        # Filter nodes
        subgraph_nodes = {nid: nodes[nid] for nid in keep_node_ids}
        
        # Filter edges
        subgraph_edges = []
        for edge in edges:
            if edge['subject'] in keep_node_ids and edge['object'] in keep_node_ids:
                subgraph_edges.append(edge)
        
        return {
            'scene_id': data['scene_id'] + '_subgraph',
            'nodes': subgraph_nodes,
            'edges_text': subgraph_edges,
            'scene_clip_emb': data.get('scene_clip_emb', [0.0] * 512),
            'scene_description': data.get('scene_description', '')
        }

    def __getitem__(self, idx):
        """
        Returns a proper positive pair (same room, different scans).
        
        NOW WITH SUBGRAPH AUGMENTATION:
        - 50% of time: use full scene
        - 50% of time: use random subgraph (40-70% of nodes)
        
        Returns scene_clip_emb separately (not in node features!)
        """
        src_path = self.scene_files[idx]
        src_scene_id = os.path.basename(src_path).replace('.json', '')
        
        # Find positive pair (same room, different scan)
        group_id = self.scene_to_group.get(src_scene_id)
        
        if group_id and group_id in self.group_to_scenes:
            same_room_scenes = self.group_to_scenes[group_id]
            candidates = [s for s in same_room_scenes if s != src_scene_id]
            
            if candidates:
                ref_scene_id = random.choice(candidates)
                ref_path = os.path.join(self.dataset_dir, ref_scene_id + '.json')
            else:
                ref_idx = random.randint(0, len(self.scene_files) - 1)
                while ref_idx == idx:
                    ref_idx = random.randint(0, len(self.scene_files) - 1)
                ref_path = self.scene_files[ref_idx]
                ref_scene_id = os.path.basename(ref_path).replace('.json', '')
        else:
            ref_idx = random.randint(0, len(self.scene_files) - 1)
            while ref_idx == idx:
                ref_idx = random.randint(0, len(self.scene_files) - 1)
            ref_path = self.scene_files[ref_idx]
            ref_scene_id = os.path.basename(ref_path).replace('.json', '')
        
        # Load scene data
        with open(src_path) as f:
            src_data = json.load(f)
        with open(ref_path) as f:
            ref_data = json.load(f)
        
        # Extract scene CLIP BEFORE subgraph augmentation
        src_scene_clip = torch.tensor(
            src_data.get('scene_clip_emb', [0.0] * 512), 
            dtype=torch.float32
        )
        ref_scene_clip = torch.tensor(
            ref_data.get('scene_clip_emb', [0.0] * 512),
            dtype=torch.float32
        )
        
        # ====== SUBGRAPH AUGMENTATION ======
        # 50% of time: create random subgraph
        if random.random() < 0.5:
            src_data = self._create_subgraph(src_data)
        if random.random() < 0.5:
            ref_data = self._create_subgraph(ref_data)
        # ====================================
        
        src = self._load_scene_from_data(src_data)
        ref = self._load_scene_from_data(ref_data)
        
        # Determine room labels
        src_group = self.scene_to_group.get(src_scene_id, src_scene_id)
        ref_group = self.scene_to_group.get(ref_scene_id, ref_scene_id)

        return {
            "node_feats_src": src[0],  # 518D (no scene CLIP!)
            "geom_edges_src": src[1],
            "geom_attr_src": src[2],
            "text_edges_src": src[3],
            "text_attr_src": src[4],

            "node_feats_ref": ref[0],  # 518D (no scene CLIP!)
            "geom_edges_ref": ref[1],
            "geom_attr_ref": ref[2],
            "text_edges_ref": ref[3],
            "text_attr_ref": ref[4],
            
            "scene_clip_src": src_scene_clip,  # 512D - returned separately!
            "scene_clip_ref": ref_scene_clip,  # 512D - returned separately!
            "room_id": src_group,
            "is_positive": (src_group == ref_group),
        }
    
    def _create_subgraph(self, scene_data):
        """
        Create random subgraph with 40-70% of nodes.
        Simulates ScanScribe partial queries during training!
        """
        nodes = scene_data['nodes']
        edges = scene_data.get('edges_text', [])
        
        # Random ratio between 0.4 and 0.7
        ratio = random.uniform(0.4, 0.7)
        num_nodes = len(nodes)
        num_keep = max(3, int(num_nodes * ratio))
        
        # Random node sampling
        all_node_ids = list(nodes.keys())
        keep_node_ids = set(random.sample(all_node_ids, num_keep))
        
        # Filter nodes
        subgraph_nodes = {nid: nodes[nid] for nid in keep_node_ids}
        
        # Filter edges (only keep edges between kept nodes)
        subgraph_edges = []
        for edge in edges:
            if edge['subject'] in keep_node_ids and edge['object'] in keep_node_ids:
                subgraph_edges.append(edge)
        
        # Create subgraph (preserve scene CLIP!)
        return {
            'scene_id': scene_data['scene_id'] + '_subgraph',
            'nodes': subgraph_nodes,
            'edges_text': subgraph_edges,
            'scene_clip_emb': scene_data.get('scene_clip_emb', [0.0] * 512),
            'scene_description': scene_data.get('scene_description', '')
        }
    
    def __len__(self):
        return len(self.scene_files)