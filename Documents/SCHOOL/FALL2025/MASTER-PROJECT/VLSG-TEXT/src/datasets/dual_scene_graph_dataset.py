import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset

from utils.clip_utils import get_clip_text_embedding

print(get_clip_text_embedding("left of").shape)

# ============================================================
# Utility Functions
# ============================================================

def extract_centroids_and_radii(nodes):
    """Return centroid array (N,3) and radius array (N,) in consistent order."""
    obj_ids = list(nodes.keys())
    centroids = np.array([nodes[o]["centroid"] for o in obj_ids], dtype=float)
    radii     = np.array([nodes[o]["radius"]   for o in obj_ids], dtype=float)
    return obj_ids, centroids, radii


# ============================================================
# 1. GEOMETRIC EDGES (KNN)
# ============================================================

def build_geometric_edges_knn(nodes, K=5):
    """
    Build geometric KNN edges with 8-dim edge features.
    """
    obj_ids, centroids, radii = extract_centroids_and_radii(nodes)
    N = len(obj_ids)

    # Pairwise dist matrix
    dmat = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :],
        axis=2
    )
    np.fill_diagonal(dmat, np.inf)

    # Top-K neighbors
    knn_idx = np.argsort(dmat, axis=1)[:, :K]

    # Build edge tensors
    edge_index = []
    edge_attr = []

    for i in range(N):
        ci, ri = centroids[i], radii[i]

        for j in knn_idx[i]:
            cj, rj = centroids[j], radii[j]
            vec = cj - ci
            dist = float(np.linalg.norm(vec))

            feat = np.array([
                vec[0], vec[1], vec[2],
                dist,
                ri, rj,
                0.0, 0.0
            ], dtype=np.float32)

            edge_index.append([i, j])
            edge_attr.append(feat)

    if not edge_index:
        return (
            torch.zeros((2,0), dtype=torch.long),
            torch.zeros((0,8), dtype=torch.float32)
        )

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float32)
    return edge_index, edge_attr


# ============================================================
# 2. TEXT EDGES
# ============================================================
def build_text_edges(relations, rel2id, id_to_idx):
    edge_index = []
    rel_ids = []  # store only IDs here

    for r in relations:
        s = id_to_idx.get(r["subject"])
        o = id_to_idx.get(r["object"])
        if s is None or o is None:
            continue

        edge_index.append([s, o])
        rel_ids.append(rel2id[r["relation"].lower()])

    if not edge_index:
        return (
            torch.zeros((2,0), dtype=torch.long),
            torch.zeros((0,1), dtype=torch.long)
        )

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    rel_ids = torch.tensor(rel_ids, dtype=torch.long).unsqueeze(-1)  # (E,1)

    return edge_index, rel_ids
# ============================================================
# 3. RELATION SPARSIFICATION
# ============================================================

def sparsify_relations(nodes, relations, K=5):
    """Keep only K closest relation targets per subject node."""
    centroids = {nid: np.array(n["centroid"], dtype=float) for nid, n in nodes.items()}

    rel_by_subject = {}
    for r in relations:
        s, o = r["subject"], r["object"]
        if s not in centroids or o not in centroids:
            continue
        d = np.linalg.norm(centroids[o] - centroids[s])
        rel_by_subject.setdefault(s, []).append((d, r))

    output = []
    for s, lst in rel_by_subject.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])[:K]
        output.extend([r for (_, r) in lst_sorted])

    return output


# ============================================================
# 4. SceneGraph Dataset
# ============================================================

class DualSceneGraphDataset(Dataset):
    def __init__(self, dataset_dir):

        self.dataset_dir = dataset_dir
        self.scene_files = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".json")
        ])

        print(f"\n[DATASET] Found {len(self.scene_files)} scenes")

        # Build relation vocabulary
        self.rel2id = {"none": 0}
        next_id = 1

        for path in self.scene_files:
            data = json.load(open(path))
            for r in data.get("edges_text", []):
                rel = r["relation"].lower()
                if rel not in self.rel2id:
                    self.rel2id[rel] = next_id
                    next_id += 1
        # Build matrix of shape (num_relations, 512)
        rel_clip_matrix = []
        for rel in self.rel2id.keys():
            emb = get_clip_text_embedding(rel)   # 512-d
            rel_clip_matrix.append(emb)
        self.rel_clip_matrix = torch.tensor(rel_clip_matrix, dtype=torch.float32)

        print("[DATASET] Relation vocab:", self.rel2id)
        print(f"[DATASET] Vocab size = {len(self.rel2id)}")


    # --------------------------------------------------------
    # Helper: sample a random different scene as negative
    # --------------------------------------------------------
    def _sample_negative(self, idx):
        other = list(range(len(self.scene_files)))
        other.remove(idx)
        return random.choice(other)


    # --------------------------------------------------------
    # Helper: load + encode one scene
    # --------------------------------------------------------
    def _load_scene(self, json_path):
        data = json.load(open(json_path))
        nodes = data["nodes"]

        # Relations → sparsify
        rels = sparsify_relations(nodes, data.get("edges_text", []), K=5)

        # Node ordering
        node_ids = list(nodes.keys())
        id2idx = {nid: i for i, nid in enumerate(node_ids)}

        # Node features
        feats = []
        for nid in node_ids:
            n = nodes[nid]
            clip_vec = np.array(n["clip_text_emb"], dtype=np.float32)

            feats.append(np.concatenate([
                np.array(n["centroid"], dtype=np.float32),   # 3
                np.array(n["mean_color"], dtype=np.float32), # 3
                clip_vec                                      # 512
]))

        node_feats = torch.tensor(np.vstack(feats), dtype=torch.float32)

        geom_edges, geom_attr = build_geometric_edges_knn(nodes, K=5)
        text_edges, text_attr = build_text_edges(rels, self.rel2id, id2idx)

        return node_feats, geom_edges, geom_attr, text_edges, text_attr


    # --------------------------------------------------------
    # Main data loader: returns positive–negative pair
    # --------------------------------------------------------
    def __getitem__(self, idx):

        src_path = self.scene_files[idx]
        ref_path = self.scene_files[self._sample_negative(idx)]

        # Load both scenes
        node_feats_src, geom_edges_src, geom_attr_src, text_edges_src, text_attr_src = \
            self._load_scene(src_path)

        node_feats_ref, geom_edges_ref, geom_attr_ref, text_edges_ref, text_attr_ref = \
            self._load_scene(ref_path)

        return {
            "node_feats_src": node_feats_src,
            "geom_edges_src": geom_edges_src,
            "geom_attr_src": geom_attr_src,
            "text_edges_src": text_edges_src,
            "text_attr_src": text_attr_src,

            "node_feats_ref": node_feats_ref,
            "geom_edges_ref": geom_edges_ref,
            "geom_attr_ref": geom_attr_ref,
            "text_edges_ref": text_edges_ref,
            "text_attr_ref": text_attr_ref,
        }


    def __len__(self):
        return len(self.scene_files)
    


        # Correct structure for return
        # return {
        #     "node_feats_src": node_feats_src,
        #     "geom_edges_src": geom_edges_src,
        #     "geom_attr_src": geom_attr_src,
        #     "text_edges_src": text_edges_src,
        #     "text_attr_src": text_attr_src,

        #     "node_feats_ref": node_feats_ref,
        #     "geom_edges_ref": geom_edges_ref,
        #     "geom_attr_ref": geom_attr_ref,
        #     "text_edges_ref": text_edges_ref,
        #     "text_attr_ref": text_attr_ref,
        # }