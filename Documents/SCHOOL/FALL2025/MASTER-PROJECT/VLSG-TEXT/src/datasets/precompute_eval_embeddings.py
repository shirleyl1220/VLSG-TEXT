
import torch
import torch.nn.functional as F
import numpy as np
import clip
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append('../data_processing')
sys.path.append('../../../')
from src.eval.scene_graph import SceneGraph

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from src.models.sgaligner.src.aligner.dual_scene_aligner_v2 import DualSceneAligner
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CLIP
clip_model, _ = clip.load("ViT-B/32", device=device)

def get_clip_embedding(text, clip_model, device):
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu()

def get_base_label(label):
    parts = label.split('_')
    spatial = {'north','south','east','west','center','upper','middle','lower'}
    base = []
    for part in parts:
        if part in spatial:
            break
        base.append(part)
    return '_'.join(base) if base else label

# ── Load data ──────────────────────────────────────────────
print("Loading graphs...")
_3dssg_scenes = torch.load('/content/drive/MyDrive/VLSG_Files/3dssg_graphs_518D.pt',
                            weights_only=False, map_location='cpu')
_3dssg_graphs = {}
for sid in tqdm(_3dssg_scenes, desc="3DSSG"):
    _3dssg_graphs[sid] = SceneGraph(sid, graph_type='3dssg', graph=_3dssg_scenes[sid],
                                    max_dist=1.0, embedding_type='word2vec', use_attributes=True)

scanscribe_test_55 = torch.load('/content/drive/MyDrive/VLSG_Files/scanscribe_graphs_test_518D.pt',
                                 weights_only=False, map_location='cpu')
query_graphs = {}
for sid in tqdm(scanscribe_test_55, desc="ScanScribe 55"):
    for tid in scanscribe_test_55[sid].keys():
        key = f"{sid}_{str(tid).zfill(5)}"
        query_graphs[key] = SceneGraph(sid, txt_id=tid, graph_type='scanscribe',
                                       graph=scanscribe_test_55[sid][tid],
                                       embedding_type='word2vec', use_attributes=True)
query_graphs = {k: v for k, v in query_graphs.items() if len(v.edge_idx[0]) >= 1}

# ── Step 1: Node + relation CLIP caches ───────────────────
print("\nCollecting unique labels and relations...")
all_labels = set()
all_relations = set()

for g in list(query_graphs.values()) + list(_3dssg_graphs.values()):
    for nid in g.nodes:
        all_labels.add(g.nodes[nid].label)
    if hasattr(g, 'edge_relations') and g.edge_relations:
        for r in g.edge_relations:
            all_relations.add(str(r).lower())

print(f"  Unique labels: {len(all_labels)}")
print(f"  Unique relations: {len(all_relations)}")

node_clip_cache = {}
for label in tqdm(all_labels, desc="Node CLIP"):
    node_clip_cache[label] = get_clip_embedding(label, clip_model, device)

rel_clip_cache = {}
for rel in tqdm(all_relations, desc="Relation CLIP"):
    rel_clip_cache[rel] = get_clip_embedding(rel, clip_model, device)

scene_clip_cache = {}
for key, g in tqdm(list(query_graphs.items()) + list({sid: g for sid, g in _3dssg_graphs.items()}.items()), 
                    desc="Scene CLIP"):
    labels = [g.nodes[nid].label for nid in g.nodes]
    unique_labels = list(set(labels))[:10]
    scene_desc = f"A room with {', '.join(unique_labels)}"
    scene_clip_cache[key] = get_clip_embedding(scene_desc, clip_model, device)

torch.save({
    'node_clip': node_clip_cache,
    'rel_clip': rel_clip_cache,
    'scene_clip': scene_clip_cache
}, '/content/drive/MyDrive/VLSG_Files/clip_embedding_cache.pt')
print("✓ Saved clip_embedding_cache.pt")

# ── Step 2: Model embeddings ───────────────────────────────
print("\nLoading model...")

class SimpleGraphMatcher(nn.Module):
    def __init__(self, base_model, scene_clip_dim=512, hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        self.fusion = nn.Sequential(
            nn.LayerNorm(base_model.hidden_dim + scene_clip_dim),
            nn.Linear(base_model.hidden_dim + scene_clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    def forward(self, batch, scene_clip_src, scene_clip_ref):
        out = self.base_model(batch)
        src_emb = self.fusion(torch.cat([out['src_emb'], scene_clip_src], dim=-1))
        ref_emb = self.fusion(torch.cat([out['ref_emb'], scene_clip_ref], dim=-1))
        return {'src_emb': src_emb, 'ref_emb': ref_emb}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
args = parser.parse_args()

base_model = DualSceneAligner(node_input_dim=518, relation_dim=512,
                               hidden_dim=256, dropout=0.0).to(device)
model = SimpleGraphMatcher(base_model, scene_clip_dim=512, hidden_dim=256).to(device)
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
print("✓ Model loaded")

def build_batch_from_cache(graph, node_clip_cache, rel_clip_cache, 
                            scene_clip_cache, graph_key, device):
    node_feats = []
    for nid in graph.nodes:
        label = graph.nodes[nid].label
        node_clip = node_clip_cache.get(label, torch.zeros(512))
        feat = torch.cat([torch.zeros(6), node_clip])  # centroid+color zeros
        node_feats.append(feat)
    node_feats = torch.stack(node_feats)

    edge_idx = graph.edge_idx
    num_nodes = len(graph.nodes)
    if len(edge_idx) > 0 and len(edge_idx[0]) > 0:
        edges = torch.tensor(edge_idx, dtype=torch.long)
        valid_mask = (edges[0] < num_nodes) & (edges[1] < num_nodes)
        edges = edges[:, valid_mask]
        num_edges = edges.size(1)
        geom_attr = torch.zeros(num_edges, 8)
        if hasattr(graph, 'edge_relations') and graph.edge_relations:
            valid_indices = valid_mask.nonzero(as_tuple=True)[0].tolist()
            rel_embs = [rel_clip_cache.get(str(graph.edge_relations[i]).lower(),
                        torch.zeros(512)) for i in valid_indices]
            text_attr = torch.stack(rel_embs)
        else:
            text_attr = torch.zeros(num_edges, 512)
    else:
        edges = torch.zeros(2, 0, dtype=torch.long)
        geom_attr = torch.zeros(0, 8)
        text_attr = torch.zeros(0, 512)

    scene_clip = scene_clip_cache.get(graph_key, torch.zeros(512))

    return {
        "node_feats_src": node_feats.to(device),
        "geom_edges_src": edges.to(device),
        "geom_attr_src": geom_attr.to(device),
        "text_edges_src": edges.clone().to(device),
        "text_attr_src": text_attr.to(device),
        "node_feats_ref": node_feats.to(device),
        "geom_edges_ref": edges.to(device),
        "geom_attr_ref": geom_attr.to(device),
        "text_edges_ref": edges.clone().to(device),
        "text_attr_ref": text_attr.to(device),
        "src_batch": torch.zeros(node_feats.size(0), dtype=torch.long).to(device),
        "ref_batch": torch.zeros(node_feats.size(0), dtype=torch.long).to(device),
        "batch_size": 1,
        "scene_clip_src": scene_clip.unsqueeze(0).to(device),
        "scene_clip_ref": scene_clip.unsqueeze(0).to(device),
    }

# ── Precompute DB embeddings ───────────────────────────────
print("\nPrecomputing 3DSSG database embeddings...")
db_emb_cache = {}
with torch.no_grad():
    for scene_id, g in tqdm(_3dssg_graphs.items(), desc="DB"):
        batch = build_batch_from_cache(g, node_clip_cache, rel_clip_cache,
                                        scene_clip_cache, scene_id, device)
        out = model(batch, scene_clip_src=batch['scene_clip_src'],
                    scene_clip_ref=batch['scene_clip_ref'])
        db_emb_cache[scene_id] = {
            'emb': F.normalize(out['ref_emb'], dim=-1).cpu(),
            'scene_clip': batch['scene_clip_ref'].cpu(),
            'labels': set(get_base_label(g.nodes[nid].label) for nid in g.nodes)
        }

torch.save(db_emb_cache, '/content/drive/MyDrive/VLSG_Files/db_emb_cache.pt')
print("✓ Saved db_emb_cache.pt")

# ── Precompute query embeddings ────────────────────────────
print("\nPrecomputing query embeddings...")
query_emb_cache = {}
with torch.no_grad():
    for key, g in tqdm(query_graphs.items(), desc="Queries"):
        batch = build_batch_from_cache(g, node_clip_cache, rel_clip_cache,
                                        scene_clip_cache, key, device)
        out = model(batch, scene_clip_src=batch['scene_clip_src'],
                    scene_clip_ref=batch['scene_clip_ref'])
        query_emb_cache[key] = {
            'emb': F.normalize(out['src_emb'], dim=-1).cpu(),
            'scene_clip': batch['scene_clip_src'].cpu(),
            'labels': set(get_base_label(g.nodes[nid].label) for nid in g.nodes),
            'scene_id': g.scene_id
        }

torch.save(query_emb_cache, '/content/drive/MyDrive/VLSG_Files/query_emb_cache.pt')
print("✓ Saved query_emb_cache.pt")
print("\n✅ All done. Run eval script with --use_cache flag.")