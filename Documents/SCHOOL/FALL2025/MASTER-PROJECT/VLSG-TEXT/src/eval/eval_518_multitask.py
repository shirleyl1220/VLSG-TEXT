"""
Evaluation script for SimpleGraphMatcher (518-dim + Scene CLIP fusion).

Properly loads the SimpleGraphMatcher wrapper with fusion layer.
"""

import time
import argparse
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import clip
from pathlib import Path

sys.path.append('../data_processing')
sys.path.append('../../../')
from scene_graph import SceneGraph
from helper import get_matching_subgraph

# Import base model
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
sys.path.append('../../../../')
from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner
import torch.nn as nn

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load CLIP
print("Loading CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=device)
print("✓ CLIP loaded")

random.seed(42)


# ============================================================
# Model Definition (Must match training script!)
# ============================================================

class SimpleGraphMatcher(nn.Module):
    """Just GNN + Scene CLIP fusion. No overlap head."""
    def __init__(self, base_model, scene_clip_dim=512, hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        
        # Simple fusion (MUST match training!)
        self.fusion = nn.Sequential(
            nn.LayerNorm(base_model.hidden_dim + scene_clip_dim),
            nn.Linear(base_model.hidden_dim + scene_clip_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, batch, scene_clip_src, scene_clip_ref):
        # GNN
        out = self.base_model(batch)
        gnn_src = out["src_emb"]
        gnn_ref = out["ref_emb"]
        
        # Fusion
        src_combined = torch.cat([gnn_src, scene_clip_src], dim=-1)
        ref_combined = torch.cat([gnn_ref, scene_clip_ref], dim=-1)
        
        src_emb = self.fusion(src_combined)
        ref_emb = self.fusion(ref_combined)
        
        return {
            "src_emb": src_emb,
            "ref_emb": ref_emb
        }


# ============================================================
# Helper Functions
# ============================================================

def get_clip_embedding(label, clip_model, device):
    """Get CLIP embedding for a label."""
    with torch.no_grad():
        tokens = clip.tokenize([label]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()


def get_scene_clip_embedding(labels_list, clip_model, device):
    """Get scene-level CLIP from list of object labels."""
    unique_labels = list(set(labels_list))[:10]
    scene_desc = f"A room with {', '.join(unique_labels)}"
    
    with torch.no_grad():
        tokens = clip.tokenize([scene_desc]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()


def convert_scene_graph_to_batch(query_graph, db_graph, clip_model, device):
    """
    Convert two SceneGraph objects to 518-dim format + scene CLIP.
    
    Node features: centroid(3) + color(3) + node_CLIP(512) = 518 dims
    Scene CLIP: 512 dims (separate, not in nodes)
    """
    
    def get_node_features_518(graph):
        """Convert SceneGraph to 518-dim features (NO scene CLIP in nodes!)."""
        node_feats = []
        
        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            
            # Centroid (3D) - zero out for fair comparison
            centroid = np.zeros(3, dtype=np.float32)
            
            # Color (3D) - default gray
            color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            
            # Node-level CLIP (512D)
            node_clip = get_clip_embedding(node.label, clip_model, device)
            
            # Concatenate: 3 + 3 + 512 = 518
            feat = np.concatenate([centroid, color, node_clip])
            node_feats.append(feat)
        
        return torch.tensor(np.array(node_feats), dtype=torch.float32)
    
    def get_scene_clip_512(graph):
        """Get scene-level CLIP (512D)."""
        labels = [graph.nodes[nid].label for nid in graph.nodes]
        return get_scene_clip_embedding(labels, clip_model, device)
    
    def get_edge_info(graph):
        """Get edge information from SceneGraph."""
        edge_idx = graph.edge_idx
        edge_feats = graph.edge_features
        
        num_nodes = len(graph.nodes)
        
        if len(edge_idx) > 0 and len(edge_idx[0]) > 0:
            edges = torch.tensor(edge_idx, dtype=torch.long)
            
            # Filter out invalid edges
            valid_mask = (edges[0] < num_nodes) & (edges[1] < num_nodes) & (edges[0] >= 0) & (edges[1] >= 0)
            edges = edges[:, valid_mask]
            
            if edges.size(1) == 0:
                return (torch.zeros(2, 0, dtype=torch.long),
                       torch.zeros(0, 8, dtype=torch.float32),
                       torch.zeros(2, 0, dtype=torch.long),
                       torch.zeros(0, 1, dtype=torch.float32))
            
            num_edges = edges.size(1)
            
            # Geometric attributes (8D)
            geom_attr = torch.zeros(num_edges, 8, dtype=torch.float32)
            
            if edge_feats is not None and len(edge_feats) > 0:
                edge_feats_tensor = torch.tensor(np.array(edge_feats), dtype=torch.float32)
                if edge_feats_tensor.dim() == 1:
                    edge_feats_tensor = edge_feats_tensor.unsqueeze(-1)
                
                edge_feats_tensor = edge_feats_tensor[valid_mask]
                
                feat_dim = min(8, edge_feats_tensor.size(-1))
                geom_attr[:, :feat_dim] = edge_feats_tensor[:, :feat_dim]
            
            # Text edges
            text_edges = edges.clone()
            text_attr = torch.ones(num_edges, 1, dtype=torch.float32)
        else:
            edges = torch.zeros(2, 0, dtype=torch.long)
            geom_attr = torch.zeros(0, 8, dtype=torch.float32)
            text_edges = torch.zeros(2, 0, dtype=torch.long)
            text_attr = torch.zeros(0, 1, dtype=torch.float32)
        
        return edges, geom_attr, text_edges, text_attr
    
    # Get features
    query_nodes = get_node_features_518(query_graph)
    query_edges, query_geom_attr, query_text_edges, query_text_attr = get_edge_info(query_graph)
    query_scene_clip = get_scene_clip_512(query_graph)
    
    db_nodes = get_node_features_518(db_graph)
    db_edges, db_geom_attr, db_text_edges, db_text_attr = get_edge_info(db_graph)
    db_scene_clip = get_scene_clip_512(db_graph)
    
    # Create batch
    batch = {
        "node_feats_src": query_nodes.to(device),
        "geom_edges_src": query_edges.to(device),
        "geom_attr_src": query_geom_attr.to(device),
        "text_edges_src": query_text_edges.to(device),
        "text_attr_src": query_text_attr.to(device),
        
        "node_feats_ref": db_nodes.to(device),
        "geom_edges_ref": db_edges.to(device),
        "geom_attr_ref": db_geom_attr.to(device),
        "text_edges_ref": db_text_edges.to(device),
        "text_attr_ref": db_text_attr.to(device),
        
        "src_batch": torch.zeros(query_nodes.size(0), dtype=torch.long).to(device),
        "ref_batch": torch.zeros(db_nodes.size(0), dtype=torch.long).to(device),
        "batch_size": 1,
        
        # Scene CLIP (separate!)
        "scene_clip_src": torch.tensor(query_scene_clip, dtype=torch.float32).unsqueeze(0).to(device),
        "scene_clip_ref": torch.tensor(db_scene_clip, dtype=torch.float32).unsqueeze(0).to(device),
    }
    
    return batch


# ============================================================
# Evaluation Function
# ============================================================

def eval_acc_dual_aligner(model, database_3dssg, dataset, clip_model, mode='scanscribe', 
                          eval_iter_count=100, out_of=10, valid_top_k=[1, 3, 5, 10]):
    """
    Evaluate SimpleGraphMatcher.
    """
    model.eval()
    
    print(f"\n{'='*70}")
    print(f"Evaluating on {mode}")
    print(f"{'='*70}")
    
    # Organize by scene
    buckets = {}
    for idx, g in enumerate(dataset):
        if g.scene_id not in buckets:
            buckets[g.scene_id] = []
        buckets[g.scene_id].append(idx)
    
    print(f"Unique scenes: {len(buckets)}, Total graphs: {len(dataset)}")
    
    # Evaluation
    eval_iters = 10
    all_valid = {}
    
    debug_count = 0
    
    for eval_round in tqdm(range(eval_iters), desc=f"Eval {mode}"):
        valid = {k: [] for k in valid_top_k}
        
        sampled_test_indices = [
            [random.sample(buckets[g], 1)[0] for g in random.sample(list(buckets.keys()), out_of)]
            for _ in range(eval_iter_count)
        ]
        
        for batch_idx, t_set in enumerate(sampled_test_indices):
            match_scores = []
            scene_ids = []
            
            query_scene_id = dataset[t_set[0]].scene_id
            
            for i in t_set:
                query = dataset[t_set[0]]
                db = database_3dssg[dataset[i].scene_id]
                
                # Optional: Subgraph matching
                query_subgraph = query
                db_subgraph = db
                
                # Convert to batch
                batch = convert_scene_graph_to_batch(query_subgraph, db_subgraph, clip_model, device)
                
                with torch.no_grad():
                    # Forward through model
                    out = model(
                        batch,
                        scene_clip_src=batch['scene_clip_src'],
                        scene_clip_ref=batch['scene_clip_ref']
                    )
                    
                    # Compute similarity
                    src_emb = out['src_emb']
                    ref_emb = out['ref_emb']
                    
                    # Cosine similarity
                    src_norm = F.normalize(src_emb, dim=-1)
                    ref_norm = F.normalize(ref_emb, dim=-1)
                    similarity = (src_norm * ref_norm).sum().item()
                    
                    match_scores.append(similarity)
                    scene_ids.append(dataset[i].scene_id)
            
            # Sort by similarity (high to low)
            match_scores = np.array(match_scores)
            sorted_indices = np.argsort(match_scores)[::-1]
            
            # DEBUG: Show first 5 batches
            if debug_count < 21:
                print(f"\n{'='*70}")
                print(f"DEBUG Batch {debug_count + 1} (Round {eval_round}, Batch {batch_idx})")
                print(f"{'='*70}")
                print(f"Query scene: {query_scene_id}")
                
                print(f"\n🎯 TOP 10 PREDICTIONS:")
                for rank_idx, idx in enumerate(sorted_indices[:10]):
                    scene_id = scene_ids[idx]
                    score = match_scores[idx]
                    is_correct = "✓ CORRECT" if scene_id == query_scene_id else "✗ wrong"
                    print(f"  Rank {rank_idx+1}: {scene_id:40s} score={score:.4f} {is_correct}")
                
                # Ground truth rank
                gt_rank = None
                for rank_idx, idx in enumerate(sorted_indices):
                    if scene_ids[idx] == query_scene_id:
                        gt_rank = rank_idx + 1
                        break
                
                print(f"\n  📊 Ground truth ranked at: {gt_rank}/{len(sorted_indices)}")
                
                if gt_rank and gt_rank <= 3:
                    print(f"  ✅ GOOD!")
                elif gt_rank and gt_rank <= 5:
                    print(f"  ⚠️  OKAY")
                else:
                    print(f"  ❌ POOR")
                
                print(f"{'='*70}\n")
                debug_count += 1
            
            # Check top-k
            for k in valid_top_k:
                top_k_scenes = [scene_ids[idx] for idx in sorted_indices[:k]]
                valid[k].append(1 if query_scene_id in top_k_scenes else 0)
        
        for k in valid_top_k:
            if k not in all_valid:
                all_valid[k] = []
            all_valid[k].append(np.mean(valid[k]))
    
    # Results
    accuracy = {k: (np.mean(all_valid[k]), np.std(all_valid[k])) for k in valid_top_k}
    
    print(f"\nResults:")
    for k in accuracy:
        mean, std = accuracy[k]
        print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}%")
    
    model.train()
    return accuracy


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_relations', type=int, default=9)
    args = parser.parse_args()
    
    print(f"\nCheckpoint: {args.checkpoint}\n")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Create dummy relation embeddings
    dummy_rel_emb = nn.Embedding(args.num_relations, 64).to(device)
    nn.init.normal_(dummy_rel_emb.weight, mean=0, std=0.1)
    
    # Create base model
    base_model = DualSceneAligner(
        node_input_dim=518,
        relation_dim=64,
        hidden_dim=256,
        rel_clip_matrix=dummy_rel_emb.weight,
        dropout=0.0
    ).to(device)
    
    # Wrap with SimpleGraphMatcher
    model = SimpleGraphMatcher(
        base_model=base_model,
        scene_clip_dim=512,
        hidden_dim=256
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Load data
    print("Loading 3DSSG database...")
    _3dssg_scenes = torch.load('/content/drive/MyDrive/VLSG_Files/3dssg_graphs_518D.pt', 
                               weights_only=False, map_location='cpu')
    _3dssg_graphs = {}
    for sid in tqdm(_3dssg_scenes, desc="3DSSG"):
        _3dssg_graphs[sid] = SceneGraph(sid, graph_type='3dssg', graph=_3dssg_scenes[sid],
                                       max_dist=1.0, embedding_type='word2vec', use_attributes=True)
    
    print(f"✓ Loaded {len(_3dssg_graphs)} 3DSSG scenes")
    
    print("Loading ScanScribe test...")
    scanscribe_test = torch.load('/content/drive/MyDrive/VLSG_Files/scanscribe_graphs_test_518D.pt',
                                 weights_only=False, map_location='cpu')
    scanscribe_graphs = {}
    for sid in tqdm(scanscribe_test, desc="ScanScribe"):
        for tid in scanscribe_test[sid].keys():
            key = f"{sid}_{str(tid).zfill(5)}"
            scanscribe_graphs[key] = SceneGraph(sid, txt_id=tid, graph_type='scanscribe',
                                               graph=scanscribe_test[sid][tid],
                                               embedding_type='word2vec', use_attributes=True)
    
    scanscribe_graphs = {k: v for k, v in scanscribe_graphs.items() if len(v.edge_idx[0]) >= 1}
    
    print(f"✓ Loaded {len(scanscribe_graphs)} ScanScribe queries\n")
    
    # Evaluate
    scanscribe_acc = eval_acc_dual_aligner(model, _3dssg_graphs, list(scanscribe_graphs.values()), 
                                          clip_model, mode='scanscribe')
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS - ScanScribe")
    print(f"{'='*70}")
    for k in [1, 3, 5, 10]:
        mean, std = scanscribe_acc[k]
        print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}%")
    print(f"{'='*70}\n")