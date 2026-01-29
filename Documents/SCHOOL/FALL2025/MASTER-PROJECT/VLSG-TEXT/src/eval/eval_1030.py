"""
Evaluation script for DualSceneAligner (1030-dim) using SceneGraph format.

Properly converts Word2Vec SceneGraph objects to CLIP+scene-CLIP format.
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

sys.path.append('../data_processing')
sys.path.append('../../../')
from scene_graph import SceneGraph
from helper import get_matching_subgraph

# Import wrapper
sys.path.append('../../../../')
from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner
from src.models.sgaligner.src.aligner.dual_scene_aligner_wrapper import load_model_with_matching

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load CLIP for converting labels
print("Loading CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=device)
print("✓ CLIP loaded")

random.seed(42)


def get_clip_embedding(label, clip_model, device):
    """Get CLIP embedding for a label."""
    with torch.no_grad():
        tokens = clip.tokenize([label]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()


def get_scene_clip_embedding(labels_list, clip_model, device):
    """Get scene-level CLIP from list of object labels."""
    # Create scene description
    unique_labels = list(set(labels_list))[:10]  # Top 10 unique objects
    scene_desc = f"A room with {', '.join(unique_labels)}"
    
    with torch.no_grad():
        tokens = clip.tokenize([scene_desc]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()


def convert_scene_graph_to_batch(query_graph, db_graph, clip_model, device):
    """
    Convert two SceneGraph objects to 1030-dim format.
    
    Features: centroid(3) + color(3) + node_CLIP(512) + scene_CLIP(512) = 1030
    """
    
    def get_node_features_1030(graph):
        """Convert SceneGraph to 1030-dim features."""
        node_feats = []
        labels = []
        
        # Collect all labels for scene-level CLIP
        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            labels.append(node.label)
        
        # Get scene-level CLIP (same for all nodes)
        scene_clip = get_scene_clip_embedding(labels, clip_model, device)
        
        # Build node features
        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            
            # Centroid (3D) - zero out for fair comparison
            centroid = np.zeros(3, dtype=np.float32)
            
            # Color (3D) - default gray
            color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            
            # Node-level CLIP (512D)
            node_clip = get_clip_embedding(node.label, clip_model, device)
            
            # Concatenate: 3 + 3 + 512 + 512 = 1030
            feat = np.concatenate([centroid, color, node_clip, scene_clip])
            node_feats.append(feat)
        
        return torch.tensor(np.array(node_feats), dtype=torch.float32)
    
    def get_edge_info(graph):
        """Get edge information from SceneGraph."""
        edge_idx = graph.edge_idx
        edge_feats = graph.edge_features
        
        if len(edge_idx) > 0 and len(edge_idx[0]) > 0:
            edges = torch.tensor(edge_idx, dtype=torch.long)
            num_edges = edges.size(1)
            
            # Create 8-dimensional geometric attributes
            geom_attr = torch.zeros(num_edges, 8, dtype=torch.float32)
            
            if edge_feats is not None and len(edge_feats) > 0:
                edge_feats_tensor = torch.tensor(np.array(edge_feats), dtype=torch.float32)
                if edge_feats_tensor.dim() == 1:
                    edge_feats_tensor = edge_feats_tensor.unsqueeze(-1)
                
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
    query_nodes = get_node_features_1030(query_graph)
    query_edges, query_geom_attr, query_text_edges, query_text_attr = get_edge_info(query_graph)
    
    db_nodes = get_node_features_1030(db_graph)
    db_edges, db_geom_attr, db_text_edges, db_text_attr = get_edge_info(db_graph)
    
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
        "batch_size": 1
    }
    
    return batch


def eval_acc_dual_aligner(model, database_3dssg, dataset, clip_model, mode='scanscribe', 
                          eval_iter_count=100, out_of=10, valid_top_k=[1, 3, 5, 10]):
    """
    Evaluate DualSceneAligner using same methodology as BigGNN.
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
    eval_iters = 10  # Reduced from 100 for speed
    all_valid = {}
    
    debug_count = 0  # Track debug outputs
    
    for eval_round in tqdm(range(eval_iters), desc=f"Eval {mode}"):
        valid = {k: [] for k in valid_top_k}
        
        sampled_test_indices = [
            [random.sample(buckets[g], 1)[0] for g in random.sample(list(buckets.keys()), out_of)]
            for _ in range(eval_iter_count)
        ]
        
        for batch_idx, t_set in enumerate(sampled_test_indices):
            true_match = []
            match_scores = []
            scene_ids = []
            
            query_scene_id = dataset[t_set[0]].scene_id
            
            for i in t_set:
                query = dataset[t_set[0]]
                db = database_3dssg[dataset[i].scene_id]
                
                # Subgraph matching (CRITICAL!)
                query_subgraph, db_subgraph = get_matching_subgraph(query, db)
                if db_subgraph is None or len(db_subgraph.nodes) <= 1 or len(db_subgraph.edge_idx[0]) < 1:
                    db_subgraph = db
                if query_subgraph is None or len(query_subgraph.nodes) <= 1 or len(query_subgraph.edge_idx[0]) < 1:
                    query_subgraph = query
                
                # Convert to 1030-dim format
                batch = convert_scene_graph_to_batch(query_subgraph, db_subgraph, clip_model, device)
                
                with torch.no_grad():
                    out = model(batch)
                    matching_prob = out["matching_prob"]
                
                match_scores.append(matching_prob.item())
                true_match.append(1 if query.scene_id == db.scene_id else 0)
                scene_ids.append(dataset[i].scene_id)
            
            # Sort and check top-k
            match_scores = np.array(match_scores)
            true_match = np.array(true_match)
            sorted_indices = np.argsort(match_scores)[::-1]  # High to low
            
            # DEBUG: Show first 3 batches of each round
            if debug_count < 3:
                print(f"\n{'='*70}")
                print(f"DEBUG Batch {debug_count + 1} (Round {eval_round}, Batch {batch_idx})")
                print(f"{'='*70}")
                print(f"Query scene: {query_scene_id}")
                print(f"\nTop 10 predictions (sorted by score):")
                for rank_idx, idx in enumerate(sorted_indices[:10]):
                    scene_id = scene_ids[idx]
                    score = match_scores[idx]
                    is_correct = "✓ CORRECT" if scene_id == query_scene_id else "✗ wrong"
                    print(f"  Rank {rank_idx+1}: {scene_id:40s} score={score:.4f} {is_correct}")
                
                # Show where ground truth is ranked
                gt_rank = None
                for rank_idx, idx in enumerate(sorted_indices):
                    if scene_ids[idx] == query_scene_id:
                        gt_rank = rank_idx + 1
                        break
                
                print(f"\n  Ground truth ranked at: {gt_rank}/{len(sorted_indices)}")
                print(f"{'='*70}\n")
                debug_count += 1
            
            # Sort for evaluation (ascending order for matching original logic)
            sorted_indices_asc = np.argsort(match_scores)
            true_match_sorted = true_match[sorted_indices_asc]
            
            for k in valid_top_k:
                valid[k].append(1 if 1 in true_match_sorted[-k:] else 0)
        
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_relations', type=int, default=50)
    parser.add_argument('--use_cosine', action='store_true', default=True)
    args = parser.parse_args()
    
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Using cosine: {args.use_cosine}\n")
    
    # Load model with 1030-dim config
    dummy_clip = torch.randn(args.num_relations, 512) * 0.1
    config = {
        'node_input_dim': 1030,  # ← UPDATED!
        'relation_dim': 512,
        'hidden_dim': 256,  # ← UPDATED!
        'rel_clip_matrix': dummy_clip.to(device),
        'dropout': 0.1
    }
    
    model = load_model_with_matching(
        checkpoint_path=args.checkpoint,
        base_model_config=config,
        hidden_dim=256,  # ← UPDATED!
        use_cosine=args.use_cosine,
        device=device
    )
    
    # Load ScanScribe data only
    print("Loading ScanScribe test data...")
    
    # Load 3DSSG database (needed for queries)
    print("Loading 3DSSG database...")
    _3dssg_scenes = torch.load('/Users/shirley/Documents/SCHOOL/SPRING25/masterproject/attempt2/whereami-text2sgm/playground/graph_models/data_checkpoints/processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt', 
                               weights_only=False, map_location='cpu')
    _3dssg_graphs = {}
    for sid in tqdm(_3dssg_scenes, desc="3DSSG"):
        _3dssg_graphs[sid] = SceneGraph(sid, graph_type='3dssg', graph=_3dssg_scenes[sid],
                                       max_dist=1.0, embedding_type='word2vec', use_attributes=True)
    
    print(f"✓ Loaded {len(_3dssg_graphs)} 3DSSG database scenes")
    
    # Load ScanScribe test queries
    scanscribe_test = torch.load('/Users/shirley/Documents/SCHOOL/SPRING25/masterproject/attempt2/whereami-text2sgm/playground/graph_models/data_checkpoints/processed_data/testing/scanscribe_graphs_test_final_no_graph_min.pt',
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
    
    # Evaluate on ScanScribe only
    scanscribe_acc = eval_acc_dual_aligner(model, _3dssg_graphs, list(scanscribe_graphs.values()), 
                                          clip_model, mode='scanscribe')
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS - ScanScribe")
    print(f"{'='*70}")
    for k in [1, 3, 5, 10]:
        mean, std = scanscribe_acc[k]
        print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}%")
    print(f"{'='*70}\n")