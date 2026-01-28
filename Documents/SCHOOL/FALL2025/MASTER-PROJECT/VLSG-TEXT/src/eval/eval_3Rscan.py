"""
Evaluation script for DualSceneAligner on 3RScan data.
Uses EXACT same methodology as BigGNN evaluation.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import argparse
import sys
print("PYTHON EXECUTABLE:", sys.executable)
# Import wrapper
sys.path.append('.')
from src.models.sgaligner.src.aligner.dual_scene_aligner_wrapper import load_model_with_matching

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_scene_graph(scene_path):
    """Load a single scene graph from JSON file."""
    with open(scene_path, 'r') as f:
        data = json.load(f)
    return data


def scene_graph_to_batch(query_data, db_data, device):
    """
    Convert two scene graphs to model batch format.
    EXACTLY matches training data format.
    """
    def build_features(scene_data):
        nodes = scene_data['nodes']
        node_ids = list(nodes.keys())
        id2idx = {str(nid): i for i, nid in enumerate(node_ids)}
        
        # Node features (518 dims)
        node_feats = []
        for nid in node_ids:
            n = nodes[nid]
            centroid = np.array(n['centroid'], dtype=np.float32)
            color = np.array(n['mean_color'], dtype=np.float32) / 255.0
            clip_vec = np.array(n.get('clip_text_emb', np.zeros(512)), dtype=np.float32)
            
            feat = np.concatenate([centroid, color, clip_vec])
            node_feats.append(feat)
        
        node_feats = torch.tensor(np.array(node_feats), dtype=torch.float32)
        
        # Geometric edges (k-NN)
        centroids = np.array([nodes[nid]['centroid'] for nid in node_ids], dtype=float)
        N = len(node_ids)
        K = 5
        
        dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        knn_idx = np.argsort(dmat, axis=1)[:, :K]
        
        geom_edge_index = []
        geom_edge_attr = []
        
        for i in range(N):
            ci = centroids[i]
            ri = nodes[node_ids[i]].get('radius', 0.4)
            
            for j in knn_idx[i]:
                cj = centroids[j]
                rj = nodes[node_ids[j]].get('radius', 0.4)
                
                vec = cj - ci
                dist = float(np.linalg.norm(vec))
                feat = np.array([vec[0], vec[1], vec[2], dist, ri, rj, 0.0, 0.0], dtype=np.float32)
                
                geom_edge_index.append([i, j])
                geom_edge_attr.append(feat)
        
        if geom_edge_index:
            geom_edges = torch.tensor(geom_edge_index, dtype=torch.long).t()
            geom_attr = torch.tensor(geom_edge_attr, dtype=torch.float32)
        else:
            geom_edges = torch.zeros(2, 0, dtype=torch.long)
            geom_attr = torch.zeros(0, 8, dtype=torch.float32)
        
        # Text edges
        text_relations = scene_data.get('edges_text', [])
        text_edge_index = []
        text_rel_ids = []
        
        for r in text_relations:
            s = id2idx.get(str(r.get('subject', '')))
            o = id2idx.get(str(r.get('object', '')))
            
            if s is not None and o is not None:
                text_edge_index.append([s, o])
                text_rel_ids.append(1)
        
        if text_edge_index:
            text_edges = torch.tensor(text_edge_index, dtype=torch.long).t()
            text_attr = torch.tensor(text_rel_ids, dtype=torch.long).unsqueeze(-1)
        else:
            text_edges = torch.zeros(2, 0, dtype=torch.long)
            text_attr = torch.zeros(0, 1, dtype=torch.long)
        
        return node_feats, geom_edges, geom_attr, text_edges, text_attr
    
    # Build query and db
    q_feats, q_geom_e, q_geom_a, q_text_e, q_text_a = build_features(query_data)
    d_feats, d_geom_e, d_geom_a, d_text_e, d_text_a = build_features(db_data)
    
    batch = {
        "node_feats_src": q_feats.to(device),
        "geom_edges_src": q_geom_e.to(device),
        "geom_attr_src": q_geom_a.to(device),
        "text_edges_src": q_text_e.to(device),
        "text_attr_src": q_text_a.to(device),
        "src_batch": torch.zeros(q_feats.size(0), dtype=torch.long).to(device),
        
        "node_feats_ref": d_feats.to(device),
        "geom_edges_ref": d_geom_e.to(device),
        "geom_attr_ref": d_geom_a.to(device),
        "text_edges_ref": d_text_e.to(device),
        "text_attr_ref": d_text_a.to(device),
        "ref_batch": torch.zeros(d_feats.size(0), dtype=torch.long).to(device),
        
        "batch_size": 1
    }
    
    return batch


def eval_acc_dual_aligner(model, scene_paths, scene_to_group, 
                          eval_iter_count=100, out_of=10, valid_top_k=[1, 3, 5, 10]):
    """
    EXACT BigGNN evaluation methodology.
    
    For each iteration:
    1. Sample `out_of` scenes from different rooms
    2. Use first scene as query
    3. Compare query against all `out_of` scenes (including itself)
    4. Rank by matching_prob
    5. Check if true match is in top-k
    """
    model.eval()
    
    print(f"\n{'='*70}")
    print("Evaluating on 3RScan")
    print(f"{'='*70}")
    
    # Group scenes by room
    buckets = {}
    for scene_path in scene_paths:
        scene_id = os.path.basename(scene_path).replace('.json', '')
        group = scene_to_group.get(scene_id)
        
        if group:
            if group not in buckets:
                buckets[group] = []
            buckets[group].append(scene_path)
    
    print(f"Unique rooms: {len(buckets)}, Total scenes: {len(scene_paths)}")
    
    # Evaluation - EXACT BigGNN methodology
    eval_iters = 100
    all_valid = {}
    
    for eval_round in tqdm(range(eval_iters), desc="Eval rounds"):
        valid = {k: [] for k in valid_top_k}
        
        # Sample `eval_iter_count` test sets
        sampled_test_indices = []
        for _ in range(eval_iter_count):
            # Sample `out_of` rooms
            sampled_groups = random.sample(list(buckets.keys()), min(out_of, len(buckets)))
            # Sample one scene from each room
            test_set = [random.choice(buckets[g]) for g in sampled_groups]
            sampled_test_indices.append(test_set)
        
        for t_set in sampled_test_indices:
            query_path = t_set[0]  # First scene is query
            query_data = load_scene_graph(query_path)
            query_scene_id = os.path.basename(query_path).replace('.json', '')
            query_group = scene_to_group.get(query_scene_id)
            
            true_match = []
            match_scores = []
            
            # Compare query against all scenes in test set
            for db_path in t_set:
                db_data = load_scene_graph(db_path)
                db_scene_id = os.path.basename(db_path).replace('.json', '')
                db_group = scene_to_group.get(db_scene_id)
                
                # Create batch
                batch = scene_graph_to_batch(query_data, db_data, device)
                
                # Get matching probability
                with torch.no_grad():
                    out = model(batch)
                    matching_prob = out["matching_prob"]
                
                match_scores.append(matching_prob.item())
                # 1 if same room, 0 otherwise
                true_match.append(1 if query_group == db_group else 0)
            
            # Sort by matching probability (highest first)
            match_scores = np.array(match_scores)
            true_match = np.array(true_match)
            sorted_indices = np.argsort(match_scores)[::-1]  # Descending
            true_match_sorted = true_match[sorted_indices]
            
            # Check top-k
            for k in valid_top_k:
                # Success if ANY true match in top-k
                valid[k].append(1 if true_match_sorted[:k].sum() > 0 else 0)
        
        # Store results for this round
        for k in valid_top_k:
            if k not in all_valid:
                all_valid[k] = []
            all_valid[k].append(np.mean(valid[k]))
    
    # Compute final accuracy
    accuracy = {k: (np.mean(all_valid[k]), np.std(all_valid[k])) for k in valid_top_k}
    
    print(f"\nResults:")
    for k in accuracy:
        mean, std = accuracy[k]
        print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}%")
    
    model.train()
    return accuracy


def main(args):
    print(f"\n{'='*70}")
    print("3RScan Evaluation - BigGNN Methodology")
    print(f"{'='*70}\n")
    
    # Load metadata
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Build scene-to-room mapping
    scene_to_group = {}
    for entry in metadata:
        group_id = entry['reference']
        scene_to_group[group_id] = group_id
        
        for scan in entry.get('scans', []):
            scan_id = scan['reference']
            scene_to_group[scan_id] = group_id
    
    # Get all scene paths
    all_scene_files = sorted([
        os.path.join(args.dataset_dir, f) 
        for f in os.listdir(args.dataset_dir) 
        if f.endswith('.json')
    ])
    
    print(f"Found {len(all_scene_files)} scenes")
    print(f"Mapped to {len(set(scene_to_group.values()))} unique rooms")
    
    # Split train/test (80/20)
    random.shuffle(all_scene_files)
    split_idx = int(len(all_scene_files) * 0.8)
    train_files = all_scene_files[:split_idx]
    # test_files = all_scene_files[split_idx:]
    test_files = all_scene_files #just fo testing
    
    print(f"\nTrain scenes: {len(train_files)}")
    print(f"Test scenes: {len(test_files)}")
    
    # Load model with matching head
    print(f"\nLoading model: {args.checkpoint}")
    
    dummy_clip = torch.randn(args.num_relations, 512) * 0.1
    config = {
        'node_input_dim': 518,
        'relation_dim': 512,
        'hidden_dim': 128,
        'rel_clip_matrix': dummy_clip.to(device),
        'dropout': 0.1
    }
    
    model = load_model_with_matching(
        checkpoint_path=args.checkpoint,
        base_model_config=config,
        hidden_dim=128,
        use_cosine=True,  # Use cosine similarity as matching score
        device=device
    )
    
    # Evaluate
    accuracy = eval_acc_dual_aligner(
        model, 
        test_files, 
        scene_to_group,
        eval_iter_count=100,
        out_of=10,
        valid_top_k=[1, 3, 5, 10]
    )
    
    # Results
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    for k in [1, 3, 5, 10]:
        mean, std = accuracy[k]
        print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}%")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset_dir', type=str, default='scene_graphs_unique',
                       help='Directory with scene graph JSONs')
    parser.add_argument('--metadata_path', type=str, 
                       default='/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/meta_files/3RScan.json',
                       help='Path to 3RScan.json metadata')
    parser.add_argument('--num_relations', type=int, default=50,
                       help='Number of relation types')
    
    args = parser.parse_args()
    main(args)