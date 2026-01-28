"""
Evaluation on ScanScribe text descriptions.

Converts ScanScribe graphs (from text) to your training format,
then matches against your 3RScan database.
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
import clip

# Import wrapper
sys.path.append('.')
from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner
from src.models.sgaligner.src.aligner.dual_scene_aligner_wrapper import load_model_with_matching


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load CLIP for converting text labels to embeddings
print("Loading CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=device)
print("✓ CLIP loaded")


def scanscribe_to_your_format(scanscribe_graph, scene_id):
    """
    Convert ScanScribe graph format to your training format.
    
    ScanScribe has: label + word2vec/ada embeddings
    Your format needs: label + centroid + color + CLIP
    """
    nodes_dict = {}
    
    for node in scanscribe_graph['nodes']:
        # Get CLIP embedding from label
        label = node['label']
        
        with torch.no_grad():
            tokens = clip.tokenize([label]).to(device)
            clip_emb = clip_model.encode_text(tokens)
            clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)
        
        nodes_dict[node['id']] = {
            "label": label,
            "base_label": label,
            "centroid": [0.0, 0.0, 0.0],  # No spatial info from text
            "mean_color": [128.0, 128.0, 128.0],  # Default gray
            "radius": 0.4,
            "clip_text_emb": clip_emb[0].cpu().numpy().tolist()
        }
    
    # Convert edges
    edges_text = []
    for edge in scanscribe_graph['edges']:
        edges_text.append({
            "subject": edge['source'],
            "object": edge['target'],
            "relation": edge['relationship']
        })
    
    return {
        "scene_id": scene_id,
        "nodes": nodes_dict,
        "edges_text": edges_text
    }


def load_scene_graph(scene_path):
    """Load a 3RScan scene graph from JSON."""
    with open(scene_path, 'r') as f:
        data = json.load(f)
    return data


def scene_graph_to_batch(query_data, db_data, device):
    """
    Convert two scene graphs to model batch format.
    Works for both ScanScribe queries and 3RScan database scenes.
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
        
        # Geometric edges (k-NN based on centroids)
        centroids = np.array([nodes[nid]['centroid'] for nid in node_ids], dtype=float)
        N = len(node_ids)
        K = 5
        
        dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        
        if N > 1:
            knn_idx = np.argsort(dmat, axis=1)[:, :min(K, N-1)]
        else:
            knn_idx = np.array([]).reshape(0, 0)
        
        geom_edge_index = []
        geom_edge_attr = []
        
        for i in range(N):
            ci = centroids[i]
            ri = nodes[node_ids[i]].get('radius', 0.4)
            
            for j in (knn_idx[i] if N > 1 else []):
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
        
        # Text edges (semantic relations)
        text_relations = scene_data.get('edges_text', [])
        text_edge_index = []
        text_rel_ids = []
        
        for r in text_relations:
            s = id2idx.get(str(r.get('subject', '')))
            o = id2idx.get(str(r.get('object', '')))
            
            if s is not None and o is not None:
                text_edge_index.append([s, o])
                text_rel_ids.append(1)  # Dummy relation ID
        
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


def evaluate_scanscribe(model, scanscribe_graphs, db_scene_paths, 
                        scene_to_group, top_k=[1, 3, 5, 10]):
    """
    Evaluate on ScanScribe text descriptions.
    
    For each text description:
    1. Convert to scene graph (already done in scanscribe_graphs)
    2. Convert to your format
    3. Match against database of 3RScan scenes
    4. Check if correct room is in top-k
    """
    model.eval()
    
    print(f"\n{'='*70}")
    print("Evaluating on ScanScribe")
    print(f"{'='*70}")
    
    # Build database scene ID to path mapping
    db_scene_id_to_path = {}
    for path in db_scene_paths:
        scene_id = os.path.basename(path).replace('.json', '')
        db_scene_id_to_path[scene_id] = path
    
    print(f"Database scenes: {len(db_scene_id_to_path)}")
    print(f"ScanScribe scenes: {len(scanscribe_graphs)}")
    
    # Filter ScanScribe to only scenes in our database
    matching_scenes = []
    for scene_id in scanscribe_graphs.keys():
        if scene_id in db_scene_id_to_path:
            matching_scenes.append(scene_id)
    
    print(f"Matching scenes: {len(matching_scenes)}")
    
    # DEBUG: Show which scenes matched
    print(f"\nFirst 10 matching scenes:")
    for i, scene_id in enumerate(matching_scenes[:10]):
        num_texts = len(scanscribe_graphs[scene_id])
        room_id = scene_to_group.get(scene_id, "Unknown")
        print(f"  {i+1}. {scene_id} (room: {room_id}, {num_texts} texts)")
    
    if len(matching_scenes) == 0:
        print("⚠️  No matching scenes between ScanScribe and your database!")
        return {k: (0.0, 0.0) for k in top_k}
    
    # Evaluate
    results = {k: [] for k in top_k}
    
    # Counter for debug prints
    eval_count = 0
    
    # For each ScanScribe scene
    for scene_id in tqdm(matching_scenes, desc="Evaluating ScanScribe"):
        scene_texts = scanscribe_graphs[scene_id]
        
        # For each text description of this scene
        for text_id, scanscribe_graph in scene_texts.items():
            eval_count += 1
            
            # DEBUG: Print first 3 evaluations
            if eval_count <= 3:
                print(f"\n{'='*70}")
                print(f"DEBUG: Evaluation {eval_count}")
                print(f"{'='*70}")
                print(f"Query Scene: {scene_id}")
                print(f"Text ID: {text_id}")
                print(f"Ground Truth Room: {scene_to_group.get(scene_id)}")
                print(f"\nQuery Graph:")
                print(f"  Nodes: {len(scanscribe_graph['nodes'])}")
                node_labels = [n['label'] for n in scanscribe_graph['nodes'][:5]]
                print(f"  First 5 node labels: {', '.join(node_labels)}")
                print(f"  Edges: {len(scanscribe_graph['edges'])}")
            
            # Convert to your format
            query_data = scanscribe_to_your_format(scanscribe_graph, scene_id)
            
            # Get ground truth room
            true_group = scene_to_group.get(scene_id)
            
            # Match against all database scenes
            match_scores = []
            db_scene_ids = []
            
            for db_scene_id, db_path in db_scene_id_to_path.items():
                db_data = load_scene_graph(db_path)
                
                # Create batch
                batch = scene_graph_to_batch(query_data, db_data, device)
                
                # Get matching probability
                with torch.no_grad():
                    out = model(batch)
                    matching_prob = out["matching_prob"]
                
                match_scores.append(matching_prob.item())
                db_scene_ids.append(db_scene_id)
            
            # Sort by matching score (highest first)
            match_scores = np.array(match_scores)
            sorted_indices = np.argsort(match_scores)[::-1]
            
            # DEBUG: Print ranking for first 3 evaluations
            if eval_count <= 3:
                print(f"\nMatching against {len(db_scene_ids)} database scenes...")
                print(f"\nTop 10 predictions:")
                for rank in range(min(10, len(sorted_indices))):
                    idx = sorted_indices[rank]
                    pred_scene_id = db_scene_ids[idx]
                    pred_group = scene_to_group.get(pred_scene_id)
                    score = match_scores[idx]
                    
                    is_correct = "✓ CORRECT" if pred_group == true_group else "✗"
                    pred_group_str = pred_group if pred_group is not None else "UNKNOWN"

                    print(
                        f"  Rank {rank+1}: {pred_scene_id:40s} "
                        f"(room: {pred_group_str:40s}) "
                        f"score={score:.4f} {is_correct}"
                    )                
                # Check top-k results for this query
                print(f"\nResults for this query:")
                for k in [1, 3, 5]:
                    top_k_scenes = [db_scene_ids[i] for i in sorted_indices[:k]]
                    top_k_groups = [scene_to_group.get(sid) for sid in top_k_scenes]
                    correct = true_group in top_k_groups
                    print(f"  Top-{k}: {'✓ Correct' if correct else '✗ Wrong'}")
            
            # Check top-k
            for k in top_k:
                top_k_scenes = [db_scene_ids[i] for i in sorted_indices[:k]]
                top_k_groups = [scene_to_group.get(sid) for sid in top_k_scenes]
                
                # Success if correct room in top-k
                correct = 1 if true_group in top_k_groups else 0
                results[k].append(correct)
    
    # Compute accuracy
    accuracy = {}
    for k in top_k:
        if results[k]:
            mean_acc = np.mean(results[k])
            std_acc = np.std(results[k])
            accuracy[k] = (mean_acc, std_acc)
        else:
            accuracy[k] = (0.0, 0.0)
    
    print(f"\n{'='*70}")
    print(f"Evaluation Summary")
    print(f"{'='*70}")
    print(f"Total queries evaluated: {eval_count}")
    print(f"Scenes with texts: {len(matching_scenes)}")
    print(f"Database scenes: {len(db_scene_id_to_path)}")
    
    print(f"\nResults:")
    for k in top_k:
        mean, std = accuracy[k]
        num_correct = int(mean * len(results[k]))
        print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}% ({num_correct}/{len(results[k])} correct)")
    
    model.train()
    return accuracy


def main(args):
    print(f"\n{'='*70}")
    print("ScanScribe → 3RScan Evaluation")
    print(f"{'='*70}\n")
    
    # Load metadata
    print("Loading 3RScan metadata...")
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
    
    # Load your 3RScan database
    print(f"\nLoading 3RScan database from {args.dataset_dir}...")
    db_scene_paths = sorted([
        os.path.join(args.dataset_dir, f) 
        for f in os.listdir(args.dataset_dir) 
        if f.endswith('.json')
    ])
    
    print(f"Database scenes: {len(db_scene_paths)}")
    
    # Load ScanScribe test graphs
    print(f"\nLoading ScanScribe from {args.scanscribe_path}...")
    scanscribe_graphs = torch.load(args.scanscribe_path, map_location='cpu', weights_only=False)
    
    print(f"ScanScribe scenes: {len(scanscribe_graphs)}")
    total_texts = sum(len(texts) for texts in scanscribe_graphs.values())
    print(f"Total text descriptions: {total_texts}")
    
    # Load model
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
        use_cosine=True,
        device=device
    )
    
    # Evaluate
    accuracy = evaluate_scanscribe(
        model,
        scanscribe_graphs,
        db_scene_paths,
        scene_to_group,
        top_k=[1, 3, 5, 10]
    )
    
    # Final results
    print(f"\n{'='*70}")
    print("FINAL RESULTS - ScanScribe")
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
                       help='Directory with 3RScan scene graph JSONs (your database)')
    parser.add_argument('--scanscribe_path', type=str,
                       default='playground/graph_models/data_checkpoints/processed_data/testing/scanscribe_graphs_test_final_no_graph_min.pt',
                       help='Path to ScanScribe test graphs')
    parser.add_argument('--metadata_path', type=str,
                       default='/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/meta_files/3RScan.json',
                       help='Path to 3RScan.json metadata')
    parser.add_argument('--num_relations', type=int, default=50,
                       help='Number of relation types')
    
    args = parser.parse_args()
    main(args)