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
            "clip_text_emb": clip_emb[0].detach().cpu().tolist()
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
        # Get nodes - handle both dict and list
        nodes_raw = scene_data.get('nodes')
        
        if nodes_raw is None:
            raise KeyError(f"No 'nodes' key found. Keys: {scene_data.keys()}")
        
        if isinstance(nodes_raw, dict):
            # Dict format: {node_id: {label, centroid, ...}}
            node_ids = list(nodes_raw.keys())
            nodes = [nodes_raw[nid] for nid in node_ids]
            id2idx = {str(nid): i for i, nid in enumerate(node_ids)}
        else:
            # List format: [{id, label, ...}, ...]
            nodes = nodes_raw
            node_ids = [str(n['id']) for n in nodes]
            id2idx = {str(nid): i for i, nid in enumerate(node_ids)}
        
        # Node features (518 dims)
        node_feats = []
        for i in range(len(nodes)):
            n = nodes[i]
            centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Zero out for fair comparison
            color = np.array(n['mean_color'], dtype=np.float32) / 255.0
            clip_vec = np.array(n.get('clip_text_emb', np.zeros(512)), dtype=np.float32)
            
            feat = np.concatenate([centroid, color, clip_vec])
            node_feats.append(feat)
        
        node_feats = torch.tensor(np.array(node_feats), dtype=torch.float32)
        
        # Geometric edges (k-NN based on centroids)
        centroids = np.array([nodes[i]['centroid'] for i in range(len(nodes))], dtype=float)
        N = len(nodes)
        K = 5
        
        geom_edge_index = []
        geom_edge_attr = []
        
        if N > 1:
            dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
            np.fill_diagonal(dmat, np.inf)
            knn_idx = np.argsort(dmat, axis=1)[:, :min(K, N-1)]
            
            for i in range(N):
                ci = centroids[i]
                ri = nodes[i].get('radius', 0.4)
                
                for j in knn_idx[i]:
                    cj = centroids[j]
                    rj = nodes[j].get('radius', 0.4)
                    
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


def evaluate_scanscribe(model, scanscribe_graphs, db_scene_paths, 
                        scene_to_group, top_k=[1, 3, 5, 10]):
    model.eval()
    
    # Build database scene ID to path mapping
    db_scene_id_to_path = {}
    for path in db_scene_paths:
        filename = os.path.basename(path).replace('.json', '')
        db_scene_id_to_path[filename] = path
    
    # GROUP database scenes by ROOM
    room_to_files = {}
    for filename, path in db_scene_id_to_path.items():
        # Extract base scene ID
        if '_text_' in filename:
            base_scene_id = filename.split('_text_')[0]
        else:
            base_scene_id = filename
        
        room_id = scene_to_group.get(base_scene_id)
        if room_id:
            if room_id not in room_to_files:
                room_to_files[room_id] = []
            room_to_files[room_id].append(path)
    
    print(f"Unique rooms in database: {len(room_to_files)}")
    
    # Compute ONE embedding per room (average of all scans)
    print("Computing room-level embeddings...")
    room_embeddings = {}
    
    for room_id, file_paths in tqdm(room_to_files.items(), desc="Processing rooms"):
        embeddings = []
        
        for db_path in file_paths:
            db_data = load_scene_graph(db_path)
            # Create dummy query (we only need ref embedding)
            dummy_query = db_data  # Use same data for both
            batch = scene_graph_to_batch(dummy_query, db_data, device)
            
            with torch.no_grad():
                out = model(batch)
                ref_emb = out["ref_emb"]
                embeddings.append(ref_emb)
        
        # Average all embeddings for this room
        room_embeddings[room_id] = torch.stack(embeddings).mean(dim=0)
    
    print(f"Computed {len(room_embeddings)} room embeddings")
    
    # NOW evaluate queries against ROOMS (not individual scans)
    results = {k: [] for k in top_k}
    eval_count = 0
    
    for scene_id in tqdm(scanscribe_graphs.keys(), desc="Evaluating"):
        scene_texts = scanscribe_graphs[scene_id]
        
        for text_id, scanscribe_graph in scene_texts.items():
            eval_count += 1
            
            # Convert query to your format
            query_data = scanscribe_to_your_format(scanscribe_graph, scene_id)
            true_group = scene_to_group.get(scene_id)
            
            # Get query embedding
            dummy_db = query_data
            batch = scene_graph_to_batch(query_data, dummy_db, device)
            
            with torch.no_grad():
                out = model(batch)
                query_emb = out["src_emb"]
            
            # Compare against ALL rooms
            match_scores = []
            room_ids = []
            
            for room_id, room_emb in room_embeddings.items():
                # Cosine similarity
                cos_sim = F.cosine_similarity(query_emb, room_emb, dim=-1)
                matching_prob = (cos_sim + 1) / 2
                
                match_scores.append(matching_prob.item())
                room_ids.append(room_id)
            
            # Sort by score
            match_scores = np.array(match_scores)
            sorted_indices = np.argsort(match_scores)[::-1]
            
            # Check top-k
            for k in top_k:
                top_k_rooms = [room_ids[i] for i in sorted_indices[:k]]
                correct = 1 if true_group in top_k_rooms else 0
                results[k].append(correct)

            if eval_count <= 3:
                print(f"\n{'='*70}")
                print(f"Query {eval_count}: {scene_id}")
                print(f"Ground truth: {true_group}")
                print(f"\nTop 10 predictions:")
                for rank in range(min(10, len(sorted_indices))):
                    idx = sorted_indices[rank]
                    pred_room = room_ids[idx]
                    score = match_scores[idx]
                    is_correct = "✓" if pred_room == true_group else "✗"
                    print(f"  Rank {rank+1}: {pred_room[:40]:40s} score={score:.4f} {is_correct}")
                
    # Compute accuracy
    accuracy = {}
    for k in top_k:
        mean_acc = np.mean(results[k])
        std_acc = np.std(results[k])
        accuracy[k] = (mean_acc, std_acc)
    
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
        and not f.startswith('metadata') 
        and not f.startswith('training_splits')
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
        hidden_dim=256,  # Your model outputs 256, not 128!
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
    parser.add_argument('--dataset_dir', type=str, 
                       default='src/datasets/combined_dataset_clip',
                       help='Directory with scene graph JSONs (your database)')
    parser.add_argument('--scanscribe_path', type=str,
                       default='/content/drive/MyDrive/VLSG_Files/scanscribe_graphs_test_final_no_graph_min.pt',
                       help='Path to ScanScribe test graphs')
    parser.add_argument('--metadata_path', type=str,
                       default='/content/drive/MyDrive/VLSG_Files/3RScan.json',
                       help='Path to 3RScan.json metadata')
    parser.add_argument('--num_relations', type=int, default=50,
                       help='Number of relation types')
    
    args = parser.parse_args()
    main(args)