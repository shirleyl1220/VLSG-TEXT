"""
Evaluation script for DualSceneAligner using the EXACT same methodology as BigGNN.

This ensures fair comparison between the two models.
"""

import time
import argparse
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random

sys.path.append('../data_processing')
sys.path.append('../../../')
from scene_graph import SceneGraph
from data_distribution_analysis.helper import get_matching_subgraph

# Import wrapper
sys.path.append('../../../../')
from dual_scene_aligner_with_matching import load_model_with_matching

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(42)


def convert_scene_graph_to_batch(query_graph, db_graph, device):
    """
    Convert two SceneGraph objects to the batch format expected by DualSceneAligner.
    
    NOTE: You may need to adjust this based on your exact SceneGraph format.
    """
    # Helper: get node features
    def get_node_features(graph):
        node_feats = []
        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            feat = np.array(node.features)
            
            # Pad to 518 dimensions (adjust if your format is different)
            if len(feat) < 518:
                feat = np.pad(feat, (0, 518 - len(feat)), mode='constant')
            elif len(feat) > 518:
                feat = feat[:518]
            
            node_feats.append(feat)
        
        return torch.tensor(np.array(node_feats), dtype=torch.float32)
    
    # Helper: get edge info
    def get_edge_info(graph):
        edge_idx = graph.edge_idx
        edge_feats = graph.edge_features
        
        if len(edge_idx) > 0 and len(edge_idx[0]) > 0:
            edges = torch.tensor(edge_idx, dtype=torch.long)
            
            if edge_feats is not None and len(edge_feats) > 0:
                geom_attr = torch.tensor(np.array(edge_feats), dtype=torch.float32)
                if geom_attr.dim() == 1:
                    geom_attr = geom_attr.unsqueeze(-1)
                if geom_attr.size(-1) < 6:
                    geom_attr = F.pad(geom_attr, (0, 6 - geom_attr.size(-1)))
            else:
                geom_attr = torch.zeros(edges.size(1), 6)
            
            text_edges = edges.clone()
            text_attr = torch.ones(edges.size(1), 1)
        else:
            edges = torch.zeros(2, 0, dtype=torch.long)
            geom_attr = torch.zeros(0, 6)
            text_edges = torch.zeros(2, 0, dtype=torch.long)
            text_attr = torch.zeros(0, 1)
        
        return edges, geom_attr, text_edges, text_attr
    
    # Get features
    query_nodes = get_node_features(query_graph)
    query_edges, query_geom_attr, query_text_edges, query_text_attr = get_edge_info(query_graph)
    
    db_nodes = get_node_features(db_graph)
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


def eval_acc_dual_aligner(model, database_3dssg, dataset, mode='scanscribe', 
                          eval_iter_count=100, out_of=10, valid_top_k=[1, 3, 5, 10]):
    """
    Evaluate DualSceneAligner using EXACT same methodology as BigGNN.
    """
    model.eval()
    
    print(f"\n{'='*70}")
    print(f"Evaluating on {mode}")
    print(f"{'='*70}")
    
    # Organize by scene (SAME AS BIGGNN)
    buckets = {}
    for idx, g in enumerate(dataset):
        if g.scene_id not in buckets:
            buckets[g.scene_id] = []
        buckets[g.scene_id].append(idx)
    
    print(f"Unique scenes: {len(buckets)}, Total graphs: {len(dataset)}")
    
    # Evaluation (SAME AS BIGGNN)
    eval_iters = 100
    all_valid = {}
    
    for eval_round in tqdm(range(eval_iters), desc=f"Eval {mode}"):
        valid = {k: [] for k in valid_top_k}
        
        sampled_test_indices = [
            [random.sample(buckets[g], 1)[0] for g in random.sample(list(buckets.keys()), out_of)]
            for _ in range(eval_iter_count)
        ]
        
        for t_set in sampled_test_indices:
            true_match = []
            match_scores = []
            
            for i in t_set:
                query = dataset[t_set[0]]
                db = database_3dssg[dataset[i].scene_id]
                
                # Subgraph matching (SAME AS BIGGNN)
                query_subgraph, db_subgraph = get_matching_subgraph(query, db)
                if db_subgraph is None or len(db_subgraph.nodes) <= 1 or len(db_subgraph.edge_idx[0]) < 1:
                    db_subgraph = db
                if query_subgraph is None or len(query_subgraph.nodes) <= 1 or len(query_subgraph.edge_idx[0]) < 1:
                    query_subgraph = query
                
                # Convert and evaluate
                batch = convert_scene_graph_to_batch(query_subgraph, db_subgraph, device)
                
                with torch.no_grad():
                    out = model(batch)
                    matching_prob = out["matching_prob"]
                
                match_scores.append(matching_prob.item())
                true_match.append(1 if query.scene_id == db.scene_id else 0)
            
            # Sort and check top-k (SAME AS BIGGNN)
            match_scores = np.array(match_scores)
            true_match = np.array(true_match)
            sorted_indices = np.argsort(match_scores)
            true_match = true_match[sorted_indices]
            
            for k in valid_top_k:
                valid[k].append(1 if 1 in true_match[-k:] else 0)
        
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
    parser.add_argument('--use_cosine', action='store_true')
    args = parser.parse_args()
    
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Using cosine: {args.use_cosine}\n")
    
    # Load model
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
        use_cosine=args.use_cosine,
        device=device
    )
    
    # Load data
    print("Loading data...")
    _3dssg_scenes = torch.load('../data_checkpoints/processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt', 
                               weights_only=False, map_location='cpu')
    _3dssg_graphs = {}
    for sid in tqdm(_3dssg_scenes, desc="3DSSG"):
        _3dssg_graphs[sid] = SceneGraph(sid, graph_type='3dssg', graph=_3dssg_scenes[sid],
                                       max_dist=1.0, embedding_type='word2vec', use_attributes=True)
    
    scanscribe_test = torch.load('../data_checkpoints/processed_data/testing/scanscribe_graphs_test_final_no_graph_min.pt',
                                 weights_only=False, map_location='cpu')
    scanscribe_graphs = {}
    for sid in tqdm(scanscribe_test, desc="ScanScribe"):
        for tid in scanscribe_test[sid].keys():
            key = f"{sid}_{str(tid).zfill(5)}"
            scanscribe_graphs[key] = SceneGraph(sid, txt_id=tid, graph_type='scanscribe',
                                               graph=scanscribe_test[sid][tid],
                                               embedding_type='word2vec', use_attributes=True)
    
    scanscribe_graphs = {k: v for k, v in scanscribe_graphs.items() if len(v.edge_idx[0]) >= 1}
    
    human_test = torch.load('../data_checkpoints/processed_data/human/human_graphs_processed.pt',
                           weights_only=False, map_location='cpu')
    human_graphs = {k: SceneGraph(k.split('_')[0], graph_type='human', graph=human_test[k],
                                 embedding_type='word2vec', use_attributes=True)
                   for k in human_test if k.split('_')[0] in _3dssg_graphs}
    
    print(f"\n3DSSG: {len(_3dssg_graphs)}, ScanScribe: {len(scanscribe_graphs)}, Human: {len(human_graphs)}\n")
    
    # Evaluate
    scanscribe_acc = eval_acc_dual_aligner(model, _3dssg_graphs, list(scanscribe_graphs.values()), 
                                          mode='scanscribe')
    human_acc = eval_acc_dual_aligner(model, _3dssg_graphs, list(human_graphs.values()), 
                                     mode='human')
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print("\nScanScribe:")
    for k in [1, 3, 5, 10]:
        print(f"  Top-{k}: {scanscribe_acc[k][0]*100:.2f}% ± {scanscribe_acc[k][1]*100:.2f}%")
    print("\nHuman:")
    for k in [1, 3, 5, 10]:
        print(f"  Top-{k}: {human_acc[k][0]*100:.2f}% ± {human_acc[k][1]*100:.2f}%")
    print(f"{'='*70}\n")