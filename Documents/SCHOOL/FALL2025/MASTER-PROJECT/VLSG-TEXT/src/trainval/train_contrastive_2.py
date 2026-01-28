"""
COMPLETE Training Script with Supervised Contrastive Loss

Integrated with your existing DualSceneAligner model.
This WILL fix your negative separation problem!
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import sys

# Import your model
sys.path.append('.')
from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss - The RIGHT loss for your task!
    
    Key differences from VICReg:
    1. Uses room labels (supervised)
    2. Directly pushes negatives apart
    3. Target: pos_sim=0.9+, neg_sim=0.1-
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, room_labels):
        """
        Args:
            embeddings: [batch_size, emb_dim]
            room_labels: [batch_size] - room IDs
        """
        # Normalize
        embeddings = F.normalize(embeddings, dim=1)
        
        # Similarity matrix
        sim_matrix = embeddings @ embeddings.T / self.temperature
        
        # Create masks
        labels = room_labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        neg_mask = (labels != labels.T).float()
        
        # Remove diagonal
        pos_mask.fill_diagonal_(0)
        
        # Numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Exp and mask negatives
        exp_logits = torch.exp(logits)
        exp_negatives = exp_logits * neg_mask
        
        # Log probability
        log_prob = logits - torch.log(exp_negatives.sum(1, keepdim=True) + 1e-12)
        
        # Mean over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-12)
        
        # Loss (negative log likelihood)
        loss = -mean_log_prob_pos.mean()
        
        return loss


def load_scene_graph(filepath):
    """Load scene graph from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def scene_graph_to_batch(graphs, device):
    """
    Convert list of scene graphs to model batch format.
    
    This is adapted from your existing code.
    """
    batch_node_feats_src = []
    batch_geom_edges_src = []
    batch_geom_attr_src = []
    batch_text_edges_src = []
    batch_text_attr_src = []
    batch_indices_src = []
    
    node_offset = 0
    
    for graph_data in graphs:
        nodes = graph_data['nodes']
        node_ids = list(nodes.keys())
        id2idx = {str(nid): i for i, nid in enumerate(node_ids)}
        
        # Node features
        node_feats = []
        for nid in node_ids:
            n = nodes[nid]
            centroid = np.array(n['centroid'], dtype=np.float32)
            color = np.array(n['mean_color'], dtype=np.float32) / 255.0
            clip_vec = np.array(n.get('clip_text_emb', np.zeros(512)), dtype=np.float32)
            
            feat = np.concatenate([centroid, color, clip_vec])
            node_feats.append(feat)
        
        node_feats = torch.tensor(np.array(node_feats), dtype=torch.float32)
        batch_node_feats_src.append(node_feats)
        
        # Geometric edges (k-NN)
        centroids = np.array([nodes[nid]['centroid'] for nid in node_ids], dtype=float)
        N = len(node_ids)
        K = 5
        
        if N > 1:
            dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
            np.fill_diagonal(dmat, np.inf)
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
                
                geom_edge_index.append([i + node_offset, j + node_offset])
                geom_edge_attr.append(feat)
        
        if geom_edge_index:
            batch_geom_edges_src.append(torch.tensor(geom_edge_index, dtype=torch.long).t())
            batch_geom_attr_src.append(torch.tensor(np.array(geom_edge_attr), dtype=torch.float32))
        else:
            batch_geom_edges_src.append(torch.zeros(2, 0, dtype=torch.long))
            batch_geom_attr_src.append(torch.zeros(0, 8, dtype=torch.float32))
        
        # Text edges
        text_relations = graph_data.get('edges_text', [])
        text_edge_index = []
        
        for r in text_relations:
            s = id2idx.get(str(r.get('subject', '')))
            o = id2idx.get(str(r.get('object', '')))
            
            if s is not None and o is not None:
                text_edge_index.append([s + node_offset, o + node_offset])
        
        if text_edge_index:
            batch_text_edges_src.append(torch.tensor(text_edge_index, dtype=torch.long).t())
            batch_text_attr_src.append(torch.ones(len(text_edge_index), 1, dtype=torch.long))
        else:
            batch_text_edges_src.append(torch.zeros(2, 0, dtype=torch.long))
            batch_text_attr_src.append(torch.zeros(0, 1, dtype=torch.long))
        
        # Batch indices
        batch_indices_src.extend([len(batch_node_feats_src) - 1] * N)
        node_offset += N
    
    # Concatenate
    batch = {
        "node_feats_src": torch.cat(batch_node_feats_src, dim=0).to(device),
        "geom_edges_src": torch.cat(batch_geom_edges_src, dim=1).to(device),
        "geom_attr_src": torch.cat(batch_geom_attr_src, dim=0).to(device),
        "text_edges_src": torch.cat(batch_text_edges_src, dim=1).to(device),
        "text_attr_src": torch.cat(batch_text_attr_src, dim=0).to(device),
        "src_batch": torch.tensor(batch_indices_src, dtype=torch.long).to(device),
        "batch_size": len(graphs)
    }
    
    return batch


def train_epoch(model, data_dir, scene_files, room_mapping, epoch, optimizer, criterion, batch_size=16):
    """
    Train for one epoch with supervised contrastive loss.
    """
    model.train()
    
    # Shuffle scenes
    random.shuffle(scene_files)
    
    # Create batches
    batches = [scene_files[i:i+batch_size] for i in range(0, len(scene_files), batch_size)]
    
    epoch_loss = 0
    pos_sims = []
    neg_sims = []
    
    for batch_files in tqdm(batches, desc=f"Epoch {epoch}"):
        if len(batch_files) < 2:
            continue
        
        # Load graphs
        graphs = []
        room_ids = []
        
        for filename in batch_files:
            filepath = os.path.join(data_dir, filename)
            graph_data = load_scene_graph(filepath)
            graphs.append(graph_data)
            
            # Get room ID
            scene_id = graph_data['scene_id']
            room_id = room_mapping[scene_id]
            room_ids.append(room_id)
        
        # Convert to batch
        batch = scene_graph_to_batch(graphs, device)
        room_ids = torch.tensor(room_ids, dtype=torch.long).to(device)
        
        # Forward pass
        embeddings = model.encode_src(batch)  # [batch_size, 128]
        
        # Compute loss
        loss = criterion(embeddings, room_ids)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Track similarities
        with torch.no_grad():
            emb_norm = F.normalize(embeddings, dim=1)
            sim_matrix = emb_norm @ emb_norm.T
            
            for i in range(len(room_ids)):
                for j in range(i+1, len(room_ids)):
                    sim = sim_matrix[i, j].item()
                    if room_ids[i] == room_ids[j]:
                        pos_sims.append(sim)
                    else:
                        neg_sims.append(sim)
    
    avg_loss = epoch_loss / len(batches)
    
    # Print stats
    print(f"\n{'='*70}")
    print(f"Epoch {epoch} Results")
    print(f"{'='*70}")
    print(f"Loss: {avg_loss:.4f}")
    
    if pos_sims:
        print(f"Positive pairs: {np.mean(pos_sims):.4f} ± {np.std(pos_sims):.4f}")
    if neg_sims:
        print(f"Negative pairs: {np.mean(neg_sims):.4f} ± {np.std(neg_sims):.4f}")
    if pos_sims and neg_sims:
        separation = np.mean(pos_sims) - np.mean(neg_sims)
        print(f"Separation: {separation:.4f}")
        
        if separation > 0.6:
            print("✓✓✓ EXCELLENT SEPARATION!")
        elif separation > 0.4:
            print("✓✓ GOOD separation")
        elif separation > 0.2:
            print("✓ OK separation")
        else:
            print("✗ Poor separation - keep training")
    
    return avg_loss, np.mean(pos_sims) if pos_sims else 0, np.mean(neg_sims) if neg_sims else 0


def main(args):
    print(f"{'='*70}")
    print("Training with Supervised Contrastive Loss")
    print(f"{'='*70}\n")
    
    # Load data
    all_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json') and f != 'metadata.json']
    
    # Create room mapping
    room_to_idx = {}
    idx_counter = 0
    
    for filename in all_files:
        filepath = os.path.join(args.data_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        room = data['scene_id']
        if room not in room_to_idx:
            room_to_idx[room] = idx_counter
            idx_counter += 1
    
    print(f"Total files: {len(all_files)}")
    print(f"Unique rooms: {len(room_to_idx)}\n")
    
    # Train/val split
    random.shuffle(all_files)
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files\n")
    
    # Initialize model
    dummy_clip = torch.randn(50, 512) * 0.1
    model = DualSceneAligner(
        node_input_dim=518,
        relation_dim=512,
        hidden_dim=128,
        rel_clip_matrix=dummy_clip.to(device),
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss and optimizer
    criterion = SupervisedContrastiveLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    best_separation = 0
    
    for epoch in range(1, args.epochs + 1):
        loss, pos_sim, neg_sim = train_epoch(
            model, args.data_dir, train_files, room_to_idx,
            epoch, optimizer, criterion, args.batch_size
        )
        
        separation = pos_sim - neg_sim
        
        # Save best
        if separation > best_separation:
            best_separation = separation
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'separation': separation,
                'pos_sim': pos_sim,
                'neg_sim': neg_sim,
            }, os.path.join(args.output_dir, 'model_best_contrastive.pth'))
            print(f"✓ Saved best model (separation: {separation:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='combined_dataset_clip')
    parser.add_argument('--output_dir', type=str, default='checkpoints_contrastive')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)