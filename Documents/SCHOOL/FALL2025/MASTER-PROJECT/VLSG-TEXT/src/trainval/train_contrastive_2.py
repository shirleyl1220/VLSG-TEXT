"""
Training Script with Supervised Contrastive Loss
Fixed for your DualSceneAligner model
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

sys.path.append('.')
from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, room_labels):
        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = embeddings @ embeddings.T / self.temperature
        
        labels = room_labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        neg_mask = (labels != labels.T).float()
        pos_mask.fill_diagonal_(0)
        
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits)
        exp_negatives = exp_logits * neg_mask
        
        log_prob = logits - torch.log(exp_negatives.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-12)
        
        loss = -mean_log_prob_pos.mean()
        return loss


def scene_graph_to_batch(graphs, device):
    batch_node_feats = []
    batch_geom_edges = []
    batch_geom_attr = []
    batch_text_edges = []
    batch_text_attr = []
    batch_indices = []
    
    node_offset = 0
    
    for graph_data in graphs:
        nodes = graph_data['nodes']
        node_ids = list(nodes.keys())
        id2idx = {str(nid): i for i, nid in enumerate(node_ids)}
        
        node_feats = []
        for nid in node_ids:
            n = nodes[nid]
            centroid = np.array(n['centroid'], dtype=np.float32)
            color = np.array(n['mean_color'], dtype=np.float32) / 255.0
            clip_vec = np.array(n.get('clip_text_emb', np.zeros(512)), dtype=np.float32)
            feat = np.concatenate([centroid, color, clip_vec])
            node_feats.append(feat)
        
        node_feats = torch.tensor(np.array(node_feats), dtype=torch.float32)
        batch_node_feats.append(node_feats)
        
        centroids = np.array([nodes[nid]['centroid'] for nid in node_ids], dtype=float)
        N = len(node_ids)
        K = 5
        
        geom_edge_index = []
        geom_edge_attr = []
        
        if N > 1:
            dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
            np.fill_diagonal(dmat, np.inf)
            knn_idx = np.argsort(dmat, axis=1)[:, :min(K, N-1)]
            
            for i in range(N):
                ci = centroids[i]
                ri = nodes[node_ids[i]].get('radius', 0.4)
                
                for j in knn_idx[i]:
                    cj = centroids[j]
                    rj = nodes[node_ids[j]].get('radius', 0.4)
                    
                    vec = cj - ci
                    dist = float(np.linalg.norm(vec))
                    feat = np.array([vec[0], vec[1], vec[2], dist, ri, rj, 0.0, 0.0], dtype=np.float32)
                    
                    geom_edge_index.append([i + node_offset, j + node_offset])
                    geom_edge_attr.append(feat)
        
        if geom_edge_index:
            batch_geom_edges.append(torch.tensor(geom_edge_index, dtype=torch.long).t())
            batch_geom_attr.append(torch.tensor(np.array(geom_edge_attr), dtype=torch.float32))
        else:
            batch_geom_edges.append(torch.zeros(2, 0, dtype=torch.long))
            batch_geom_attr.append(torch.zeros(0, 8, dtype=torch.float32))
        
        text_relations = graph_data.get('edges_text', [])
        text_edge_index = []
        
        for r in text_relations:
            s = id2idx.get(str(r.get('subject', '')))
            o = id2idx.get(str(r.get('object', '')))
            
            if s is not None and o is not None:
                text_edge_index.append([s + node_offset, o + node_offset])
        
        if text_edge_index:
            batch_text_edges.append(torch.tensor(text_edge_index, dtype=torch.long).t())
            batch_text_attr.append(torch.ones(len(text_edge_index), 1, dtype=torch.long))
        else:
            batch_text_edges.append(torch.zeros(2, 0, dtype=torch.long))
            batch_text_attr.append(torch.zeros(0, 1, dtype=torch.long))
        
        batch_indices.extend([len(batch_node_feats) - 1] * N)
        node_offset += N
    
    node_feats_all = torch.cat(batch_node_feats, dim=0).to(device)
    geom_edges_all = torch.cat(batch_geom_edges, dim=1).to(device)
    geom_attr_all = torch.cat(batch_geom_attr, dim=0).to(device)
    text_edges_all = torch.cat(batch_text_edges, dim=1).to(device)
    text_attr_all = torch.cat(batch_text_attr, dim=0).to(device)
    batch_tensor = torch.tensor(batch_indices, dtype=torch.long).to(device)
    
    batch = {
        "node_feats_src": node_feats_all,
        "geom_edges_src": geom_edges_all,
        "geom_attr_src": geom_attr_all,
        "text_edges_src": text_edges_all,
        "text_attr_src": text_attr_all,
        "src_batch": batch_tensor,
        "node_feats_ref": node_feats_all,
        "geom_edges_ref": geom_edges_all,
        "geom_attr_ref": geom_attr_all,
        "text_edges_ref": text_edges_all,
        "text_attr_ref": text_attr_all,
        "ref_batch": batch_tensor,
        "batch_size": len(graphs)
    }
    
    return batch


def create_room_aware_batches(data_dir, scene_files, room_mapping, batch_size=16, samples_per_room=3):
    room_to_files = {}
    
    for filename in scene_files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        scene_id = data['scene_id']
        room_idx = room_mapping[scene_id]
        
        if room_idx not in room_to_files:
            room_to_files[room_idx] = []
        room_to_files[room_idx].append(filename)
    
    print(f"\nRoom-Aware Batch Sampling:")
    print(f"  Total rooms: {len(room_to_files)}")
    print(f"  Samples per room: {samples_per_room}")
    print(f"  Batch size: {batch_size}")
    
    rooms_per_batch = batch_size // samples_per_room
    print(f"  Rooms per batch: {rooms_per_batch}")
    
    batches = []
    room_ids = list(room_to_files.keys())
    random.shuffle(room_ids)
    
    for i in range(0, len(room_ids), rooms_per_batch):
        batch_rooms = room_ids[i:i+rooms_per_batch]
        batch_files = []
        
        for room_idx in batch_rooms:
            files = room_to_files[room_idx]
            
            if len(files) >= samples_per_room:
                sampled = random.sample(files, samples_per_room)
            else:
                sampled = files + random.choices(files, k=samples_per_room - len(files))
            
            batch_files.extend(sampled)
        
        if len(batch_files) >= 2:
            batches.append(batch_files)
    
    print(f"  Created {len(batches)} batches")
    print(f"  Avg batch size: {np.mean([len(b) for b in batches]):.1f}\n")
    
    return batches


def train_epoch(model, data_dir, scene_files, room_mapping, epoch, optimizer, criterion, 
                batch_size=16, samples_per_room=3):
    model.train()
    
    batches = create_room_aware_batches(data_dir, scene_files, room_mapping, 
                                        batch_size, samples_per_room)
    
    epoch_loss = 0
    pos_sims = []
    neg_sims = []
    
    for batch_files in tqdm(batches, desc=f"Epoch {epoch}"):
        if len(batch_files) < 2:
            continue
        
        graphs = []
        room_ids = []
        
        for filename in batch_files:
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as f:
                graph_data = json.load(f)
            graphs.append(graph_data)
            
            scene_id = graph_data['scene_id']
            room_id = room_mapping[scene_id]
            room_ids.append(room_id)
        
        batch = scene_graph_to_batch(graphs, device)
        room_ids = torch.tensor(room_ids, dtype=torch.long).to(device)
        
        if random.random() < 0.05:
            unique_rooms = torch.unique(room_ids)
            print(f"\n  Batch has {len(unique_rooms)} unique rooms, {len(room_ids)} total scans")
            for room in unique_rooms[:3]:
                count = (room_ids == room).sum().item()
                print(f"    Room {room.item()}: {count} scans")
        
        output = model(batch)
        embeddings = output["src_emb"]
        
        loss = criterion(embeddings, room_ids)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
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
    
    all_files = [f for f in os.listdir(args.data_dir) 
                 if f.endswith('.json') and f not in ['metadata.json', 'training_splits.json']]
    
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
    
    random.shuffle(all_files)
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files\n")
    
    dummy_clip = torch.randn(50, 512) * 0.1
    model = DualSceneAligner(
        node_input_dim=518,
        relation_dim=512,
        hidden_dim=128,
        rel_clip_matrix=dummy_clip.to(device),
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    criterion = SupervisedContrastiveLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_separation = 0
    
    print(f"\n{'='*70}")
    print("Training Strategy")
    print(f"{'='*70}")
    print(f"Batch size: {args.batch_size}")
    print(f"Samples per room: {args.samples_per_room}")
    print(f"Rooms per batch: {args.batch_size // args.samples_per_room}")
    print(f"\nThis ensures each batch has:")
    print(f"  - Multiple scans from same rooms (positives)")
    print(f"  - Multiple different rooms (negatives)")
    print(f"  - Better contrastive learning!")
    print(f"{'='*70}\n")
    
    for epoch in range(1, args.epochs + 1):
        loss, pos_sim, neg_sim = train_epoch(
            model, args.data_dir, train_files, room_to_idx,
            epoch, optimizer, criterion, args.batch_size, args.samples_per_room
        )
        
        separation = pos_sim - neg_sim
        
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
        
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='src/datasets/combined_dataset_clip')
    parser.add_argument('--output_dir', type=str, default='checkpoints_contrastive')
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--samples_per_room', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    
    args = parser.parse_args()
    
    if args.batch_size % args.samples_per_room != 0:
        print(f"WARNING: batch_size ({args.batch_size}) should be divisible by samples_per_room ({args.samples_per_room})")
        args.batch_size = (args.batch_size // args.samples_per_room) * args.samples_per_room
        print(f"Adjusting batch_size to {args.batch_size}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)