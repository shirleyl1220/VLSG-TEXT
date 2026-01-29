"""
Training with Scene-Level CLIP + Subgraph Augmentation

Key improvements:
1. Uses scene-level CLIP embeddings (room-type discrimination)
2. Subgraph augmentation during training (simulates ScanScribe queries)
3. Contrastive loss for matching
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import sys
import datetime
import argparse
import json
from sklearn.cluster import DBSCAN

from src.datasets.dual_scene_graph_dataset_1030 import DualSceneGraphDataset
from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math


# ============================================================
# Scene-Level Model (NEW!)
# ============================================================

class DualSceneAlignerWithSceneCLIP(nn.Module):
    """
    Extends DualSceneAligner to use scene-level CLIP embeddings.
    
    Combines:
    - GNN output (graph structure + node features)
    - Scene-level CLIP (room type discrimination)
    """
    def __init__(self, base_model, scene_clip_dim=512, hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        self.scene_clip_dim = scene_clip_dim
        
        # Project concatenated features to final dimension
        # GNN outputs hidden_dim, scene CLIP is 512, concat = 768
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + scene_clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, batch, scene_clip_src=None, scene_clip_ref=None):
        """
        Args:
            batch: Standard graph batch
            scene_clip_src: [batch_size, 512] scene-level CLIP for source
            scene_clip_ref: [batch_size, 512] scene-level CLIP for reference
        """
        # Get GNN embeddings
        out = self.base_model(batch)
        gnn_src = out["src_emb"]  # [batch_size, hidden_dim]
        gnn_ref = out["ref_emb"]
        
        # If scene CLIP provided, fuse it
        if scene_clip_src is not None and scene_clip_ref is not None:
            # Concatenate GNN + scene CLIP
            src_combined = torch.cat([gnn_src, scene_clip_src], dim=-1)
            ref_combined = torch.cat([gnn_ref, scene_clip_ref], dim=-1)
            
            # Project to final dimension
            src_emb = self.fusion(src_combined)
            ref_emb = self.fusion(ref_combined)
        else:
            # Fallback: use GNN only
            src_emb = gnn_src
            ref_emb = gnn_ref
        
        return {
            "src_emb": src_emb,
            "ref_emb": ref_emb
        }


# ============================================================
# Subgraph Augmentation (NEW!)
# ============================================================

def create_subgraph_from_indices(scene_data, keep_node_ids):
    """
    Create a subgraph by keeping only specified node IDs.
    
    Args:
        scene_data: Full scene graph dict
        keep_node_ids: List of node IDs to keep (as strings)
    
    Returns:
        Subgraph scene data dict
    """
    if not keep_node_ids:
        return None
    
    # Filter nodes
    subgraph_nodes = {}
    for nid in keep_node_ids:
        if nid in scene_data['nodes']:
            subgraph_nodes[nid] = scene_data['nodes'][nid]
    
    # Filter edges (keep only edges between kept nodes)
    keep_set = set(keep_node_ids)
    subgraph_edges = []
    for edge in scene_data.get('edges_text', []):
        if edge['subject'] in keep_set and edge['object'] in keep_set:
            subgraph_edges.append(edge)
    
    # Create subgraph
    subgraph = {
        'scene_id': scene_data['scene_id'] + '_subgraph',
        'nodes': subgraph_nodes,
        'edges_text': subgraph_edges
    }
    
    # Preserve scene-level CLIP if present
    if 'scene_clip_emb' in scene_data:
        subgraph['scene_clip_emb'] = scene_data['scene_clip_emb']
    if 'scene_description' in scene_data:
        subgraph['scene_description'] = scene_data['scene_description']
    
    return subgraph


def get_matching_subgraph_simple(scene1_data, scene2_data, subgraph_ratio=0.6):
    """
    Simplified subgraph matching using DBSCAN clustering on CLIP embeddings.
    
    Creates subgraphs that share similar objects (like ScanScribe does).
    
    Args:
        scene1_data: First scene graph dict
        scene2_data: Second scene graph dict
        subgraph_ratio: Fraction of nodes to keep (0.5-0.7 typical for ScanScribe)
    
    Returns:
        subgraph1, subgraph2 (or None if no overlap)
    """
    # Collect node CLIP embeddings from both scenes
    all_embeddings = []
    all_node_ids = []
    all_graph_idx = []  # 0 for scene1, 1 for scene2
    
    for nid, node in scene1_data['nodes'].items():
        if 'clip_text_emb' in node:
            all_embeddings.append(node['clip_text_emb'])
            all_node_ids.append(nid)
            all_graph_idx.append(0)
    
    for nid, node in scene2_data['nodes'].items():
        if 'clip_text_emb' in node:
            all_embeddings.append(node['clip_text_emb'])
            all_node_ids.append(nid)
            all_graph_idx.append(1)
    
    if len(all_embeddings) < 2:
        return None, None
    
    # Cluster using DBSCAN (groups similar objects)
    all_embeddings = np.array(all_embeddings)
    clustering = DBSCAN(eps=0.5, min_samples=1, metric='cosine').fit(all_embeddings)
    
    # Find clusters that contain nodes from BOTH graphs
    clusters = {}
    for i, cluster_id in enumerate(clustering.labels_):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(i)
    
    # Collect overlapping nodes
    scene1_keep = []
    scene2_keep = []
    
    for cluster_id, indices in clusters.items():
        graph_indices = [all_graph_idx[i] for i in indices]
        
        # Only keep clusters with nodes from BOTH scenes
        if 0 in graph_indices and 1 in graph_indices:
            for i in indices:
                if all_graph_idx[i] == 0:
                    scene1_keep.append(all_node_ids[i])
                else:
                    scene2_keep.append(all_node_ids[i])
    
    # If no overlap, return None
    if not scene1_keep or not scene2_keep:
        return None, None
    
    # Randomly sample to get desired subgraph size
    num_keep_1 = max(3, int(len(scene1_keep) * subgraph_ratio))
    num_keep_2 = max(3, int(len(scene2_keep) * subgraph_ratio))
    
    if len(scene1_keep) > num_keep_1:
        scene1_keep = list(np.random.choice(scene1_keep, num_keep_1, replace=False))
    if len(scene2_keep) > num_keep_2:
        scene2_keep = list(np.random.choice(scene2_keep, num_keep_2, replace=False))
    
    # Create subgraphs
    subgraph1 = create_subgraph_from_indices(scene1_data, scene1_keep)
    subgraph2 = create_subgraph_from_indices(scene2_data, scene2_keep)
    
    return subgraph1, subgraph2


# ============================================================
# Supervised Contrastive Loss
# ============================================================

class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss for room matching."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [N, D] embeddings
            labels: [N] room labels (same room = same label)
        """
        # Normalize
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        
        # Compute similarity matrix
        sim_matrix = embeddings @ embeddings.T / self.temperature
        
        # Create positive mask (same room)
        labels = labels.view(-1, 1)
        mask_positive = (labels == labels.T).float()
        
        # Remove self-similarity
        mask_positive.fill_diagonal_(0)
        
        # Negative mask
        mask_negative = 1 - mask_positive
        mask_negative.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        
        # For each anchor
        loss = 0
        num_positives = mask_positive.sum(dim=1)
        
        for i in range(embeddings.size(0)):
            if num_positives[i] == 0:
                continue
            
            # Positive similarities
            pos_sim = (exp_sim[i] * mask_positive[i]).sum()
            
            # All similarities (positive + negative)
            all_sim = (exp_sim[i] * (mask_positive[i] + mask_negative[i])).sum()
            
            # Contrastive loss
            loss += -torch.log(pos_sim / (all_sim + 1e-8))
        
        return loss / (num_positives > 0).sum()


# ============================================================
# Dataset with Scene CLIP
# ============================================================

class SceneGraphDatasetWithSceneCLIP(DualSceneGraphDataset):
    """Extended dataset that loads scene-level CLIP embeddings."""
    
    def __getitem__(self, idx):
        # Get base data
        sample = super().__getitem__(idx)
        
        # Load scene CLIP embeddings
        src_file = self.file_pairs[idx]['src']
        ref_file = self.file_pairs[idx]['ref']
        
        with open(src_file) as f:
            src_data = json.load(f)
        with open(ref_file) as f:
            ref_data = json.load(f)
        
        # Extract scene CLIP (if available)
        scene_clip_src = torch.tensor(src_data.get('scene_clip_emb', np.zeros(512)), dtype=torch.float32)
        scene_clip_ref = torch.tensor(ref_data.get('scene_clip_emb', np.zeros(512)), dtype=torch.float32)
        
        sample['scene_clip_src'] = scene_clip_src
        sample['scene_clip_ref'] = scene_clip_ref
        sample['room_label_src'] = self.room_labels.get(src_data['scene_id'], -1)
        sample['room_label_ref'] = self.room_labels.get(ref_data['scene_id'], -1)
        
        return sample


# ============================================================
# Collate with Scene CLIP
# ============================================================

def collate_with_scene_clip(batch_list):
    """Collate function that handles scene CLIP embeddings."""
    # Standard graph batching
    batch_size = len(batch_list)
    
    # ... (same as before for node features, edges, etc.)
    node_feats_src_list = []
    geom_edges_src_list = []
    geom_attr_src_list = []
    text_edges_src_list = []
    text_attr_src_list = []
    
    node_feats_ref_list = []
    geom_edges_ref_list = []
    geom_attr_ref_list = []
    text_edges_ref_list = []
    text_attr_ref_list = []
    
    src_batch_idx = []
    ref_batch_idx = []
    
    src_node_offset = 0
    ref_node_offset = 0
    
    # NEW: Collect scene CLIP embeddings
    scene_clip_src_list = []
    scene_clip_ref_list = []
    room_labels_src = []
    room_labels_ref = []
    
    for i, sample in enumerate(batch_list):
        # SOURCE
        n_src = sample["node_feats_src"].size(0)
        node_feats_src_list.append(sample["node_feats_src"])
        
        g_edges = sample["geom_edges_src"]
        if g_edges.size(1) > 0:
            g_edges = g_edges + src_node_offset
        geom_edges_src_list.append(g_edges)
        geom_attr_src_list.append(sample["geom_attr_src"])
        
        t_edges = sample["text_edges_src"]
        if t_edges.size(1) > 0:
            t_edges = t_edges + src_node_offset
        text_edges_src_list.append(t_edges)
        text_attr_src_list.append(sample["text_attr_src"].view(-1, 1))
        
        src_batch_idx.extend([i] * n_src)
        src_node_offset += n_src
        
        # REFERENCE
        n_ref = sample["node_feats_ref"].size(0)
        node_feats_ref_list.append(sample["node_feats_ref"])
        
        g_edges = sample["geom_edges_ref"]
        if g_edges.size(1) > 0:
            g_edges = g_edges + ref_node_offset
        geom_edges_ref_list.append(g_edges)
        geom_attr_ref_list.append(sample["geom_attr_ref"])
        
        t_edges = sample["text_edges_ref"]
        if t_edges.size(1) > 0:
            t_edges = t_edges + ref_node_offset
        text_edges_ref_list.append(t_edges)
        text_attr_ref_list.append(sample["text_attr_ref"].view(-1, 1))
        
        ref_batch_idx.extend([i] * n_ref)
        ref_node_offset += n_ref
        
        # Scene CLIP
        scene_clip_src_list.append(sample['scene_clip_src'])
        scene_clip_ref_list.append(sample['scene_clip_ref'])
        room_labels_src.append(sample['room_label_src'])
        room_labels_ref.append(sample['room_label_ref'])
    
    return {
        "node_feats_src": torch.cat(node_feats_src_list, dim=0),
        "geom_edges_src": torch.cat(geom_edges_src_list, dim=1),
        "geom_attr_src": torch.cat(geom_attr_src_list, dim=0),
        "text_edges_src": torch.cat(text_edges_src_list, dim=1),
        "text_attr_src": torch.cat(text_attr_src_list, dim=0),
        
        "node_feats_ref": torch.cat(node_feats_ref_list, dim=0),
        "geom_edges_ref": torch.cat(geom_edges_ref_list, dim=1),
        "geom_attr_ref": torch.cat(geom_attr_ref_list, dim=0),
        "text_edges_ref": torch.cat(text_edges_ref_list, dim=1),
        "text_attr_ref": torch.cat(text_attr_ref_list, dim=0),
        
        "src_batch": torch.tensor(src_batch_idx, dtype=torch.long),
        "ref_batch": torch.tensor(ref_batch_idx, dtype=torch.long),
        "batch_size": batch_size,
        
        # Scene CLIP
        "scene_clip_src": torch.stack(scene_clip_src_list),
        "scene_clip_ref": torch.stack(scene_clip_ref_list),
        "room_labels": torch.tensor(room_labels_src + room_labels_ref, dtype=torch.long)
    }


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset - scene CLIP is already in node features (1030 dims)!
    dataset = DualSceneGraphDataset(
        dataset_dir=args.dataset_dir,
        metadata_path=args.metadata_path,
        augment_ratio=0.0
    )
    
    print(f"\n{'='*70}")
    print("DATASET DEBUG")
    print(f"{'='*70}")
    print(f"Dataset size: {len(dataset)}")
    
    # Check first sample
    sample = dataset[0]
    print(f"\nFirst sample inspection:")
    print(f"  Node features shape: {sample['node_feats_src'].shape}")
    print(f"  Expected: (N, 1030) where N = num nodes")
    
    # Check feature breakdown
    first_node = sample['node_feats_src'][0]
    print(f"\nFirst node feature breakdown:")
    print(f"  Centroid (0-2):     {first_node[:3]}")
    print(f"  Color (3-5):        {first_node[3:6]}")
    print(f"  Node CLIP (6-517):  mean={first_node[6:518].mean():.4f}, std={first_node[6:518].std():.4f}, norm={first_node[6:518].norm():.4f}")
    print(f"  Scene CLIP (518-1029): mean={first_node[518:].mean():.4f}, std={first_node[518:].std():.4f}, norm={first_node[518:].norm():.4f}")
    
    # Check if scene CLIP is same for all nodes
    if sample['node_feats_src'].shape[0] > 1:
        scene_clip_0 = sample['node_feats_src'][0, 518:]
        scene_clip_1 = sample['node_feats_src'][1, 518:]
        scene_clip_diff = (scene_clip_0 - scene_clip_1).abs().max().item()
        print(f"\nScene CLIP consistency check:")
        print(f"  Max diff between node 0 and node 1: {scene_clip_diff:.6f}")
        if scene_clip_diff < 1e-6:
            print(f"  ✓ Scene CLIP is same for all nodes (as expected!)")
        else:
            print(f"  ❌ WARNING: Scene CLIP differs between nodes!")
    
    # Check diversity across scenes
    if len(dataset) >= 2:
        sample2 = dataset[1]
        scene_clip_scene0 = sample['node_feats_src'][0, 518:]
        scene_clip_scene1 = sample2['node_feats_src'][0, 518:]
        scene_diversity = (scene_clip_scene0 - scene_clip_scene1).abs().mean().item()
        print(f"\nScene-to-scene diversity:")
        print(f"  Avg diff between scene 0 and scene 1: {scene_diversity:.4f}")
        if scene_diversity > 0.01:
            print(f"  ✓ Scenes have different scene CLIP embeddings!")
        else:
            print(f"  ❌ WARNING: Scenes have very similar scene CLIP!")
    
    print(f"{'='*70}\n")
    
    # Standard collate function
    def collate_graph_batch(batch_list):
        """Collate function for batching variable-sized graphs."""
        batch_size = len(batch_list)

        node_feats_src_list = []
        geom_edges_src_list = []
        geom_attr_src_list = []
        text_edges_src_list = []
        text_attr_src_list = []

        node_feats_ref_list = []
        geom_edges_ref_list = []
        geom_attr_ref_list = []
        text_edges_ref_list = []
        text_attr_ref_list = []

        src_batch_idx = []
        ref_batch_idx = []

        src_node_offset = 0
        ref_node_offset = 0

        for i, sample in enumerate(batch_list):
            # SOURCE
            n_src = sample["node_feats_src"].size(0)
            node_feats_src_list.append(sample["node_feats_src"])

            g_edges = sample["geom_edges_src"]
            if g_edges.size(1) > 0:
                g_edges = g_edges + src_node_offset
            geom_edges_src_list.append(g_edges)
            geom_attr_src_list.append(sample["geom_attr_src"])

            t_edges = sample["text_edges_src"]
            if t_edges.size(1) > 0:
                t_edges = t_edges + src_node_offset
            text_edges_src_list.append(t_edges)
            text_attr_src_list.append(sample["text_attr_src"].view(-1, 1))

            src_batch_idx.extend([i] * n_src)
            src_node_offset += n_src

            # REFERENCE
            n_ref = sample["node_feats_ref"].size(0)
            node_feats_ref_list.append(sample["node_feats_ref"])

            g_edges = sample["geom_edges_ref"]
            if g_edges.size(1) > 0:
                g_edges = g_edges + ref_node_offset
            geom_edges_ref_list.append(g_edges)
            geom_attr_ref_list.append(sample["geom_attr_ref"])

            t_edges = sample["text_edges_ref"]
            if t_edges.size(1) > 0:
                t_edges = t_edges + ref_node_offset
            text_edges_ref_list.append(t_edges)
            text_attr_ref_list.append(sample["text_attr_ref"].view(-1, 1))

            ref_batch_idx.extend([i] * n_ref)
            ref_node_offset += n_ref

        return {
            "node_feats_src": torch.cat(node_feats_src_list, dim=0),
            "geom_edges_src": torch.cat(geom_edges_src_list, dim=1),
            "geom_attr_src": torch.cat(geom_attr_src_list, dim=0),
            "text_edges_src": torch.cat(text_edges_src_list, dim=1),
            "text_attr_src": torch.cat(text_attr_src_list, dim=0),

            "node_feats_ref": torch.cat(node_feats_ref_list, dim=0),
            "geom_edges_ref": torch.cat(geom_edges_ref_list, dim=1),
            "geom_attr_ref": torch.cat(geom_attr_ref_list, dim=0),
            "text_edges_ref": torch.cat(text_edges_ref_list, dim=1),
            "text_attr_ref": torch.cat(text_attr_ref_list, dim=0),

            "src_batch": torch.tensor(src_batch_idx, dtype=torch.long),
            "ref_batch": torch.tensor(ref_batch_idx, dtype=torch.long),
            "batch_size": batch_size,
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_graph_batch
    )
    
    # Build model - scene CLIP already in features!
    num_relations = max(dataset.rel2id.values()) + 1
    dummy_clip_matrix = torch.randn(num_relations, 512) * 0.1
    
    model = DualSceneAligner(
        node_input_dim=1030,  # ← INCLUDES SCENE CLIP!
        relation_dim=512,
        hidden_dim=256,  # Output dimension
        rel_clip_matrix=dummy_clip_matrix.to(device),
        dropout=0.1
    ).to(device)
    
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    global_step = 0
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward - scene CLIP already in node features!
            out = model(batch)
            
            # Get embeddings
            src_emb = out['src_emb']
            ref_emb = out['ref_emb']
            
            # Combine for contrastive loss
            all_embeddings = torch.cat([src_emb, ref_emb], dim=0)
            
            # Create labels (assume consecutive pairs are from same room for now)
            # This is a simplified version - you may want to use actual room labels
            room_labels = torch.arange(args.batch_size, device=device).repeat(2)
            
            # Loss
            loss = loss_fn(all_embeddings, room_labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient norm
            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            optimizer.step()
            scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1 - step / (len(dataloader) * args.epochs))
            scheduler.step()
            
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Detailed logging every N steps
            if global_step :
                with torch.no_grad():
                    # Normalize for similarity computation
                    src_norm_emb = F.normalize(src_emb, dim=-1)
                    ref_norm_emb = F.normalize(ref_emb, dim=-1)
                    
                    # Positive/negative separation
                    pos_mask = torch.eye(args.batch_size, device=device).bool()
                    neg_mask = ~pos_mask
                    cross_sim = src_norm_emb @ ref_norm_emb.T
                    
                    pos_sim = cross_sim[pos_mask].mean().item()
                    neg_sim = cross_sim[neg_mask].mean().item() if neg_mask.sum() > 0 else 0.0
                    separation = pos_sim - neg_sim
                    
                    # Stats
                    src_var = src_emb.var(dim=0).mean().item()
                    ref_var = ref_emb.var(dim=0).mean().item()
                    src_std = src_emb.std(dim=0).mean().item()
                    ref_std = ref_emb.std(dim=0).mean().item()

                print(f"\n{'='*70}")
                print(f"[Epoch {epoch}] Step {global_step}/{len(dataloader)*(epoch+1)}")
                print(f"{'='*70}")
                print(f"  Loss = {loss.item():.4f}")
                print(f"  ---")
                print(f"  📊 VARIANCE & STD:")
                print(f"    Var(src, ref) = {src_var:.4f}, {ref_var:.4f}")
                print(f"    Std(src, ref) = {src_std:.4f}, {ref_std:.4f}")
                print(f"  ---")
                print(f"  🎯 CONTRASTIVE QUALITY:")
                print(f"    Pos pairs: {pos_sim:.3f} (want > 0.7)")
                print(f"    Neg pairs: {neg_sim:.3f} (want < 0.3)")
                print(f"    Separation: {separation:.3f} (want > 0.4)")
                
                if separation > 0.4:
                    print(f"    ⭐⭐ EXCELLENT!")
                elif separation > 0.3:
                    print(f"    ⭐ GOOD!")
                elif separation > 0.1:
                    print(f"    ⚠️  MODERATE")
                else:
                    print(f"    ❌ POOR - Check data!")
                
                if src_var < 0.5 or ref_var < 0.5:
                    print(f"  ⚠️  WARNING: Low variance → collapse risk!")
                
                print(f"  ---")
                print(f"  Grad norm = {grad_norm:.4f}")
                print(f"  LR = {scheduler.get_last_lr()[0]:.6f}")
                print(f"{'='*70}\n")
        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Summary")
        print(f"{'='*70}")
        print(f"  Avg Loss         = {avg_epoch_loss:.4f}")
        
        # Final batch statistics
        with torch.no_grad():
            print(f"  Final SRC: mean={src_emb.mean().item():.4f}, std={src_emb.std().item():.4f}")
            print(f"  Final REF: mean={ref_emb.mean().item():.4f}, std={ref_emb.std().item():.4f}")
            print(f"  ---")
            
            if batch['batch_size'] >= 2 and 'separation' in locals():
                print(f"  📊 Final Separation: {separation:.3f}")
                if separation > 0.5:
                    print(f"     ✓✓✓ EXCELLENT! Model learning very well!")
                elif separation > 0.4:
                    print(f"     ✓✓ Great progress this epoch!")
                elif separation > 0.3:
                    print(f"     ✓ Good improvement!")
                elif separation < 0.1 and epoch > 10:
                    print(f"     ⚠️  WARNING: No improvement after {epoch} epochs")
                    print(f"        Consider adjusting hyperparameters")
        
        print(f"{'='*70}\n")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss/len(dataloader)
            }, f"{args.save_dir}/model_epoch_{epoch}.pth")
    
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--save_dir", default="checkpoints_scene_clip")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    train(args)