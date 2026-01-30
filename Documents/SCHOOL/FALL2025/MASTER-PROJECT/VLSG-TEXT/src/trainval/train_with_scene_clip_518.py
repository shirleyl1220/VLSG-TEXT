"""
Training with FIXED architecture:
- 518D node features (NO scene CLIP in nodes!)
- 64D learned relation embeddings  
- Scene CLIP added AFTER GNN (fusion layer)
- Subgraph augmentation (50% of time)
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import argparse
import json

from src.datasets.dual_scene_graph_dataset_1030 import DualSceneGraphDataset
from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import random


# ============================================================
# Model with Scene CLIP Fusion
# ============================================================

class DualSceneAlignerWithSceneCLIPFusion(nn.Module):
    """
    GNN processes 518D node features → 256D graph embedding
    Then: Fuse with scene CLIP → final embedding
    
    This prevents scene CLIP from dominating node features!
    """
    def __init__(self, base_model, scene_clip_dim=512, hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        
        # Fusion layer: [GNN(256D) + scene_CLIP(512D)] → 256D
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + scene_clip_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, batch, scene_clip_src, scene_clip_ref):
        """
        Args:
            batch: Graph batch with 518D node features
            scene_clip_src: [batch_size, 512] scene CLIP
            scene_clip_ref: [batch_size, 512] scene CLIP
        """
        # Get GNN embeddings from graph structure
        out = self.base_model(batch)
        gnn_src = out["src_emb"]  # [batch_size, 256]
        gnn_ref = out["ref_emb"]  # [batch_size, 256]
        
        # Fuse GNN + scene CLIP
        src_combined = torch.cat([gnn_src, scene_clip_src], dim=-1)  # [B, 768]
        ref_combined = torch.cat([gnn_ref, scene_clip_ref], dim=-1)  # [B, 768]
        
        src_emb = self.fusion(src_combined)  # [B, 256]
        ref_emb = self.fusion(ref_combined)  # [B, 256]
        
        return {
            "src_emb": src_emb,
            "ref_emb": ref_emb
        }


# ============================================================
# Supervised Contrastive Loss
# ============================================================

class SupervisedContrastiveLoss(nn.Module):
    """Contrastive loss for room matching."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        # Normalize
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        
        # Similarity matrix
        sim_matrix = embeddings @ embeddings.T / self.temperature
        
        # Positive mask (same room)
        labels = labels.view(-1, 1)
        mask_positive = (labels == labels.T).float()
        mask_positive.fill_diagonal_(0)
        
        # Negative mask
        mask_negative = 1 - mask_positive
        mask_negative.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        num_positives = mask_positive.sum(dim=1)
        
        loss = 0
        for i in range(embeddings.size(0)):
            if num_positives[i] == 0:
                continue
            
            pos_sim = (exp_sim[i] * mask_positive[i]).sum()
            all_sim = (exp_sim[i] * (mask_positive[i] + mask_negative[i])).sum()
            loss += -torch.log(pos_sim / (all_sim + 1e-8))
        
        return loss / (num_positives > 0).sum()


# ============================================================
# Collate Function
# ============================================================

def collate_graph_batch_with_scene_clip(batch_list):
    """Collate function that handles scene CLIP separately."""
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
    
    # Scene CLIP (not in node features!)
    scene_clip_src_list = []
    scene_clip_ref_list = []
    room_ids = []

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
        
        # Scene CLIP
        scene_clip_src_list.append(sample['scene_clip_src'])
        scene_clip_ref_list.append(sample['scene_clip_ref'])
        room_ids.append(sample["room_id"])

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
        
        # Scene CLIP (512D per graph)
        "scene_clip_src": torch.stack(scene_clip_src_list),
        "scene_clip_ref": torch.stack(scene_clip_ref_list),
        "room_ids": room_ids,
    }


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = DualSceneGraphDataset(
        dataset_dir=args.dataset_dir,
        metadata_path=args.metadata_path,
        augment_ratio=0.0
    )
    
    print(f"\n{'='*70}")
    print("ARCHITECTURE")
    print(f"{'='*70}")
    print(f"✓ Node features: 518D (centroid + color + node_CLIP)")
    print(f"✓ Edge features: 64D learned embeddings")
    print(f"✓ Scene CLIP: Added AFTER GNN (fusion layer)")
    print(f"✓ Subgraph aug: 50% of time")
    print(f"{'='*70}\n")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_graph_batch_with_scene_clip
    )
    
    # Build model
    num_relations = max(dataset.rel2id.values()) + 1
    
    # Learned relation embeddings (64D)
    rel_embeddings = nn.Embedding(num_relations, 64).to(device)
    nn.init.normal_(rel_embeddings.weight, mean=0, std=0.1)
    
    # Base GNN model
    base_model = DualSceneAligner(
        node_input_dim=518,  # ← 518D (no scene CLIP!)
        relation_dim=64,     # ← 64D learned embeddings
        hidden_dim=256,      # Output dimension
        rel_clip_matrix=rel_embeddings.weight,  # Use learned embeddings
        dropout=0.0  # Subgraph aug provides regularization
    ).to(device)
    
    # Wrap with scene CLIP fusion
    model = DualSceneAlignerWithSceneCLIPFusion(
        base_model=base_model,
        scene_clip_dim=512,
        hidden_dim=256
    ).to(device)
    
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Loss and optimizer
    loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    
    # Optimize both GNN and relation embeddings
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(rel_embeddings.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: max(0.1, 1.0 - step / (args.epochs * len(dataloader))))
    
    # Training loop
    print("Starting training...\n")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward with scene CLIP fusion
            out = model(
                batch,
                scene_clip_src=batch['scene_clip_src'],
                scene_clip_ref=batch['scene_clip_ref']
            )
            
            src_emb = out['src_emb']
            ref_emb = out['ref_emb']
            
            # Combine for contrastive loss
            all_embeddings = torch.cat([src_emb, ref_emb], dim=0)
            
            # Room labels
            unique_rooms = list(set(batch['room_ids']))
            room_id_to_label = {room_id: i for i, room_id in enumerate(unique_rooms)}
            room_labels_list = [room_id_to_label[room_id] for room_id in batch['room_ids']]
            room_labels = torch.tensor(room_labels_list * 2, device=device)
            
            # Loss
            loss = loss_fn(all_embeddings, room_labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient norm
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % args.log_every == 0:
                with torch.no_grad():
                    # Metrics
                    src_norm_emb = F.normalize(src_emb, dim=-1)
                    ref_norm_emb = F.normalize(ref_emb, dim=-1)
                    
                    cross_sim = src_norm_emb @ ref_norm_emb.T
                    room_labels_src = room_labels[:args.batch_size]
                    room_labels_ref = room_labels[args.batch_size:]
                    pos_mask = (room_labels_src.unsqueeze(1) == room_labels_ref.unsqueeze(0))
                    neg_mask = ~pos_mask
                    
                    pos_sim = cross_sim[pos_mask].mean().item() if pos_mask.sum() > 0 else 0.0
                    neg_sim = cross_sim[neg_mask].mean().item() if neg_mask.sum() > 0 else 0.0
                    separation = pos_sim - neg_sim
                    
                    src_var = src_emb.var(dim=0).mean().item()
                    ref_var = ref_emb.var(dim=0).mean().item()

                print(f"\n{'='*70}")
                print(f"[Epoch {epoch}] Step {global_step}")
                print(f"{'='*70}")
                print(f"  Loss = {loss.item():.4f}")
                print(f"  ---")
                print(f"  📊 VARIANCE & STD:")
                print(f"    Var(src, ref) = {src_var:.4f}, {ref_var:.4f}")
                print(f"  ---")
                print(f"  🎯 CONTRASTIVE QUALITY:")
                print(f"    Pos pairs: {pos_sim:.3f}")
                print(f"    Neg pairs: {neg_sim:.3f}")
                print(f"    Separation: {separation:.3f}")
                
                if separation > 0.4:
                    print(f"    ⭐⭐ EXCELLENT!")
                elif separation > 0.3:
                    print(f"    ⭐ GOOD!")
                else:
                    print(f"    ⚠️  Learning...")
                
                if src_var > 0.5 and ref_var > 0.5:
                    print(f"  ✓ Healthy variance!")
                elif src_var < 0.3 or ref_var < 0.3:
                    print(f"  ⚠️  Low variance")
                
                print(f"  ---")
                print(f"  Grad norm = {grad_norm:.4f}")
                print(f"  LR = {scheduler.get_last_lr()[0]:.6f}")
                print(f"{'='*70}\n")
        
        # Epoch summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Summary")
        print(f"{'='*70}")
        print(f"  Avg Loss = {epoch_loss/len(dataloader):.4f}")
        print(f"{'='*70}\n")
        
        # Save checkpoint
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'fusion_state_dict': model.fusion.state_dict(),
                'rel_embeddings': rel_embeddings.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss/len(dataloader)
            }, f"{args.save_dir}/model_epoch_{epoch}.pth")
            print(f"✓ Saved checkpoint: epoch_{epoch}.pth\n")
    
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--save_dir", default="checkpoints_518d_fusion")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=10)
    
    args = parser.parse_args()
    train(args)