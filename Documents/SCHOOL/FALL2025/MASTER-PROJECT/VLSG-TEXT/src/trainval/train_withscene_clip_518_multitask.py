"""
Step-by-step multi-task training - MINIMAL VERSION
Start with just contrastive, then add overlap
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import argparse

from src.datasets.dual_scene_graph_dataset_518_v3 import DualSceneGraphDataset
from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


# ============================================================
# STEP 1: Simple Model (No Overlap Yet)
# ============================================================

class SimpleGraphMatcher(nn.Module):
    """Just GNN + Scene CLIP fusion. No overlap head."""
    def __init__(self, base_model, scene_clip_dim=512, hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        
        # Simple fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(base_model.hidden_dim + scene_clip_dim),  # ← Input norm
            nn.Linear(base_model.hidden_dim + scene_clip_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased for better regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)  # ← Output norm
        )
    
    def forward(self, batch, scene_clip_src, scene_clip_ref):
        # GNN
        out = self.base_model(batch)
        gnn_src = out["src_emb"]
        gnn_ref = out["ref_emb"]
        
        print(f"  [DEBUG] GNN output shape: {gnn_src.shape}")
        print(f"  [DEBUG] GNN src mean: {gnn_src.mean().item():.4f}, std: {gnn_src.std().item():.4f}")
        
        # Fusion
        src_combined = torch.cat([gnn_src, scene_clip_src], dim=-1)
        ref_combined = torch.cat([gnn_ref, scene_clip_ref], dim=-1)
        
        src_emb = self.fusion(src_combined)
        ref_emb = self.fusion(ref_combined)
        
        print(f"  [DEBUG] After fusion - src mean: {src_emb.mean().item():.4f}, std: {src_emb.std().item():.4f}")
        
        return {
            "src_emb": src_emb,
            "ref_emb": ref_emb
        }


# ============================================================
# STEP 2: Simple Contrastive Loss
# ============================================================

class SimpleContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):  # Lowered from 0.5
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        embeddings: [2B, D]
        labels: [2B] where matching labels = positive pairs
        """
        print(f"  [DEBUG] Loss input - embeddings shape: {embeddings.shape}")
        print(f"  [DEBUG] Loss input - labels: {labels[:10].tolist()}")
        
        # Normalize
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        
        # Similarity
        sim_matrix = embeddings @ embeddings.T / self.temperature
        
        print(f"  [DEBUG] Sim matrix - min: {sim_matrix.min().item():.3f}, max: {sim_matrix.max().item():.3f}")
        
        # Positive mask
        labels = labels.view(-1, 1)
        mask_positive = (labels == labels.T).float()
        mask_positive.fill_diagonal_(0)
        
        print(f"  [DEBUG] Positive pairs in batch: {mask_positive.sum().item() / 2:.0f}")
        
        # Negative mask
        mask_negative = 1 - mask_positive
        mask_negative.fill_diagonal_(0)
        
        print(f"  [DEBUG] Negative pairs in batch: {mask_negative.sum().item() / 2:.0f}")
        
        # Loss
        exp_sim = torch.exp(sim_matrix)
        num_positives = mask_positive.sum(dim=1)
        
        loss = torch.zeros(1, device=embeddings.device, requires_grad=True)
        valid_samples = 0
        
        for i in range(embeddings.size(0)):
            if num_positives[i] == 0:
                continue
            
            pos_sim = (exp_sim[i] * mask_positive[i]).sum()
            all_sim = (exp_sim[i] * (mask_positive[i] + mask_negative[i])).sum()
            
            loss = loss + (-torch.log(pos_sim / (all_sim + 1e-8)))
            valid_samples += 1
        
        final_loss = loss / max(valid_samples, 1)
        loss_value = final_loss.item() if isinstance(final_loss, torch.Tensor) else final_loss
        print(f"  [DEBUG] Final loss: {loss_value:.4f}")
        
        return final_loss


# ============================================================
# STEP 3: Collate Function
# ============================================================

def collate_fn(batch_list):
    """Simplified collate."""
    batch_size = len(batch_list)
    
    print(f"\n[DEBUG COLLATE] Processing {batch_size} samples")

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
    
    scene_clip_src_list = []
    scene_clip_ref_list = []
    is_positive_list = []

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
        
        # Metadata
        scene_clip_src_list.append(sample['scene_clip_src'])
        scene_clip_ref_list.append(sample['scene_clip_ref'])
        is_positive_list.append(sample["is_positive"])
        
        print(f"  Sample {i}: is_positive={sample['is_positive']}")

    batch = {
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
        
        "scene_clip_src": torch.stack(scene_clip_src_list),
        "scene_clip_ref": torch.stack(scene_clip_ref_list),
        "is_positive": torch.tensor(is_positive_list, dtype=torch.bool),
    }
    
    print(f"[DEBUG COLLATE] is_positive distribution: {batch['is_positive'].sum().item()}/{batch_size} positive")
    
    return batch


# ============================================================
# STEP 4: Training Loop with DETAILED DEBUGGING
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Dataset
    dataset = DualSceneGraphDataset(
        dataset_dir=args.dataset_dir,
        metadata_path=args.metadata_path,
        negative_ratio=0.5
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Model
    num_relations = max(dataset.rel2id.values()) + 1
    
    rel_embeddings = nn.Embedding(num_relations, 64).to(device)
    nn.init.normal_(rel_embeddings.weight, mean=0, std=0.1)
    
    base_model = DualSceneAligner(
        node_input_dim=518,
        relation_dim=64,
        hidden_dim=256,
        rel_clip_matrix=rel_embeddings.weight,
        dropout=0.0
    ).to(device)
    
    model = SimpleGraphMatcher(
        base_model=base_model,
        scene_clip_dim=512,
        hidden_dim=256
    ).to(device)
    
    print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Loss & Optimizer
    loss_fn = SimpleContrastiveLoss(temperature=0.07).to(device)  # Lower temp = sharper gradients
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(rel_embeddings.parameters()),
        lr=args.lr * 0.5,  # Reduce initial LR by half
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Warmup + Cosine decay scheduler
    total_steps = args.epochs * len(dataloader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))  # Cosine decay
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Training
    print("="*70)
    print("STEP 1: TRAINING WITH CONTRASTIVE LOSS ONLY")
    print("="*70)
    print("Goal: Get separation > 0.2\n")
    
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n{'='*70}")
            print(f"[Epoch {epoch}] Step {global_step}")
            print(f"{'='*70}")
            
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            batch_size = batch['batch_size']
            is_positive = batch['is_positive']
            
            print(f"[DEBUG] Batch size: {batch_size}")
            print(f"[DEBUG] is_positive: {is_positive.tolist()}")
            
            # Forward
            print("\n[DEBUG] Forward pass...")
            out = model(
                batch,
                scene_clip_src=batch['scene_clip_src'],
                scene_clip_ref=batch['scene_clip_ref']
            )
            
            src_emb = out['src_emb']
            ref_emb = out['ref_emb']
            
            # Create labels from is_positive
            print("\n[DEBUG] Creating labels...")
            all_embeddings = torch.cat([src_emb, ref_emb], dim=0)
            
            src_labels = torch.arange(batch_size, device=device)
            ref_labels = torch.arange(batch_size, device=device)
            
            # Negative pairs get different labels
            ref_labels[~is_positive] = ref_labels[~is_positive] + batch_size
            
            labels = torch.cat([src_labels, ref_labels], dim=0)
            
            print(f"[DEBUG] src_labels: {src_labels.tolist()}")
            print(f"[DEBUG] ref_labels: {ref_labels.tolist()}")
            print(f"[DEBUG] combined labels: {labels.tolist()}")
            
            # Loss
            print("\n[DEBUG] Computing loss...")
            loss = loss_fn(all_embeddings, labels)
            
            # Backward
            print("\n[DEBUG] Backward pass...")
            optimizer.zero_grad()
            
            # Only backprop if loss requires grad
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                print(f"[DEBUG] Grad norm: {grad_norm:.4f}")
            else:
                print(f"[DEBUG] Loss has no gradients (likely all negatives in batch), skipping backward pass")
            
            optimizer.step()
            scheduler.step()
            
            # Metrics
            with torch.no_grad():
                src_norm = F.normalize(src_emb, dim=-1)
                ref_norm = F.normalize(ref_emb, dim=-1)
                
                cross_sim = src_norm @ ref_norm.T
                
                pos_mask = is_positive
                neg_mask = ~is_positive
                
                pos_sim = cross_sim.diag()[pos_mask].mean().item() if pos_mask.sum() > 0 else 0.0
                neg_sim = cross_sim.diag()[neg_mask].mean().item() if neg_mask.sum() > 0 else 0.0
                separation = pos_sim - neg_sim
            
            print(f"\n{'='*70}")
            print(f"RESULTS:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Pos similarity: {pos_sim:.3f}")
            print(f"  Neg similarity: {neg_sim:.3f}")
            print(f"  Separation: {separation:.3f}")
            
            if separation > 0.2:
                print(f"  ✅ GOOD! Separation > 0.2")
            elif separation > 0:
                print(f"  ⚠️  Positive but low")
            else:
                print(f"  ❌ NEGATIVE - something wrong!")
            
            print(f"{'='*70}\n")
            
            global_step += 1
            
            # Stop after 5 steps for debugging
            if global_step % 10 == 0:
                print(f"\n[Epoch {epoch}] Step {global_step}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Separation: {separation:.3f}")

            # Save checkpoints every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"/Users/shirley/Documents/SCHOOL/FALL2025/MASTERSPROJECT/VLSG-TEXT/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/checkpoints/checkpoints_multitask/epoch_{epoch}.pth")
                
                print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    train(args)