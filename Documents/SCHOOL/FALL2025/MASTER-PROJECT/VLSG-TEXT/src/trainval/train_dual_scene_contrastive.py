
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import sys
import datetime
import argparse

from src.datasets.dual_scene_graph_dataset import DualSceneGraphDataset
from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner
import torch.nn as nn


# ============================================================
# Logger
# ============================================================

class Logger:
    def __init__(self, save_dir, filename="train_log.txt"):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(save_dir, f"{timestamp}_{filename}")
        self.log_file = open(self.log_path, "w")
        self.console = sys.__stdout__

    def write(self, msg):
        self.console.write(msg)
        self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ============================================================
# Collate Function (Fixed)
# ============================================================

def collate_graph_batch(batch_list):
    """
    Collate function for batching variable-sized graphs.
    """
    batch_size = len(batch_list)

    # Initialize lists
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
        # === SOURCE ===
        n_src = sample["node_feats_src"].size(0)
        node_feats_src_list.append(sample["node_feats_src"])

        # Geom edges (offset indices)
        g_edges = sample["geom_edges_src"]
        if g_edges.size(1) > 0:
            g_edges = g_edges + src_node_offset
        geom_edges_src_list.append(g_edges)
        geom_attr_src_list.append(sample["geom_attr_src"])

        # Text edges (offset indices)
        t_edges = sample["text_edges_src"]
        if t_edges.size(1) > 0:
            t_edges = t_edges + src_node_offset
        text_edges_src_list.append(t_edges)
        text_attr_src_list.append(sample["text_attr_src"].view(-1, 1))

        src_batch_idx.extend([i] * n_src)
        src_node_offset += n_src

        # === REFERENCE ===
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


# ============================================================
# VICReg Loss (Strong Anti-Collapse)
# ============================================================

class VICRegLoss(nn.Module):
    """
    VICReg loss with strong variance regularization.
    """
    def __init__(
        self,
        temperature=0.05,
        lambda_inv=1.0,
        lambda_var=100.0,  # High variance weight
        lambda_cov=10.0,
        var_eps=1e-4
    ):
        super().__init__()
        self.temp = temperature
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.var_eps = var_eps

    def invariance_loss(self, z1, z2):
        """InfoNCE contrastive loss"""
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)
        
        logits = z1 @ z2.T / self.temp
        labels = torch.arange(logits.size(0), device=logits.device)
        
        loss_i2j = F.cross_entropy(logits, labels)
        loss_j2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2j + loss_j2i) / 2.0

    def variance_loss(self, z):
        """Maintain variance >= 1.0 per dimension"""
        std = torch.sqrt(z.var(dim=0) + self.var_eps)
        # Penalize std < 1.0
        violation = F.relu(1.0 - std)
        return violation.mean()

    def covariance_loss(self, z):
        """Decorrelate features"""
        N, D = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (N - 1)
        
        # Off-diagonal elements should be zero
        off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
        off_diag = cov[off_diag_mask]
        
        return off_diag.pow(2).mean()

    def forward(self, z1, z2):
        inv_loss = self.invariance_loss(z1, z2)
        
        var_loss_1 = self.variance_loss(z1)
        var_loss_2 = self.variance_loss(z2)
        var_loss = (var_loss_1 + var_loss_2) / 2.0
        
        cov_loss_1 = self.covariance_loss(z1)
        cov_loss_2 = self.covariance_loss(z2)
        cov_loss = (cov_loss_1 + cov_loss_2) / 2.0
        
        total_loss = (
            self.lambda_inv * inv_loss +
            self.lambda_var * var_loss +
            self.lambda_cov * cov_loss
        )
        
        return total_loss, {
            'inv': inv_loss.item(),
            'var': var_loss.item(),
            'cov': cov_loss.item(),
            'total': total_loss.item()
        }


# ============================================================
# Training Function
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== Dataset =====
    print("\n" + "="*70)
    print("Loading Dataset")
    print("="*70)
    
    dataset = DualSceneGraphDataset(
        dataset_dir=args.dataset_dir,
        metadata_path=args.metadata_path,
        generate_text_edges=True,      # Generate missing text edges
        use_pure_geometric=True,       # Use geometric features (not CLIP)
        augment_ratio=0.1
    )

    batch_size = max(2, min(args.batch_size, len(dataset)))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_graph_batch,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")

    # ===== Model =====
    print("\n" + "="*70)
    print("Initializing Model")
    print("="*70)
    
    # Build dummy CLIP matrix for relations (we're not really using it)
    num_relations = len(dataset.rel2id)
    dummy_clip_matrix = torch.randn(num_relations, 512) * 0.1  # Small random
    num_relations = max(dataset.rel2id.values()) + 1  # Use max ID + 1
    dummy_clip_matrix = torch.randn(num_relations, 512) * 0.1
    print(f"Created embedding for {num_relations} relations (IDs 0-{num_relations-1})")
    
    model = DualSceneAligner(
        node_input_dim=518,
        relation_dim=512,
        hidden_dim=128,
        rel_clip_matrix=dummy_clip_matrix.to(device),
        dropout=0.1  # Add dropout to prevent overfitting
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ===== Loss & Optimizer =====
    print("\n" + "="*70)
    print("Setting up Training")
    print("="*70)
    
    loss_fn = VICRegLoss(
        temperature=0.05,
        lambda_inv=1.0,
        lambda_var=100.0,  # Strong variance regularization
        lambda_cov=10.0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Warmup + Cosine schedule
    def lr_schedule(step):
        warmup_steps = 50
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (args.epochs * len(dataloader) - warmup_steps)
        return 0.5 * (1 + np.cos(progress * np.pi))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ===== Logging =====
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(args.save_dir)
    sys.stdout = logger

    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Log every: {args.log_every} steps")
    print(f"Checkpoints: {args.save_dir}")

    # ===== Training Loop =====
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")

    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_losses = {'total': 0, 'inv': 0, 'var': 0, 'cov': 0}
        num_batches = 0

        for batch in dataloader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward
            out = model(batch)
            src, ref = out["src_emb"], out["ref_emb"]

            # Loss
            total_loss, loss_dict = loss_fn(src, ref)

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            # Accumulate
            epoch_loss += total_loss.item()
            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]
            num_batches += 1

            # ===== Logging =====
            if global_step % args.log_every == 0:
                with torch.no_grad():
                    # Statistics
                    src_var = src.var(dim=0).mean().item()
                    ref_var = ref.var(dim=0).mean().item()
                    
                    src_std = src.std(dim=0).mean().item()
                    ref_std = ref.std(dim=0).mean().item()
                    
                    src_norm = src.norm(dim=-1).mean().item()
                    ref_norm = ref.norm(dim=-1).mean().item()
                    
                    cos_sim = F.cosine_similarity(src, ref, dim=-1).mean().item()
                    
                    # Covariance
                    src_centered = src - src.mean(dim=0)
                    ref_centered = ref - ref.mean(dim=0)
                    src_cov = (src_centered.T @ src_centered).abs().mean().item()
                    ref_cov = (ref_centered.T @ ref_centered).abs().mean().item()

                print(f"\n[Epoch {epoch}] Step {global_step}")
                print(f"  Total Loss       = {loss_dict['total']:.4f}")
                print(f"  InfoNCE          = {loss_dict['inv']:.4f}")
                print(f"  Variance Loss    = {loss_dict['var']:.4f}")
                print(f"  Covariance Loss  = {loss_dict['cov']:.4f}")
                print(f"  ---")
                print(f"  Var(src, ref)    = {src_var:.4f}, {ref_var:.4f}")
                print(f"  STD(src, ref)    = {src_std:.4f}, {ref_std:.4f}")
                print(f"  Norm(src, ref)   = {src_norm:.4f}, {ref_norm:.4f}")
                print(f"  Cov(src, ref)    = {src_cov:.4f}, {ref_cov:.4f}")
                print(f"  Cosine Sim       = {cos_sim:.4f}")
                print(f"  Grad Norm        = {grad_norm:.4f}")
                print(f"  LR               = {scheduler.get_last_lr()[0]:.6f}")

                # Warnings
                if src_std < 0.1 or ref_std < 0.1:
                    print("  ⚠️  Low STD → Collapse risk!")
                if src_var < 0.5 or ref_var < 0.5:
                    print("  ⚠️  Low variance → Severe collapse!")
                if cos_sim > 0.93:
                    print("  ⚠️  High similarity → Embeddings too aligned!")
                
                # Good signs
                if src_std > 0.2 and ref_std > 0.2 and 0.3 < cos_sim < 0.7:
                    print("  ✓  Healthy training dynamics!")

            # ===== Save Best =====
            if total_loss.item() < best_loss and total_loss.item() > 1e-4:
                best_loss = total_loss.item()
                checkpoint = {
                    "step": global_step,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": best_loss
                }
                torch.save(checkpoint, f"{args.save_dir}/model_best.pth")
                print(f"  💾 Saved best model (loss={best_loss:.4f})")

            global_step += 1

        # ===== Epoch Summary =====
        avg_loss = epoch_loss / num_batches
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Summary")
        print(f"{'='*70}")
        print(f"  Avg Total Loss = {epoch_losses['total']:.4f}")
        print(f"  Avg InfoNCE    = {epoch_losses['inv']:.4f}")
        print(f"  Avg Var Loss   = {epoch_losses['var']:.4f}")
        print(f"  Avg Cov Loss   = {epoch_losses['cov']:.4f}")
        
        with torch.no_grad():
            print(f"  Final SRC: mean={src.mean().item():.4f}, std={src.std().item():.4f}")
            print(f"  Final REF: mean={ref.mean().item():.4f}, std={ref.std().item():.4f}")
        
        print(f"{'='*70}\n")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final checkpoint: {args.save_dir}/model_best.pth")
    
    logger.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual Scene Aligner")

    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Directory containing scene graph JSONs")
    parser.add_argument("--metadata_path", type=str, required=True,
                       help="Path to sequence.json metadata")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size (will be capped by dataset size)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4,
                       help="Learning rate (lowered from 1e-3 to prevent collapse)")
    parser.add_argument("--log_every", type=int, default=10,
                       help="Log metrics every N steps")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Training Configuration")
    print("="*70)
    print(f"Dataset:     {args.dataset_dir}")
    print(f"Metadata:    {args.metadata_path}")
    print(f"Checkpoints: {args.save_dir}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Epochs:      {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("="*70 + "\n")

    train(args)