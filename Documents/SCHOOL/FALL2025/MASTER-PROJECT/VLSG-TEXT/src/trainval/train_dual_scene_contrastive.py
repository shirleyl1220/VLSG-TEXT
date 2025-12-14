import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


# ===============================================================
# Custom Collate Function for Graph Data
# ===============================================================

def collate_graph_batch(batch_list):
    """
    Collate variable-sized graphs into a single batch.
    Creates disjoint graphs by offsetting node indices.
    """

    batch_size = len(batch_list)

    node_feats_src_list = []
    geom_edges_src_list = []
    geom_attr_src_list  = []
    text_edges_src_list = []
    text_attr_src_list  = []

    node_feats_ref_list = []
    geom_edges_ref_list = []
    geom_attr_ref_list  = []
    text_edges_ref_list = []
    text_attr_ref_list  = []

    src_batch_idx = []
    ref_batch_idx = []

    src_node_offset = 0
    ref_node_offset = 0

    for i, sample in enumerate(batch_list):
        # === SOURCE GRAPH ===
        n_src = sample["node_feats_src"].size(0)
        node_feats_src_list.append(sample["node_feats_src"])

        geom_edges = sample["geom_edges_src"]
        if geom_edges.size(1) > 0:
            geom_edges = geom_edges + src_node_offset
        geom_edges_src_list.append(geom_edges)
        geom_attr_src_list.append(sample["geom_attr_src"])

        text_edges = sample["text_edges_src"]
        if text_edges.size(1) > 0:
            text_edges = text_edges + src_node_offset
        text_edges_src_list.append(text_edges)
        text_attr_src_list.append(sample["text_attr_src"])

        src_batch_idx.extend([i] * n_src)
        src_node_offset += n_src

        # === REFERENCE GRAPH ===
        n_ref = sample["node_feats_ref"].size(0)
        node_feats_ref_list.append(sample["node_feats_ref"])

        geom_edges = sample["geom_edges_ref"]
        if geom_edges.size(1) > 0:
            geom_edges = geom_edges + ref_node_offset
        geom_edges_ref_list.append(geom_edges)
        geom_attr_ref_list.append(sample["geom_attr_ref"])

        text_edges = sample["text_edges_ref"]
        if text_edges.size(1) > 0:
            text_edges = text_edges + ref_node_offset
        text_edges_ref_list.append(text_edges)
        text_attr_ref_list.append(sample["text_attr_ref"])

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


# ===============================================================
# Anti-Collapse Losses
# ===============================================================

def variance_loss(z, eps=1e-4):
    """
    Encourages each embedding dimension to have std >= 1.0.
    Prevents collapse to a low-variance manifold.
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_loss(z):
    """
    Penalizes redundancy between embedding dimensions.
    Ensures different dimensions encode different information.
    """
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (z.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).mean()


def norm_loss(z):
    """
    Encourages embeddings to live on the unit sphere.
    Prevents them from shrinking or exploding.
    """
    norms = z.norm(dim=1)
    return ((norms - 1.0) ** 2).mean()


# ===============================================================
# Updated Training Script
# ===============================================================

def train(args):
    import os
    from src.datasets.dual_scene_graph_dataset import DualSceneGraphDataset
    from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Dataset ----------------
    dataset = DualSceneGraphDataset(args.dataset_dir)

    batch_size = min(args.batch_size, len(dataset))
    if batch_size < 2:
        batch_size = 2
        print("⚠ batch_size too small for contrastive learning, forcing = 2")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_graph_batch,
    )

    print(f"Using effective batch_size = {batch_size}")

    # ---------------- Model ----------------
    model = DualSceneAligner(
        node_input_dim=518,            # 3 + 3 + 512
        relation_dim=512,              # CLIP text dim
        hidden_dim=128,                # GAT hidden size
        rel_clip_matrix=dataset.rel_clip_matrix.to(device),   # (num_rel, 512)
        dropout=0.0
    ).to(device)

    # ---------------- Loss ----------------
    class InfoNCELoss(nn.Module):
        def __init__(self, temp=0.07):
            super().__init__()
            self.temp = temp

        def forward(self, src, ref):
            src = F.normalize(src, dim=-1)
            ref = F.normalize(ref, dim=-1)
            logits = src @ ref.T / self.temp
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = (F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)) / 2
            return loss

    info_nce = InfoNCELoss()

    # Anti-collapse multipliers (tune later)
    λ_var  = 10.0
    λ_cov  = 1.0
    λ_norm = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader)
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")
    global_step = 0

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for epoch in range(args.epochs):
        epoch_loss = 0

        for batch in dataloader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            out = model(batch)
            src, ref = out["src_emb"], out["ref_emb"]

            # ---- MAIN LOSS ----
            loss_nce = info_nce(src, ref)

            # ---- ANTI-COLLAPSE ----
            loss_v = variance_loss(src) + variance_loss(ref)
            loss_c = covariance_loss(src) + covariance_loss(ref)
            loss_n = norm_loss(src) + norm_loss(ref)

            loss = (
                loss_nce
                + λ_var * loss_v
                + λ_cov * loss_c
                + λ_norm * loss_n
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            # ===== Debug =====
            if global_step % args.log_every == 0:
                src_std = src.std(dim=0).mean().item()
                ref_std = ref.std(dim=0).mean().item()
                diag_cos = F.cosine_similarity(src, ref).mean().item()
                print(f"\n[Epoch {epoch}] Step {global_step}")
                print(f"  Loss = {loss.item():.4f}")
                print(f"  • STD(src) = {src_std:.4f}, STD(ref) = {ref_std:.4f}")
                print(f"  • Cos diag = {diag_cos:.4f}")

                if src_std < 0.1 or ref_std < 0.1:
                    print("  ⚠️ Low variance → collapsing risk!")
                if diag_cos > 0.93:
                    print("  ⚠️ High similarity → collapsing risk!")

            # ---- SAVE BEST ----
            if loss.item() < best_loss and loss.item() > 0.0001:
                best_loss = loss.item()
                save_path = f"{args.save_dir}/model_best.pth"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "loss": best_loss
                }, save_path)
                print(f"  💾 Saved best model → {save_path}")

            global_step += 1

        print(f"\n====== Epoch {epoch} Finished | Avg Loss = {epoch_loss/len(dataloader):.4f} ======\n")

    print("\n🎉 Training complete.\n")


# ===============================================================
# Main
# ===============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)

    args = parser.parse_args()
    train(args)