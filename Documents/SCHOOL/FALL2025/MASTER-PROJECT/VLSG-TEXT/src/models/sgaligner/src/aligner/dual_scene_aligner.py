"""
Dual Scene Aligner (full CLIP support, stable dimensions, no collapse)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from src.models.sgaligner.src.aligner.networks.edge_gat import MultiGAT_Edge


# ============================================================
# 1. Relation Embedding using CLIP (already precomputed)
# ============================================================

class CLIPRelationEmbedding(nn.Module):
    """Lookup table for relation CLIP embeddings (E, 512)."""

    def __init__(self, clip_matrix):  
        super().__init__()
        # clip_matrix = tensor (num_relations, 512)
        self.register_buffer("clip_embs", clip_matrix)

    def forward(self, rel_ids):
        rel_ids = rel_ids.view(-1).long()
        return self.clip_embs[rel_ids]
    

# ============================================================
# 2. Attention pooling (batch-aware)
# ============================================================

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x, batch):
        """
        x: (N, D)
        batch: (N,) graph id for each node
        """
        w = self.att(x)                # (N,1)
        w = w - w.max()                # stability
        w = torch.exp(w)

        # Sum attention weights per graph
        denom = scatter_mean(w, batch, dim=0)[batch] * len(batch)

        pooled = scatter_mean(w * x / (denom + 1e-8), batch, dim=0)
        return pooled   # (num_graphs, D)
    

# ============================================================
# 3. Dual Scene Aligner (CLIP version)
# ============================================================
class DualSceneAligner(nn.Module):
    def __init__(
        self,
        node_input_dim=518,       # 3 + 3 + 512
        relation_dim=512,
        hidden_dim=128,

        rel_clip_matrix=None,    # (num_relations, 512)
        dropout=0.0
    ):
        super().__init__()

        self.rel_emb = CLIPRelationEmbedding(rel_clip_matrix)

        # -----------------------------------------------------
        # GEOMETRY GAT
        # -----------------------------------------------------
        self.gat_geom = MultiGAT_Edge(
            n_units=[hidden_dim, hidden_dim, hidden_dim],
            n_heads=[2, 2],
            edge_dim=8,
            dropout=dropout
        )

        # -----------------------------------------------------
        # TEXT GAT
        # -----------------------------------------------------
        self.gat_text = MultiGAT_Edge(
            n_units=[hidden_dim, hidden_dim, hidden_dim],
            n_heads=[2, 2],
            edge_dim=relation_dim,   # ← MUST MATCH CLIP
            dropout=dropout
        )

        # -----------------------------------------------------
        # Fusion layer (very important!)
        # -----------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # -----------------------------------------------------
        # Attention pooling
        # -----------------------------------------------------
        self.pool = AttentionPool(hidden_dim)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
)
        # -----------------------------------------------------
        # Final scene embedding
        # -----------------------------------------------------
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
        )

    # ---------------------------------------------------------
    # PER-SCENE ENCODING
    # ---------------------------------------------------------
    def encode_scene(self, node_feats, geom_edges, geom_attr,
                     text_edges, text_attr, batch):

        # Normalize CLIP NODE embeddings
        node_feats[:, 6:] = F.normalize(node_feats[:, 6:], p=2, dim=-1)
        node_feats = self.node_encoder(node_feats)
        # 1) Geometry
        g = self.gat_geom(
            node_feats,
            geom_edges,
            geom_attr,
            edge_weight_factor=0.2,
            skip_edges_first_layer=True
        )

        # 2) Text relation embeddings
        rel_ids = text_attr.squeeze(-1).long()    # (E,)
        rel = self.rel_emb(rel_ids)               # (E,512)
        # 3) Text GAT
        t = self.gat_text(
            g,
            text_edges,
            rel,
            edge_weight_factor=0.1,
            skip_edges_first_layer=True
        )

        # 4) Fusion at node level (THIS FIXES ALIGNMENT)
        fused = torch.cat([g, t], dim=-1)
        fused = self.fusion(fused)

        # 5) Pool to graph level
        pooled = self.pool(fused, batch)

        # 6) Final projection
        return self.final(pooled)

    # ---------------------------------------------------------
    def forward(self, batch):

        src = self.encode_scene(
            batch["node_feats_src"],
            batch["geom_edges_src"],
            batch["geom_attr_src"],
            batch["text_edges_src"],
            batch["text_attr_src"],
            batch["src_batch"]
        )

        ref = self.encode_scene(
            batch["node_feats_ref"],
            batch["geom_edges_ref"],
            batch["geom_attr_ref"],
            batch["text_edges_ref"],
            batch["text_attr_ref"],
            batch["ref_batch"]
        )

        return {"src_emb": src, "ref_emb": ref}