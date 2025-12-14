import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# =====================================================================
# 1. TEXT NODE ENCODER (CLIP / SIGLIP)
# =====================================================================

class TextNodeEncoder(nn.Module):
    """
    Encodes each text node label ('sofa', 'table') using a CLIP/SigLIP
    text encoder and projects it into the same embedding dimension as
    the scene graph encoder (typically 256).
    """
    def __init__(self, model_name: str, out_dim: int):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)

        hidden_dim = self.text_model.config.hidden_size
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, node_labels_batch):
        """
        node_labels_batch: list[list[str]] of length B
        Returns:
            Tensor: [B, N_max, out_dim]
        """

        device = self.proj.weight.device
        batch_embeds = []
        max_nodes = 0

        for labels in node_labels_batch:
            encoded = self.tokenizer(
                labels, return_tensors="pt",
                padding=True, truncation=True
            ).to(device)

            outputs = self.text_model(**encoded)
            cls_embeds = outputs.last_hidden_state[:, 0, :]  # [N, hidden]

            projected = self.proj(cls_embeds)                # [N, out_dim]
            batch_embeds.append(projected)
            max_nodes = max(max_nodes, projected.size(0))

        # pad into tensor [B, N_max, out_dim]
        B = len(batch_embeds)
        D = batch_embeds[0].size(1)
        padded = torch.zeros(B, max_nodes, D, device=device)

        for i, emb in enumerate(batch_embeds):
            padded[i, :emb.size(0), :] = emb

        return padded



# =====================================================================
# 2. TEXT GRAPH ENCODER (Transformer-based GNN)
# =====================================================================

class TextGraphEncoder(nn.Module):
    """
    Processes text-node embeddings through a Transformer encoder to add
    relation-aware features and contextual interactions between nodes.
    """
    def __init__(self, dim, hidden_dim=512, num_layers=2, num_heads=4):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



# =====================================================================
# 3. MAIN ALIGNER MODEL (Hybrid Model)
# =====================================================================

class TextNodeSGIEAligner(nn.Module):
    """
    Hybrid replacement for PatchSGIEAligner.

    - Uses CLIP/SigLIP + Transformer to encode text nodes.
    - Uses MultiModalSingleScanEncoder to encode scene graph nodes.
    - Outputs node embeddings for node-to-node contrastive loss.
    """

    def __init__(
        self,
        sg_encoder,           #  MultiModalSingleScanEncoder instance
        text_model_name: str, # e.g. "google/siglip-base-patch16-384"
        text_emb_dim: int,    # must match cfg.model.obj.embedding_dim (e.g., 256)
        gnn_hidden_dim=512,
        gnn_layers=2
    ):
        super().__init__()

        self.sg_encoder = sg_encoder

        # text node pipeline
        self.text_node_encoder = TextNodeEncoder(
            model_name=text_model_name,
            out_dim=text_emb_dim
        )

        self.text_graph_encoder = TextGraphEncoder(
            dim=text_emb_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers
        )

    def forward(self, data_dict):
        """
        Expected in data_dict:
        -----------------------------------
        'text_node_labels' : list[list[str]]
        'scene_nodes'      : input to sg_encoder
        'scene_edges'      : adjacency / graph edges
        'batch_size'       : int
        -----------------------------------

        Output:
        -----------------------------------
        {
            'text_node_embeds':  [B, N_text, D],
            'scene_node_embeds': [B, N_scene, D]
        }
        -----------------------------------
        """

        # 1. Encode text nodes
        text_feats = self.text_node_encoder(data_dict['text_node_labels'])
        text_feats = self.text_graph_encoder(text_feats)

        ### NOT DONE#####
        # 2. Encode scene graph nodes using MultiModalSingleScanEncoder
        # NOTE:sg_encoder expects a dict with:
        #  - tot_obj_pts
        #  - tot_bow_vec_object_attr_feats
        #  - tot_bow_vec_object_edge_feats
        #  - tot_rel_pose
        #  - graph_per_obj_count
        #  - graph_per_edge_count
        #  - edges
        #  - batch_size
        scene_feats_dict = self.sg_encoder(data_dict)
        scene_node_embeds = scene_feats_dict['joint']  # fused node embeddings

        # reshape into [B, N_scene, D]
        B = data_dict['batch_size']
        total_nodes = scene_node_embeds.size(0)
        assert total_nodes % B == 0, "Uneven node count across batch"

        N = total_nodes // B
        scene_node_embeds = scene_node_embeds.view(B, N, -1)

        return {
            'text_node_embeds': text_feats,
            'scene_node_embeds': scene_node_embeds
        }