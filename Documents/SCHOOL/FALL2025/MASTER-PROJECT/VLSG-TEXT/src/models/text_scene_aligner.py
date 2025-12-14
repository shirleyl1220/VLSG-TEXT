import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from src.models.sgaligner.src.aligner.sg_aligner import MultiModalSingleScanEncoder
from src.models.patch_SGI_aligner import Mlps
from model_utils import TransformerEncoder


# ---------------------------------------------------------------
# Device selection (Apple Silicon → MPS, else CUDA, else CPU)
# ---------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------
# Text Node Encoder (CLIP text tower)
# ---------------------------------------------------------------
class TextNodeEncoder(nn.Module):
    def __init__(self, model_name, out_dim):
        super().__init__()

        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name).to(self.device)

        hidden_dim = self.text_model.config.hidden_size
        self.proj = nn.Linear(hidden_dim, out_dim).to(self.device)

    def forward(self, node_labels_batch):
        """
        node_labels_batch: list of list[str], batch of text node labels
        return: padded tensor [B, N_text_max, D]
        """
        batch_embs = []
        max_nodes = 0

        for labels in node_labels_batch:
            enc = self.tokenizer(
                labels, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            out = self.text_model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :]  # [num_nodes, hidden_dim]
            proj = self.proj(cls_emb)                 # [num_nodes, D]

            batch_embs.append(proj)
            max_nodes = max(max_nodes, proj.size(0))

        B = len(batch_embs)
        D = batch_embs[0].size(-1)
        padded = torch.zeros(B, max_nodes, D, device=self.device)

        for i, emb in enumerate(batch_embs):
            padded[i, :emb.size(0), :] = emb

        return padded  # [B, N, D]


# ---------------------------------------------------------------
# Text Graph Encoder (TransformerEncoder using Option B, CLS)
# ---------------------------------------------------------------
class TextGraphTransformer(nn.Module):
    def __init__(self, dim, num_layers=4, num_heads=4):
        super().__init__()
        # Using your TransformerEncoder with map_in, CLS, map_out
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model_in=dim,
            d_model_inner=dim,
            d_model_out=dim,
            num_heads=num_heads,
        )

    def forward(self, x):
        """
        x: [B, N, D]
        return: CLS embedding [B, D]
        """
        cls_emb = self.encoder.forward_cls(x)
        return cls_emb


# ---------------------------------------------------------------
# Scene-Level Text-Scene Aligner
# ---------------------------------------------------------------
class TextSceneAligner(nn.Module):
    def __init__(
        self,
        text_model_name="openai/clip-vit-base-patch32",
        emb_dim=256,
        sgaligner_modules=['point', 'gat', 'rel', 'attr'],
        rel_dim=41,
        attr_dim=164,
        dropout=0.0,
    ):
        super().__init__()
        self.device = get_device()

        # ------------------------------
        # TEXT ENCODER
        # ------------------------------
        self.text_node_encoder = TextNodeEncoder(
            model_name=text_model_name, out_dim=emb_dim
        )
        self.text_graph_encoder = TextGraphTransformer(
            dim=emb_dim, num_layers=4, num_heads=4
        )

        # ------------------------------
        # SCENE GRAPH ENCODER (yours)
        # ------------------------------
        self.sg_encoder = MultiModalSingleScanEncoder(
            modules=sgaligner_modules,
            rel_dim=rel_dim,
            attr_dim=attr_dim,
            dropout=dropout
        ).to(self.device)

        # ------------------------------
        # Object Embedding Encoder (yours)
        # ------------------------------
        # Use same as patch_SGI_aligner
        self.obj_embedding_encoder = Mlps(
            in_features=emb_dim,
            hidden_features=[],
            out_features=emb_dim,
            drop=dropout
        ).to(self.device)

        # Final pooling activation
        self.post_pool_act = nn.ReLU()

    # -----------------------------------------------------------
    # Scene pooling: mean pooling over object embeddings
    # -----------------------------------------------------------
    def pool_scene(self, node_feats):
        """
        node_feats: [B, N, D]
        return: [B, D]
        """
        pooled = node_feats.mean(dim=1)
        return self.post_pool_act(pooled)

    # -----------------------------------------------------------
    # Main forward pass
    # -----------------------------------------------------------
    def forward(self, data_dict):
        """
        Expected keys in data_dict:
            text_node_labels : list[list[str]]
            tot_obj_pts
            tot_bow_vec_object_attr_feats
            tot_bow_vec_object_edge_feats
            tot_rel_pose
            graph_per_obj_count
            graph_per_edge_count
            edges
            batch_size
        """
        B = data_dict["batch_size"]
        device = self.device

        # -------------------------
        # 1. TEXT SIDE
        # -------------------------
        text_nodes = self.text_node_encoder(data_dict["text_node_labels"])
        text_embed = self.text_graph_encoder(text_nodes)  # CLS
        text_embed = F.normalize(text_embed, dim=-1)

        # -------------------------
        # 2. SCENE SIDE
        # -------------------------
        sg_out = self.sg_encoder(data_dict)       # sg_out["joint"] shape: [N_total_obj, D]
        joint = sg_out["joint"]

        # reshape to [B, N_obj, D]
        N_total = joint.size(0)
        assert N_total % B == 0, "Mismatch between objects and batch size"
        N_obj = N_total // B
        scene_node_embeds = joint.view(B, N_obj, -1)

        # Apply object embedding encoder (per-object)
        scene_node_embeds = self.obj_embedding_encoder(scene_node_embeds)

        # L2 normalize each node embedding
        scene_node_embeds = F.normalize(scene_node_embeds, dim=-1)

        # -------------------------
        # 3. SCENE POOLING
        # -------------------------
        scene_embed = self.pool_scene(scene_node_embeds)    # [B, D]
        scene_embed = F.normalize(scene_embed, dim=-1)

        # -------------------------
        # 4. OUTPUT
        # -------------------------
        return {
            "text_embed": text_embed,                  # [B, D]
            "scene_embed": scene_embed,                # [B, D]
            "scene_node_embeds": scene_node_embeds     # [B, N_obj, D]
        }

    # -----------------------------------------------------------
    # Move model to correct device
    # -----------------------------------------------------------
    def to(self, device):
        super().to(device)
        self.text_node_encoder.to(device)
        self.text_graph_encoder.to(device)
        self.sg_encoder.to(device)
        self.obj_embedding_encoder.to(device)
        return self