import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# Scene–Text Contrastive Loss (CLIP-style)
# -------------------------------------------------------------
class SceneTextContrastiveLoss(nn.Module):
    """
    Symmetric InfoNCE loss:
        L = ( CE(text -> scene) + CE(scene -> text) ) / 2
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_embed, scene_embed):
        """
        text_embed:  [B, D]   already L2 normalized
        scene_embed: [B, D]   already L2 normalized
        """

        B = text_embed.size(0)
        device = text_embed.device

        # similarity matrix = cosine similarity (because normalized)
        logits = torch.matmul(text_embed, scene_embed.T) / self.temperature

        # target indices: 0..B-1
        targets = torch.arange(B, device=device)

        # CE(text -> scene)
        loss_ts = F.cross_entropy(logits, targets)

        # CE(scene -> text)
        loss_st = F.cross_entropy(logits.T, targets)

        loss = (loss_ts + loss_st) / 2

        return {
            "loss": loss,
            "loss_text_to_scene": loss_ts.detach(),
            "loss_scene_to_text": loss_st.detach(),
        }


# -------------------------------------------------------------
# Retrieval evaluation metrics (Recall@K)
# -------------------------------------------------------------
class SceneTextRetrievalMetrics(nn.Module):
    """
    Given text embeddings and scene embeddings:
        - rank all scenes for each text
        - compute R@1, R@3, R@5
    """

    def __init__(self):
        super().__init__()

    def forward(self, text_embed, scene_embed):
        """
        text_embed:  [B, D]
        scene_embed: [B, D]
        """
        B = text_embed.size(0)
        device = text_embed.device

        # similarity matrix
        sim = torch.matmul(text_embed, scene_embed.T)  # [B, B]

        # ranked scene indices for each text
        # higher sim = better match
        sorted_idx = torch.argsort(sim, dim=1, descending=True)

        # ground truth is: text i should match scene i
        gt = torch.arange(B, device=device).view(-1, 1)

        R1 = 0
        R3 = 0
        R5 = 0

        for i in range(B):
            ranking = sorted_idx[i]  # [B]

            if gt[i].item() in ranking[:1]:
                R1 += 1
            if gt[i].item() in ranking[:3]:
                R3 += 1
            if gt[i].item() in ranking[:5]:
                R5 += 1

        R1 /= B
        R3 /= B
        R5 /= B

        return {
            "R@1": torch.tensor(R1, device=device, dtype=torch.float32),
            "R@3": torch.tensor(R3, device=device, dtype=torch.float32),
            "R@5": torch.tensor(R5, device=device, dtype=torch.float32),
        }


# -------------------------------------------------------------
# Combined Loss Interface (train + val)
# -------------------------------------------------------------
class SceneTextLossModule(nn.Module):
    """
    Wraps:
        - SceneTextContrastiveLoss (train)
        - SceneTextRetrievalMetrics (val)
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.contrastive = SceneTextContrastiveLoss(temperature)
        self.metrics = SceneTextRetrievalMetrics()

    def forward(self, out, mode="train"):
        """
        out:
            {
                "text_embed": [B, D],
                "scene_embed": [B, D],
            }

        mode:
            "train" → return only contrastive loss
            "val"   → return loss + R@1/3/5
        """

        text_embed = out["text_embed"]
        scene_embed = out["scene_embed"]

        result = self.contrastive(text_embed, scene_embed)

        if mode == "val":
            metric = self.metrics(text_embed, scene_embed)
            result.update(metric)

        return result