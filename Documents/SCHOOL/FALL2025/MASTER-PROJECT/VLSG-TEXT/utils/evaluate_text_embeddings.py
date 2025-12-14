import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


from datasets.dual_scene_graph_dataset import DualSceneGraphDataset 
from models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner
# Optional visualization
try:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    USE_VIS = True
except:
    USE_VIS = False


# ---------------------------------------------------------
# Pick device
# ---------------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
dataset_dir = "scene_graphs"
dataset = DualSceneGraphDataset(dataset_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

scene_files = dataset.scene_files
print(f"Loaded {len(scene_files)} scenes.")


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
ckpt_path = "checkpoints/model_step_200.pth"

model = DualSceneAligner(
    rel_vocab_size=len(dataset.rel2id)
).to(device)

print("Loading checkpoint:", ckpt_path)
state = torch.load(ckpt_path, map_location=device)

# FIX: extract actual weights
if "model" in state:
    print("Checkpoint contains wrapper dict → loading state['model']")
    state = state["model"]
else:
    print("Checkpoint is raw state_dict")

# ----- FIX RELATION EMBEDDING SIZE MISMATCH -----
if "rel_emb.emb.weight" in state:
    ckpt_rel = state["rel_emb.emb.weight"]
    cur_rel = model.rel_emb.emb.weight

    if ckpt_rel.size(0) >= cur_rel.size(0):
        print(f"Truncating relation embedding: ckpt {ckpt_rel.shape} -> current {cur_rel.shape}")
        state["rel_emb.emb.weight"] = ckpt_rel[:cur_rel.size(0)]
    else:
        raise ValueError("Checkpoint has fewer relations than current vocabulary!")
else:
    print("WARNING: No relation embedding found in checkpoint.")

# ------------------------------------------------------------
# Remove incompatible text-GAT weights (trained before fix)
# ------------------------------------------------------------
bad_keys = []
for k in list(state.keys()):
    if "gat_text.gat.layers" in k:
        bad_keys.append(k)

if bad_keys:
    print("⚠️ Removing incompatible text-GAT weights:")
    for k in bad_keys:
        print("   -", k)
        del state[k]

# Load non‑strict because we removed optimizer / wrapper keys
model.load_state_dict(state, strict=False)
model.eval()

print("Model loaded.")


# ---------------------------------------------------------
# Encode TEXT-ONLY embedding for a batch
# ---------------------------------------------------------
def encode_text_only(model, batch):

    node_feats = batch["node_feats_src"].squeeze(0).to(device)
    text_edges = batch["text_edges_src"].squeeze(0).to(device)
    text_attr  = batch["text_attr_src"].squeeze(0).to(device)

    # Geometry GAT first → produce 128‑dim node embeddings
    geom_edges = batch["geom_edges_src"].squeeze(0).to(device)
    geom_attr  = batch["geom_attr_src"].squeeze(0).to(device)

    g = model.gat_geom(node_feats, geom_edges, geom_attr)  # (N, 128)

    # Embed relations
    rel_emb = model.rel_emb(text_attr)  # (E, 32)

    # Text GAT using geometric features as input
    t = model.gat_text(g, text_edges, rel_emb)  # (N, 128)

    # Pool and project
    t_cls = t.mean(dim=0, keepdim=True)
    t_cls = model.text_cls(t_cls)

    print("Raw node features:", node_feats.shape)
    print("Geom GAT output:", g.shape)
    print("Text edge index:", text_edges.shape)
    print("Rel emb:", rel_emb.shape)
    print("Text GAT output:", t.shape)

    return t_cls.squeeze(0).detach().cpu()


# ---------------------------------------------------------
# Compute embeddings for all scenes
# ---------------------------------------------------------
text_embs = []

print("\nEncoding text embeddings...")
for batch in loader:
    emb = encode_text_only(model, batch)
    text_embs.append(emb)

text_embs = torch.stack(text_embs)  # (N, text_dim)
N = len(text_embs)

print(f"Text embedding matrix shape: {text_embs.shape}")


# ---------------------------------------------------------
# Compute cosine similarity matrix
# ---------------------------------------------------------
sim_matrix = F.cosine_similarity(
    text_embs[:, None, :],
    text_embs[None, :, :],
    dim=-1
)

print("\nCosine Similarity Matrix:")
print(sim_matrix)

# --------------------------------------------------------------
# DEBUGGING: Variance diagnostics & visualization
# --------------------------------------------------------------

emb = text_embs  # (N_scenes, D)
cosine_matrix = sim_matrix  # rename for consistency

# Scene-level variance (how spread-out each embedding is)
scene_var = emb.var(dim=1).detach().cpu().numpy()

# Global variance (how different dimensions behave across scenes)
global_var = emb.var(dim=0).mean().item()

print("\n========= VARIANCE DIAGNOSTICS =========")
print("Mean per-scene variance:", scene_var.mean())
print("Global embedding variance:", global_var)

# --------------------------------------------------------------
# 1. Histogram of per-scene variances
# --------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.hist(scene_var, bins=20)
plt.title("Histogram of Per-Scene Embedding Variance")
plt.xlabel("Variance")
plt.ylabel("Count")
plt.show()

# --------------------------------------------------------------
# 2. Histogram of cosine similarities
# --------------------------------------------------------------
cos = cosine_matrix.cpu().numpy()
upper = cos[np.triu_indices_from(cos, k=1)]  # off-diagonal similarities

plt.figure(figsize=(6,4))
plt.hist(upper, bins=30)
plt.title("Cosine Similarity Distribution (Across Scenes)")
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.show()

# --------------------------------------------------------------
# 3. PCA (2D) visualization
# --------------------------------------------------------------
pca = PCA(n_components=2)
pca_emb = pca.fit_transform(emb.cpu().numpy())

plt.figure(figsize=(6,6))
plt.scatter(pca_emb[:,0], pca_emb[:,1], alpha=0.7)
plt.title("Scene Embeddings (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# --------------------------------------------------------------
# 4. t-SNE visualization
# --------------------------------------------------------------
tsne = TSNE(n_components=2, perplexity=20, learning_rate=200)
tsne_emb = tsne.fit_transform(emb.cpu().numpy())

plt.figure(figsize=(6,6))
plt.scatter(tsne_emb[:,0], tsne_emb[:,1], alpha=0.7)
plt.title("Scene Embeddings (t-SNE 2D)")
plt.show()
# ---------------------------------------------------------
# Retrieval test
# ---------------------------------------------------------
def retrieval_test(sim_matrix, scene_files):
    print("\n====== RETRIEVAL TEST ======")
    for i in range(len(scene_files)):
        sims = sim_matrix[i]
        top = torch.topk(sims, k=min(3, len(scene_files)))

        print(f"\nQuery Scene: {os.path.basename(scene_files[i])}")
        for score, idx in zip(top.values, top.indices):
            print(f"  → {os.path.basename(scene_files[idx])} | cos={score:.3f}")


retrieval_test(sim_matrix, scene_files)


# ---------------------------------------------------------
# PCA 2D Visualization (if sklearn/matplotlib available)
# ---------------------------------------------------------
if USE_VIS:
    print("\nPerforming PCA for visualization...")

    pca = PCA(n_components=2)
    emb2d = pca.fit_transform(text_embs)

    plt.figure(figsize=(6, 6))
    plt.scatter(emb2d[:, 0], emb2d[:, 1])

    for i, name in enumerate(scene_files):
        plt.text(
            emb2d[i, 0],
            emb2d[i, 1],
            os.path.basename(name),
            fontsize=8
        )

    plt.title("Text-only Scene Embeddings (PCA)")
    plt.tight_layout()
    plt.show()
else:
    print("\nSkipping PCA — sklearn or matplotlib not installed.")