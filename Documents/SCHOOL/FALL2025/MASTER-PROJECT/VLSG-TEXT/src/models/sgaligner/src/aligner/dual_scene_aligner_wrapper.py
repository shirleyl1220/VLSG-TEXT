"""
Wrapper for DualSceneAligner that adds matching probability prediction.

This wrapper adds an MLP matching head (like BigGNN's SceneText_MLP) 
to predict match probability from embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class DualSceneAlignerWithMatching(nn.Module):
    """
    Wrapper that adds matching probability prediction to DualSceneAligner.
    
    Compatible with BigGNN evaluation - outputs src_emb, ref_emb, and matching_prob.
    """
    
    def __init__(self, base_model, hidden_dim=128, use_cosine=False):
        """
        Args:
            base_model: Your trained DualSceneAligner model
            hidden_dim: Dimension of embeddings from base_model (default: 128)
            use_cosine: If True, use cosine similarity instead of MLP (no training needed)
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.use_cosine = use_cosine
        
        if not use_cosine:
            # MLP matching head (same architecture as BigGNN's SceneText_MLP)
            self.matching_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # Output [0, 1] probability
            )
    
    def forward(self, batch):
        """
        Forward pass - compatible with both training and evaluation.
        
        Args:
            batch: Dictionary with graph data (same format as DualSceneAligner)
        
        Returns:
            dict with keys:
                - "src_emb": Source graph embedding
                - "ref_emb": Reference graph embedding  
                - "matching_prob": Matching probability [0, 1]
        """
        # Get embeddings from base model
        out = self.base_model(batch)
        src_emb = out["src_emb"]
        ref_emb = out["ref_emb"]
        
        if self.use_cosine:
            # Use cosine similarity as matching probability
            cos_sim = F.cosine_similarity(src_emb, ref_emb, dim=-1)
            matching_prob = (cos_sim + 1) / 2  # Map [-1, 1] to [0, 1]
        else:
            # Use MLP to predict matching probability
            concat_emb = torch.cat([src_emb, ref_emb], dim=-1)
            matching_prob = self.matching_head(concat_emb).squeeze(-1)
        
        return {
            "src_emb": src_emb,
            "ref_emb": ref_emb,
            "matching_prob": matching_prob
        }
    
    def get_embeddings_only(self, batch):
        """
        Get just the embeddings without matching probability.
        Useful for visualization or other downstream tasks.
        """
        out = self.base_model(batch)
        return out["src_emb"], out["ref_emb"]


def load_model_with_matching(checkpoint_path, base_model_config, 
                             hidden_dim=128, use_cosine=False, device='cpu'):
    """
    Convenience function to load a trained DualSceneAligner and wrap it with matching head.
    
    Args:
        checkpoint_path: Path to saved checkpoint (.pth file)
        base_model_config: Dict with config to recreate DualSceneAligner
                          Should have: node_input_dim, relation_dim, hidden_dim, 
                                      rel_clip_matrix, dropout
        hidden_dim: Embedding dimension (should match base_model's output)
        use_cosine: If True, use cosine similarity (no MLP)
        device: Device to load model on
    
    Returns:
        DualSceneAlignerWithMatching model ready for evaluation
    
    Example:
        >>> config = {
        ...     'node_input_dim': 518,
        ...     'relation_dim': 512,
        ...     'hidden_dim': 128,
        ...     'rel_clip_matrix': dummy_clip_matrix,
        ...     'dropout': 0.1
        ... }
        >>> model = load_model_with_matching('checkpoints/model_best.pth', config)
    """
    from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner
    
    # Create base model
    base_model = DualSceneAligner(**base_model_config).to(device)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        base_model.load_state_dict(checkpoint['model'])
    else:
        base_model.load_state_dict(checkpoint)
    
    # Wrap with matching head
    model = DualSceneAlignerWithMatching(
        base_model=base_model,
        hidden_dim=hidden_dim,
        use_cosine=use_cosine
    ).to(device)
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    if not use_cosine:
        print(f"  Matching head parameters: {sum(p.numel() for p in model.matching_head.parameters()):,}")
    else:
        print(f"  Using cosine similarity (no extra parameters)")
    
    return model


# ============================================================
# Training Functions for the Matching Head
# ============================================================

def train_matching_head(model_with_matching, dataloader, epochs=10, lr=1e-3, device='cpu'):
    """
    Train ONLY the matching head while keeping base model frozen.
    
    This teaches the MLP to predict whether two graphs match based on their embeddings.
    The base model embeddings are already good from VICReg training.
    
    Args:
        model_with_matching: DualSceneAlignerWithMatching instance
        dataloader: DataLoader with graph pairs
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Trained model
    """
    if model_with_matching.use_cosine:
        print("⚠️  Model uses cosine similarity - no training needed!")
        return model_with_matching
    
    print("Training matching head...")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    
    # Freeze base model - we only train the matching head
    for param in model_with_matching.base_model.parameters():
        param.requires_grad = False
    
    # Only optimize matching head
    optimizer = torch.optim.Adam(
        model_with_matching.matching_head.parameters(),
        lr=lr
    )
    
    # Binary cross entropy loss
    criterion = nn.BCELoss()
    
    model_with_matching.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            batch_size = batch.get('batch_size', 1)
            
            # Forward pass
            out = model_with_matching(batch)
            matching_prob = out["matching_prob"]
            
            # Create labels: 1 for positive pairs (same scene), 0 for negatives
            # In batched contrastive learning, diagonal elements are positive pairs
            labels = torch.eye(batch_size, device=device).flatten()[:matching_prob.size(0)]
            
            # Compute loss
            loss = criterion(matching_prob, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            predictions = (matching_prob > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    print("✓ Matching head training complete!")
    
    # Unfreeze base model for future use
    for param in model_with_matching.base_model.parameters():
        param.requires_grad = True
    
    return model_with_matching


def save_model_with_matching(model, save_path):
    """Save the complete model including matching head."""
    checkpoint = {
        'base_model': model.base_model.state_dict(),
        'use_cosine': model.use_cosine,
        'hidden_dim': model.hidden_dim
    }
    
    if not model.use_cosine:
        checkpoint['matching_head'] = model.matching_head.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"✓ Saved model to {save_path}")


def load_full_model_with_matching(checkpoint_path, base_model_config, device='cpu'):
    """Load a previously saved DualSceneAlignerWithMatching (including matching head)."""
    from src.models.sgaligner.src.aligner.dual_scene_aligner import DualSceneAligner
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create base model
    base_model = DualSceneAligner(**base_model_config).to(device)
    base_model.load_state_dict(checkpoint['base_model'])
    
    # Create wrapper
    model = DualSceneAlignerWithMatching(
        base_model=base_model,
        hidden_dim=checkpoint['hidden_dim'],
        use_cosine=checkpoint['use_cosine']
    ).to(device)
    
    # Load matching head if it exists
    if not checkpoint['use_cosine'] and 'matching_head' in checkpoint:
        model.matching_head.load_state_dict(checkpoint['matching_head'])
    
    print(f"✓ Loaded complete model from {checkpoint_path}")
    return model


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("DualSceneAligner with Matching Head - Usage Examples")
    print("="*70)
    
    # Example 1: Wrap existing model with cosine similarity (no training)
    print("\n--- Example 1: Using Cosine Similarity (Simplest) ---")
    print("""
from dual_scene_aligner_with_matching import load_model_with_matching

config = {
    'node_input_dim': 518,
    'relation_dim': 512,
    'hidden_dim': 128,
    'rel_clip_matrix': dummy_clip_matrix,
    'dropout': 0.1
}

model = load_model_with_matching(
    checkpoint_path='checkpoints/model_best.pth',
    base_model_config=config,
    hidden_dim=128,
    use_cosine=True,  # Use cosine similarity
    device='cuda'
)

# Now ready for evaluation!
# model(batch) returns {"src_emb", "ref_emb", "matching_prob"}
""")
    
    # Example 2: Wrap with MLP and train it
    print("\n--- Example 2: Using MLP Matching Head (More Flexible) ---")
    print("""
from dual_scene_aligner_with_matching import (
    load_model_with_matching, 
    train_matching_head
)

# Load with MLP head (untrained)
model = load_model_with_matching(
    checkpoint_path='checkpoints/model_best.pth',
    base_model_config=config,
    hidden_dim=128,
    use_cosine=False,  # Use MLP
    device='cuda'
)

# Train the matching head
model = train_matching_head(
    model_with_matching=model,
    dataloader=train_dataloader,
    epochs=10,
    lr=1e-3,
    device='cuda'
)

# Save for later
save_model_with_matching(model, 'checkpoints/model_with_matching.pth')
""")
    
    print("\n" + "="*70)
    print("✓ Ready to use!")
    print("="*70)