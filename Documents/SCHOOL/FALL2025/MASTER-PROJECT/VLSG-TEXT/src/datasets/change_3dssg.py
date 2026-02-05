"""
Convert 3DSSG graphs from 300-dim (Word2Vec) to 518-dim (CLIP-based) format.

Output format: centroid(3) + color(3) + node_CLIP(512) = 518 dims
"""

import torch
import numpy as np
import clip
from tqdm import tqdm
from pathlib import Path
import sys

# Add paths
sys.path.append('../data_processing')
from scene_graph import SceneGraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CLIP
print("Loading CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=device)
print("✓ CLIP loaded")


def get_clip_embedding(label, clip_model, device):
    """Get CLIP embedding for a label."""
    with torch.no_grad():
        tokens = clip.tokenize([label]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()


def convert_graph_to_518(graph_data, clip_model, device):
    """
    Convert a single graph's node features to 518-dim format.
    
    Args:
        graph_data: Dictionary containing graph information
        clip_model: CLIP model for generating embeddings
        device: Device to run CLIP on
    
    Returns:
        Updated graph_data with 518-dim node features
    """
    # Create a copy to avoid modifying original
    new_graph = graph_data.copy()
    
    # Get node information
    if 'nodes' not in graph_data:
        print(f"Warning: No 'nodes' key in graph_data")
        return new_graph
    
    nodes = graph_data['nodes']
    new_nodes = {}
    
    for node_id, node_info in nodes.items():
        # Copy node info
        new_node = node_info.copy()
        
        # Get label
        if hasattr(node_info, 'label'):
            label = node_info.label
        elif isinstance(node_info, dict) and 'label' in node_info:
            label = node_info['label']
        else:
            print(f"Warning: No label found for node {node_id}, using 'object'")
            label = 'object'
        
        # Build 518-dim feature
        # Centroid (3D) - use existing or zero
        if hasattr(node_info, 'centroid'):
            centroid = np.array(node_info.centroid, dtype=np.float32)[:3]
        elif isinstance(node_info, dict) and 'centroid' in node_info:
            centroid = np.array(node_info['centroid'], dtype=np.float32)[:3]
        else:
            centroid = np.zeros(3, dtype=np.float32)
        
        # Color (3D) - use existing or default gray
        if hasattr(node_info, 'color'):
            color = np.array(node_info.color, dtype=np.float32)[:3]
        elif isinstance(node_info, dict) and 'color' in node_info:
            color = np.array(node_info['color'], dtype=np.float32)[:3]
        else:
            color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        
        # Node CLIP (512D)
        node_clip = get_clip_embedding(label, clip_model, device)
        
        # Concatenate: 3 + 3 + 512 = 518
        feature_518 = np.concatenate([centroid, color, node_clip])
        
        # Update node feature
        if hasattr(node_info, 'feature'):
            new_node.feature = feature_518
        elif isinstance(node_info, dict):
            new_node['feature'] = feature_518
        else:
            # Create new dict structure
            new_node = {
                'label': label,
                'centroid': centroid,
                'color': color,
                'feature': feature_518
            }
        
        new_nodes[node_id] = new_node
    
    new_graph['nodes'] = new_nodes
    return new_graph


def convert_3dssg_dataset(input_path, output_path, clip_model, device):
    """
    Convert entire 3DSSG dataset to 518-dim format.
    
    Args:
        input_path: Path to input .pt file (300-dim)
        output_path: Path to save output .pt file (518-dim)
        clip_model: CLIP model
        device: Device to run on
    """
    print(f"\nLoading 3DSSG data from: {input_path}")
    data = torch.load(input_path, weights_only=False, map_location='cpu')
    
    print(f"Found {len(data)} scenes")
    
    converted_data = {}
    
    for scene_id in tqdm(data.keys(), desc="Converting scenes"):
        try:
            converted_data[scene_id] = convert_graph_to_518(
                data[scene_id], 
                clip_model, 
                device
            )
        except Exception as e:
            print(f"\nError converting scene {scene_id}: {e}")
            continue
    
    print(f"\n✓ Converted {len(converted_data)}/{len(data)} scenes")
    
    # Save
    print(f"Saving to: {output_path}")
    torch.save(converted_data, output_path)
    print("✓ Saved!")
    
    # Verify
    print("\nVerifying conversion...")
    test_scene = list(converted_data.keys())[0]
    test_node = list(converted_data[test_scene]['nodes'].values())[0]
    
    if hasattr(test_node, 'feature'):
        feature_dim = len(test_node.feature)
    elif isinstance(test_node, dict) and 'feature' in test_node:
        feature_dim = len(test_node['feature'])
    else:
        feature_dim = "Unknown"
    
    print(f"Sample node feature dimension: {feature_dim}")
    
    if feature_dim == 518:
        print("✅ Conversion successful!")
    else:
        print(f"⚠️ Warning: Expected 518 dims, got {feature_dim}")


if __name__ == '__main__':
    # Paths
    input_path = '/content/drive/MyDrive/VLSG_Files/3dssg_graphs_processed_edgelists_relationembed.pt'
    output_path = '/content/drive/MyDrive/VLSG_Files/3dssg_graphs_518D.pt'
    
    # Convert
    convert_3dssg_dataset(input_path, output_path, clip_model, device)
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("\nYou can now use this file in your evaluation script:")
    print(f"_3dssg_scenes = torch.load('{output_path}', weights_only=False, map_location='cpu')")
    print("="*70)