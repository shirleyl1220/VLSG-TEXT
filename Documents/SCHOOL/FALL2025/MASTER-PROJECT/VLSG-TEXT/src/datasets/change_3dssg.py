"""
Convert 3DSSG graphs from 300-dim (Word2Vec) to 518-dim (CLIP-based) format.

Output format: centroid(3) + color(3) + node_CLIP(512) = 518 dims
"""

import torch
import numpy as np
import clip
from tqdm import tqdm

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
        graph_data: Dictionary with 'objects', 'relationships', 'edge_lists'
        clip_model: CLIP model for generating embeddings
        device: Device to run CLIP on
    
    Returns:
        Updated graph_data with 518-dim node features
    """
    # Create a copy to avoid modifying original
    new_graph = {
        'objects': {},
        'relationships': graph_data.get('relationships', {}),
        'edge_lists': graph_data.get('edge_lists', {})
    }
    
    # Get objects
    objects = graph_data.get('objects', {})
    
    for obj_id, obj_info in objects.items():
        # Copy object info
        new_obj = obj_info.copy()
        
        # Get label
        label = obj_info.get('label', 'object')
        
        # Build 518-dim feature
        # Centroid (3D) - extract from OBB if available, else zeros
        if 'obb' in obj_info and obj_info['obb'] is not None:
            obb = obj_info['obb']
            if 'centroid' in obb:
                centroid = np.array(obb['centroid'], dtype=np.float32)[:3]
            elif isinstance(obb, dict) and 'position' in obb:
                centroid = np.array(obb['position'], dtype=np.float32)[:3]
            else:
                centroid = np.zeros(3, dtype=np.float32)
        else:
            centroid = np.zeros(3, dtype=np.float32)
        
        # Color (3D) - default gray (no color info in 3DSSG usually)
        color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        
        # Node CLIP (512D)
        node_clip = get_clip_embedding(label, clip_model, device)
        
        # Concatenate: 3 + 3 + 512 = 518
        feature_518 = np.concatenate([centroid, color, node_clip])
        
        # Update object with new feature
        new_obj['feature_518'] = feature_518  # Add as new key to preserve original
        
        new_graph['objects'][obj_id] = new_obj
    
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
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Converted {len(converted_data)}/{len(data)} scenes")
    
    # Save
    print(f"Saving to: {output_path}")
    torch.save(converted_data, output_path)
    print("✓ Saved!")
    
    # Verify
    print("\nVerifying conversion...")
    test_scene_id = list(converted_data.keys())[0]
    test_scene = converted_data[test_scene_id]
    
    if 'objects' in test_scene and len(test_scene['objects']) > 0:
        test_obj_id = list(test_scene['objects'].keys())[0]
        test_obj = test_scene['objects'][test_obj_id]
        
        if 'feature_518' in test_obj:
            feature_dim = len(test_obj['feature_518'])
            print(f"Sample node feature dimension: {feature_dim}")
            
            if feature_dim == 518:
                print("✅ Conversion successful!")
                
                # Show sample
                print(f"\nSample object (scene: {test_scene_id}, obj: {test_obj_id}):")
                print(f"  Label: {test_obj.get('label', 'N/A')}")
                print(f"  Feature shape: {test_obj['feature_518'].shape}")
                print(f"  Centroid: {test_obj['feature_518'][:3]}")
                print(f"  Color: {test_obj['feature_518'][3:6]}")
                print(f"  CLIP (first 5): {test_obj['feature_518'][6:11]}")
            else:
                print(f"⚠️ Warning: Expected 518 dims, got {feature_dim}")
        else:
            print("⚠️ Warning: 'feature_518' key not found in object")
    else:
        print("⚠️ Warning: No objects found in test scene")


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
print("\nYou can now use this file in your evaluation script.")
print("="*70)