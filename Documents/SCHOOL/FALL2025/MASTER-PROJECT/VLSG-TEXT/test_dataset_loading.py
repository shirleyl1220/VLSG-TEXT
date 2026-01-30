"""
Diagnostic script to check if dataset is loading ref/src scenes correctly
"""
import sys
import json
import torch
sys.path.append('/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT-1/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT')

from src.datasets.dual_scene_graph_dataset_518_v2 import DualSceneGraphDataset

# Load dataset
dataset = DualSceneGraphDataset(
    dataset_dir='/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT-1/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/scene_graph_clip',
    metadata_path='/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/meta_files/3RScan.json',
    augment_ratio=0.0
)

print(f"✓ Dataset loaded: {len(dataset)} scenes\n")

# Test loading first few samples
for i in range(min(3, len(dataset))):
    print(f"\n{'='*80}")
    print(f"SAMPLE {i}")
    print(f"{'='*80}")
    
    sample = dataset[i]
    
    # Check scene CLIP embeddings
    scene_clip_src = sample['scene_clip_src']
    scene_clip_ref = sample['scene_clip_ref']
    
    print(f"\n1. Scene CLIP Embeddings:")
    print(f"   src shape: {scene_clip_src.shape}")
    print(f"   ref shape: {scene_clip_ref.shape}")
    print(f"   src norm:  {scene_clip_src.norm():.4f}")
    print(f"   ref norm:  {scene_clip_ref.norm():.4f}")
    print(f"   src first 5: {scene_clip_src[:5]}")
    print(f"   ref first 5: {scene_clip_ref[:5]}")
    
    if scene_clip_src.norm() < 0.01:
        print("   ❌ SOURCE scene CLIP is zeros!")
    else:
        print("   ✓ Source scene CLIP looks good")
    
    if scene_clip_ref.norm() < 0.01:
        print("   ❌ REFERENCE scene CLIP is zeros!")
    else:
        print("   ✓ Reference scene CLIP looks good")
    
    # Check node features
    node_feats_src = sample['node_feats_src']
    node_feats_ref = sample['node_feats_ref']
    
    print(f"\n2. Node Features:")
    print(f"   src shape: {node_feats_src.shape} (should be [num_nodes, 518])")
    print(f"   ref shape: {node_feats_ref.shape} (should be [num_nodes, 518])")
    
    # Check first node
    if len(node_feats_src) > 0:
        node_0 = node_feats_src[0]
        print(f"\n   First node breakdown:")
        print(f"     Centroid (0-3):  {node_0[:3]}")
        print(f"     Color (3-6):     {node_0[3:6]}")
        print(f"     CLIP (6-518):    first 5 = {node_0[6:11]}")
        print(f"     Node CLIP norm:  {node_0[6:].norm():.4f}")
        
        if node_0[6:].norm() < 0.01:
            print("     ❌ Node CLIP embeddings are zeros!")
        else:
            print("     ✓ Node CLIP embeddings look good")
    
    # Check if scenes are different
    print(f"\n3. Scene Comparison:")
    print(f"   Same room: {sample['is_positive']}")
    print(f"   Room ID: {sample['room_id']}")
    
    # Check if src and ref are actually different
    if len(node_feats_src) > 0 and len(node_feats_ref) > 0:
        src_node_0 = node_feats_src[0]
        ref_node_0 = node_feats_ref[0]
        
        centroid_diff = (src_node_0[:3] - ref_node_0[:3]).abs().mean().item()
        clip_diff = (src_node_0[6:] - ref_node_0[6:]).abs().mean().item()
        total_diff = (src_node_0 - ref_node_0).abs().mean().item()
        
        print(f"   First node differences:")
        print(f"     Centroid diff: {centroid_diff:.4f}")
        print(f"     CLIP diff:     {clip_diff:.4f}")
        print(f"     Total diff:    {total_diff:.4f}")
        
        if total_diff < 0.001:
            print("     ❌ SCENES ARE IDENTICAL! This is a problem!")
        elif clip_diff < 0.01:
            print("     ⚠️  CLIP embeddings are too similar")
        else:
            print("     ✓ Scenes are different")
    
    # Check edges
    print(f"\n4. Graph Structure:")
    print(f"   Geom edges src: {sample['geom_edges_src'].shape}")
    print(f"   Geom edges ref: {sample['geom_edges_ref'].shape}")
    print(f"   Text edges src: {sample['text_edges_src'].shape}")
    print(f"   Text edges ref: {sample['text_edges_ref'].shape}")

# Load raw JSON to verify structure
print(f"\n\n{'='*80}")
print("VERIFYING RAW JSON STRUCTURE")
print(f"{'='*80}\n")

sample_file = dataset.scene_files[0]
print(f"Checking: {sample_file}")

with open(sample_file) as f:
    data = json.load(f)

print(f"\nJSON structure:")
print(f"  Has 'scene_clip_emb': {'scene_clip_emb' in data}")
print(f"  Has 'nodes': {'nodes' in data}")
print(f"  Has 'edges_text': {'edges_text' in data}")

if 'scene_clip_emb' in data:
    scene_clip = data['scene_clip_emb']
    print(f"  scene_clip_emb length: {len(scene_clip)}")
    print(f"  scene_clip_emb type: {type(scene_clip)}")
    print(f"  scene_clip_emb first 5: {scene_clip[:5]}")

if 'nodes' in data:
    node_ids = list(data['nodes'].keys())
    print(f"  Number of nodes: {len(node_ids)}")
    if node_ids:
        first_node = data['nodes'][node_ids[0]]
        print(f"  First node keys: {list(first_node.keys())}")
        print(f"  First node has 'clip_text_emb': {'clip_text_emb' in first_node}")
        if 'clip_text_emb' in first_node:
            print(f"  First node clip_text_emb length: {len(first_node['clip_text_emb'])}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
