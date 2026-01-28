"""
Unpack and explore ScanScribe training data.
Shows the structure and converts Word2Vec → CLIP embeddings.
"""

import torch
import json
import numpy as np
import clip
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
print("Loading CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=device)
print("✓ CLIP loaded")

# Load ScanScribe data
print("\nLoading ScanScribe data...")
scanscribe_path = '/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT-1/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/src/datasets/scanscribe_graphs_train_final_no_graph_min.pt'
scanscribe_scenes = torch.load(scanscribe_path, map_location='cpu', weights_only=False)

print(f"✓ Loaded {len(scanscribe_scenes)} scenes")

# Explore structure
print(f"\n{'='*70}")
print("DATA STRUCTURE")
print(f"{'='*70}")

# Get first scene
first_scene_id = list(scanscribe_scenes.keys())[0]
first_scene = scanscribe_scenes[first_scene_id]

print(f"\nFirst scene ID: {first_scene_id}")
print(f"Number of text descriptions: {len(first_scene)}")

# Get first text description
first_text_id = list(first_scene.keys())[0]
first_graph = first_scene[first_text_id]

print(f"\nFirst text ID: {first_text_id}")
print(f"Graph type: {type(first_graph)}")
print(f"Graph keys: {first_graph.keys()}")

# Explore nodes
print(f"\n--- NODES ---")
if 'nodes' in first_graph:
    nodes = first_graph['nodes']
    print(f"Number of nodes: {len(nodes)}")
    
    if len(nodes) > 0:
        first_node = nodes[0]
        print(f"\nFirst node structure:")
        print(f"  Keys: {first_node.keys()}")
        print(f"  ID: {first_node.get('id')}")
        print(f"  Label: {first_node.get('label')}")
        
        # Check embeddings
        if 'label_word2vec' in first_node:
            w2v_emb = np.array(first_node['label_word2vec'])
            print(f"  Word2Vec embedding shape: {w2v_emb.shape}")
            print(f"  Word2Vec first 5 values: {w2v_emb[:5]}")
        
        if 'label_ada' in first_node:
            ada_emb = np.array(first_node['label_ada'])
            print(f"  ADA embedding shape: {ada_emb.shape}")
        
        # Show first 3 nodes
        print(f"\nFirst 3 node labels:")
        for i, node in enumerate(nodes[:3]):
            print(f"  {i+1}. {node.get('label')}")

# Explore edges
print(f"\n--- EDGES ---")
if 'edges' in first_graph:
    edges = first_graph['edges']
    print(f"Number of edges: {len(edges)}")
    
    if len(edges) > 0:
        first_edge = edges[0]
        print(f"\nFirst edge structure:")
        print(f"  Keys: {first_edge.keys()}")
        print(f"  Source: {first_edge.get('source')}")
        print(f"  Target: {first_edge.get('target')}")
        print(f"  Relationship: {first_edge.get('relationship')}")
        
        # Check embeddings
        if 'relation_word2vec' in first_edge:
            w2v_rel = np.array(first_edge['relation_word2vec'])
            print(f"  Relation Word2Vec shape: {w2v_rel.shape}")
        
        # Show first 3 edges
        print(f"\nFirst 3 edges:")
        for i, edge in enumerate(edges[:3]):
            try:
                src_idx = int(edge['source']) - 1
                tgt_idx = int(edge['target']) - 1
                src_label = nodes[src_idx]['label'] if 0 <= src_idx < len(nodes) else '?'
                tgt_label = nodes[tgt_idx]['label'] if 0 <= tgt_idx < len(nodes) else '?'
                rel = edge.get('relationship', '?')
                print(f"  {i+1}. {src_label} --{rel}--> {tgt_label}")
            except:
                print(f"  {i+1}. [Invalid edge]")


# Count statistics
print(f"\n{'='*70}")
print("DATASET STATISTICS")
print(f"{'='*70}")

total_texts = 0
total_nodes = 0
total_edges = 0

for scene_id, scene_data in tqdm(scanscribe_scenes.items(), desc="Analyzing"):
    for text_id, graph in scene_data.items():
        total_texts += 1
        
        if 'nodes' in graph:
            total_nodes += len(graph['nodes'])
        
        if 'edges' in graph:
            total_edges += len(graph['edges'])

print(f"\nTotal scenes: {len(scanscribe_scenes)}")
print(f"Total text descriptions: {total_texts}")
print(f"Total nodes: {total_nodes}")
print(f"Total edges: {total_edges}")
print(f"Avg nodes per graph: {total_nodes / total_texts:.1f}")
print(f"Avg edges per graph: {total_edges / total_texts:.1f}")


# Convert Word2Vec → CLIP for first graph
print(f"\n{'='*70}")
print("CONVERTING WORD2VEC → CLIP (EXAMPLE)")
print(f"{'='*70}")

def convert_to_clip(graph_with_w2v):
    """Convert Word2Vec embeddings to CLIP embeddings."""
    
    new_graph = {
        'nodes': [],
        'edges': []
    }
    
    # Convert nodes
    for node in graph_with_w2v['nodes']:
        label = node['label']
        
        # Get CLIP embedding
        with torch.no_grad():
            tokens = clip.tokenize([label]).to(device)
            clip_emb = clip_model.encode_text(tokens)
            clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)
        
        new_node = {
            'id': node['id'],
            'label': label,
            'clip_text_emb': clip_emb[0].cpu().numpy().tolist(),
            'attributes': node.get('attributes', [])
        }
        
        new_graph['nodes'].append(new_node)
    
    # Convert edges
    for edge in graph_with_w2v['edges']:
        relationship = edge['relationship']
        
        # Get CLIP embedding for relationship
        with torch.no_grad():
            tokens = clip.tokenize([relationship]).to(device)
            clip_rel_emb = clip_model.encode_text(tokens)
            clip_rel_emb = clip_rel_emb / clip_rel_emb.norm(dim=-1, keepdim=True)
        
        new_edge = {
            'source': edge['source'],
            'target': edge['target'],
            'relationship': relationship,
            'clip_relation_emb': clip_rel_emb[0].cpu().numpy().tolist()
        }
        
        new_graph['edges'].append(new_edge)
    
    return new_graph


# Test conversion
print("\nConverting first graph...")
clip_graph = convert_to_clip(first_graph)

print(f"✓ Converted!")
print(f"  Nodes: {len(clip_graph['nodes'])}")
print(f"  First node CLIP embedding shape: {len(clip_graph['nodes'][0]['clip_text_emb'])}")
print(f"  Edges: {len(clip_graph['edges'])}")
if len(clip_graph['edges']) > 0:
    print(f"  First edge CLIP embedding shape: {len(clip_graph['edges'][0]['clip_relation_emb'])}")


print(f"\n{'='*70}")
print("YOUR STRATEGY: CLIP + Geometry + GNN")
print(f"{'='*70}")
