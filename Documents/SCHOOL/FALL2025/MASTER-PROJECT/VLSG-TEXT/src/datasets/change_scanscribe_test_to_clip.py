"""
Create 518D test set from scanscribe_cleaned_original_node_edge_features.pt (218 scenes)
Structure: {scene_id: {txt_id: {nodes, edges}}}
Node features: centroid(3) + color(3) + CLIP(512) = 518 dims
"""

import torch
import clip
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=device)
print("✓ CLIP loaded\n")

# UPDATE THESE PATHS
INPUT_PATH  = '/Users/shirley/Downloads/scanscribe_cleaned_original_node_edge_features.pt'
OUTPUT_PATH = '/Users/shirley/Downloads/scanscribe_cleaned_original_518D.pt'


def get_clip_embedding(label):
    with torch.no_grad():
        tokens = clip.tokenize([label]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()


def convert_graph_to_518D(graph):
    """Convert a single graph's nodes to 518D features, keep edges as-is."""
    new_nodes = []
    for node in graph['nodes']:
        label = node['label']

        # centroid (3) - text has no spatial info
        centroid = np.zeros(3, dtype=np.float32)

        # color (3) - default gray, normalized
        color = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # CLIP (512)
        clip_emb = get_clip_embedding(label).astype(np.float32)

        # 518D feature vector
        feat = np.concatenate([centroid, color, clip_emb])

        new_node = {**node, 'features_518': feat.tolist()}
        new_nodes.append(new_node)

    return {
        'nodes': new_nodes,
        'edges': graph['edges']
    }


# Load source
print(f"Loading {INPUT_PATH}...")
data = torch.load(INPUT_PATH, map_location='cpu', weights_only=False)
print(f"✓ Loaded {len(data)} scenes\n")


#remove leaked scenes from training data
leaked = {'2e36952b-e133-204c-911e-7644cb34e8b2', '2e36954b-e133-204c-92ad-1a66c6f63e1a', 
          '2e36955d-e133-204c-90da-122ae14d42a3', '41385827-a238-2435-8320-d8d3eb507f5e', 
          '41385847-a238-2435-838b-61864922c518', '4138584b-a238-2435-8128-a939fb07c1c8', 
          '41385851-a238-2435-8056-b7d662a97c93', '4acaebba-6c10-2a2a-8650-34c2f160db99', 
          '4acaebc0-6c10-2a2a-852e-0226d6539299', '4acaebcc-6c10-2a2a-858b-29c7e4fb410d', 
          '50e7c0ad-0730-2a5f-8635-252404ee82e0', '5630cfc9-12bf-2860-84ed-5bb189f0e94e', 
          '5630cfcb-12bf-2860-87ee-b4e4a5bf0cb0'}

# Remove from test data before converting to 518D
for sid in leaked:
    if sid in data:
        del data[sid]

print(f"Clean test set: {len(data)} scenes")  # Should be 205

# Check structure
first_scene = list(data.keys())[0]
print(f"First scene: {first_scene}")
print(f"Keys: {data[first_scene].keys()}")
first_txt = list(data[first_scene].keys())[0]
print(f"First txt_id: {first_txt}")
print(f"First node keys: {data[first_scene][first_txt]['nodes'][0].keys()}\n")

# Convert
output = {}
for scene_id in tqdm(data, desc="Converting to 518D"):
    output[scene_id] = {}
    for txt_id, graph in data[scene_id].items():
        output[scene_id][txt_id] = convert_graph_to_518D(graph)

# Save
print(f"\nSaving to {OUTPUT_PATH}...")
torch.save(output, OUTPUT_PATH)
print(f"✓ Saved {len(output)} scenes")

# Verify
print("\nVerifying...")
check = torch.load(OUTPUT_PATH, map_location='cpu', weights_only=False)
first_scene = list(check.keys())[0]
first_txt = list(check[first_scene].keys())[0]
first_node = check[first_scene][first_txt]['nodes'][0]
print(f"  Scene: {first_scene}")
print(f"  txt_id: {first_txt}")
print(f"  First node label: {first_node['label']}")
print(f"  Feature length: {len(first_node['features_518'])} (should be 518)")
print(f"  Total scenes: {len(check)}")
total_graphs = sum(len(check[s]) for s in check)
print(f"  Total graphs (all txt_ids): {total_graphs}")
print("✓ Done!")