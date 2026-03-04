import torch
import os

# The 218-scene test set
test = torch.load('/Users/shirley/Downloads/scanscribe_cleaned_original_node_edge_features.pt',
                   map_location='cpu', weights_only=False)

# Get training scene IDs from JSON filenames
train_dir = '/Users/shirley/Documents/SCHOOL/FALL2025/MASTERSPROJECT/VLSG-TEXT/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/scene_graph_clip_new'
train_ids = set(f.replace('.json', '') for f in os.listdir(train_dir) if f.endswith('.json'))

test_ids = set(test.keys())

overlap = test_ids & train_ids
print(f"Test scenes: {len(test_ids)}")
print(f"Train scenes: {len(train_ids)}")
print(f"Overlap (LEAKAGE): {len(overlap)}")
if overlap:
    print(f"Leaked scene IDs: {sorted(overlap)}")
else:
    print("✓ No overlap — test set is clean!")



scanscribe_train = torch.load('/Users/shirley/Downloads/scanscribe_graphs_train_final_no_graph_min.pt',
                               weights_only=False, map_location='cpu')
dssg = torch.load('/Users/shirley/Downloads/3dssg_graphs_processed_edgelists_relationembed.pt',
                   weights_only=False, map_location='cpu')

# Check overlap
scanscribe_ids = set(scanscribe_train.keys())
dssg_ids = set(dssg.keys())
overlap = scanscribe_ids & dssg_ids

print(f"ScanScribe train scenes: {len(scanscribe_ids)}")
print(f"3DSSG scenes: {len(dssg_ids)}")
print(f"Overlap (can form pairs): {len(overlap)}")
print(f"ScanScribe scenes NOT in 3DSSG: {len(scanscribe_ids - dssg_ids)}")

json_dir = '/Users/shirley/Downloads/scene_graph_clip_163'
missing = [sid for sid in scanscribe_train.keys() 
           if not os.path.exists(os.path.join(json_dir, f"{sid}.json"))]
print(f"Missing JSONs: {len(missing)}/{len(scanscribe_train)}")

# Load one sample and inspect
img_test = torch.load('/Users/shirley/Downloads/scanscribe_graphs_test_518D.pt',
                       weights_only=False, map_location='cpu')
sample_scene_id, sample_scene = next(iter(img_test.items()))
print("Sample scene id:", sample_scene_id)
print("Sample scene keys:", sample_scene.keys())

# Try common key names used for node features
node_key_candidates = ['node_features', 'x', 'node_feat', 'nodes']
node_key = next((k for k in node_key_candidates if k in sample_scene), None)

if node_key is not None and hasattr(sample_scene[node_key], 'shape'):
    node_tensor = sample_scene[node_key]
    print(f"Node feature key: {node_key}")
    print("Node features shape:", tuple(node_tensor.shape))
    if node_tensor.ndim >= 2:
        print("#nodes:", node_tensor.shape[0])
        print("node_dim:", node_tensor.shape[1])
else:
    print("Could not find a standard node feature key.")
    print("Tensor-like fields in this scene:")
    for k, v in sample_scene.items():
        if hasattr(v, 'shape'):
            print(f"  - {k}: shape={tuple(v.shape)}, dtype={getattr(v, 'dtype', None)}")

# Optional: print edge information if available
if 'edge_features' in sample_scene and hasattr(sample_scene['edge_features'], 'shape'):
    print("Edge features shape:", tuple(sample_scene['edge_features'].shape))
if 'edge_attr' in sample_scene and hasattr(sample_scene['edge_attr'], 'shape'):
    print("Edge attr shape:", tuple(sample_scene['edge_attr'].shape))
if 'edge_index' in sample_scene and hasattr(sample_scene['edge_index'], 'shape'):
    print("Edge index shape:", tuple(sample_scene['edge_index'].shape))