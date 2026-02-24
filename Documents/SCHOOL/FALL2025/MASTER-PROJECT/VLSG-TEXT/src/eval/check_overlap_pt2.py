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