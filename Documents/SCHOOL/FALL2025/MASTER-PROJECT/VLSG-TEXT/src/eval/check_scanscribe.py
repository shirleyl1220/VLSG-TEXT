import torch

scanscribe_graphs = torch.load(
    '/Users/shirley/Documents/SCHOOL/SPRING25/masterproject/attempt2/whereami-text2sgm/playground/graph_models/data_checkpoints/processed_data/testing/scanscribe_graphs_test_final_no_graph_min.pt',
    map_location='cpu',
    weights_only=False
)

# Get first scene, first text
first_scene_id = list(scanscribe_graphs.keys())[0]
first_scene = scanscribe_graphs[first_scene_id]
graph = first_scene[0]  # First text description

print(f"Graph dict keys: {graph.keys()}")
print(f"\n")

# Check each field
for key in graph.keys():
    value = graph[key]
    print(f"{key}:")
    print(f"  Type: {type(value)}")
    if hasattr(value, 'shape'):
        print(f"  Shape: {value.shape}")
    elif hasattr(value, '__len__'):
        print(f"  Length: {len(value)}")
    if isinstance(value, (list, tuple)) and len(value) > 0:
        print(f"  First element: {value[0]}")
    print()