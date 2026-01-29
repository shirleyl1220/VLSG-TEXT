import torch

PT_PATH = "/Users/shirley/Documents/SCHOOL/SPRING25/masterproject/attempt2/whereami-text2sgm/playground/graph_models/data_checkpoints/processed_data/testing/scanscribe_graphs_test_final_no_graph_min.pt"

data = torch.load(PT_PATH, map_location="cpu", weights_only=False)

scene_id = list(data.keys())[0]
graph = data[scene_id][0]

nodes = graph["nodes"]
edges = graph["edges"]

print("\n=== GRAPH OVERVIEW ===")
print(f"Scene ID: {scene_id}")
print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {len(edges)}")

# ------------------------------------------------
# NODE STRUCTURE
# ------------------------------------------------
print("\n=== NODE STRUCTURE ===")

node = nodes[0]
for k, v in node.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: Tensor (shape={tuple(v.shape)})")
    elif isinstance(v, (int, float)):
        print(f"{k}: scalar (id / index / score)")
    elif isinstance(v, str):
        print(f"{k}: string (label / name)")
    elif isinstance(v, list):
        print(f"{k}: list (len={len(v)})")
    elif isinstance(v, dict):
        print(f"{k}: dict (metadata)")
    else:
        print(f"{k}: {type(v)}")

# ------------------------------------------------
# EDGE STRUCTURE
# ------------------------------------------------
print("\n=== EDGE STRUCTURE ===")

edge = edges[0]
for k, v in edge.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: Tensor (shape={tuple(v.shape)})")
    elif isinstance(v, (int, float)):
        print(f"{k}: scalar (id / index)")
    elif isinstance(v, str):
        print(f"{k}: string (relation label)")
    elif isinstance(v, list):
        print(f"{k}: list (len={len(v)})")
    elif isinstance(v, dict):
        print(f"{k}: dict (metadata)")
    else:
        print(f"{k}: {type(v)}")

print("\n=== DONE ===")


import torch

PT_PATH = "/Users/shirley/Documents/SCHOOL/SPRING25/masterproject/attempt2/whereami-text2sgm/playground/graph_models/data_checkpoints/processed_data/testing/scanscribe_graphs_test_final_no_graph_min.pt"

data = torch.load(PT_PATH, map_location="cpu", weights_only=False)

scene_id = list(data.keys())[0]
graph = data[scene_id][0]

nodes = graph["nodes"]
edges = graph["edges"]

print("\n=== NODE STRING FIELDS ===")
for i, node in enumerate(nodes):
    print(f"\nNode {i}:")
    for k, v in node.items():
        if isinstance(v, str):
            print(f"  {k}: {v}")

print("\n=== EDGE STRING FIELDS ===")
for i, edge in enumerate(edges):
    print(f"\nEdge {i}:")
    for k, v in edge.items():
        if isinstance(v, str):
            print(f"  {k}: {v}")