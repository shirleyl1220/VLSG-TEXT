import torch
data = torch.load('/Users/shirley/Downloads/scanscribe_graphs_train_final_no_graph_min.pt', 
                  weights_only=False, map_location='cpu')
# Check structure
scene_id = list(data.keys())[0]
txt_id = list(data[scene_id].keys())[0]
sample = data[scene_id][txt_id]
print(type(sample))
print(sample.keys() if hasattr(sample, 'keys') else dir(sample))
# Check nodes
print("\nFirst node:")
if hasattr(sample, 'nodes'):
    node = list(sample.nodes.values())[0] if hasattr(sample.nodes, 'values') else sample.nodes[0]
    print(type(node), dir(node))