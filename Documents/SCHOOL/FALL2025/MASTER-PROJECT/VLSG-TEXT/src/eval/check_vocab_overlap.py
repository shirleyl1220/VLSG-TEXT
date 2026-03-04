import torch
import numpy as np
from tqdm import tqdm

# Load both query sets
scanscribe_test = torch.load('/Users/shirley/Downloads/scanscribe_graphs_test_final_no_graph_min.pt', 
                              weights_only=False, map_location='cpu')
image_queries = torch.load('/Users/shirley/Downloads/scanscribe_text_graphs_from_image_desc_node_edge_features.pt',
                            weights_only=False, map_location='cpu')
dssg = torch.load('/Users/shirley/Downloads/3dssg_graphs_processed_edgelists_relationembed.pt',
                   weights_only=False, map_location='cpu')

# 
# ── Get all 3DSSG labels ───────────────────────────────────
dssg_labels = set()
for scene_id in dssg:
    for obj_id in dssg[scene_id]['objects']:
        label = dssg[scene_id]['objects'][obj_id]['label'].lower()
        dssg_labels.add(label)

print(f"\n3DSSG unique labels: {len(dssg_labels)}")
print(f"Sample: {list(dssg_labels)[:10]}")

# ── ScanScribe query vocab overlap ─────────────────────────
print("\nChecking ScanScribe query overlap...")
scanscribe_matches = []
scanscribe_labels_seen = set()

for scene_id in tqdm(scanscribe_test):
    for txt_id in scanscribe_test[scene_id]:
        graph = scanscribe_test[scene_id][txt_id]
        
        # Access nodes as dict or list
        if isinstance(graph, dict):
            nodes = graph.get('nodes', {})
        else:
            # SceneGraph object
            nodes = graph.nodes if hasattr(graph, 'nodes') else {}
        
        labels = []
        if isinstance(nodes, dict):
            node_iter = nodes.values()
        elif isinstance(nodes, list):
            node_iter = nodes
        else:
            node_iter = []

        for node in node_iter:
            if isinstance(node, dict):
                label = node.get('label', 'unknown').lower()
            else:
                label = node.label.lower() if hasattr(node, 'label') else str(node).lower()
            labels.append(label)
            scanscribe_labels_seen.add(label)
        
        if labels:
            match_rate = sum(1 for l in labels if l in dssg_labels) / len(labels)
            scanscribe_matches.append(match_rate)

print(f"ScanScribe unique labels: {len(scanscribe_labels_seen)}")
print(f"ScanScribe vocab overlap with 3DSSG: {np.mean(scanscribe_matches)*100:.1f}%")
print(f"Sample ScanScribe labels: {list(scanscribe_labels_seen)[:15]}")

# ── Image query vocab overlap ──────────────────────────────
# Load your image/human query graphs
scene_id = list(image_queries.keys())[0]
print(f"Scene ID: {scene_id}")
print(f"Scene type: {type(image_queries[scene_id])}")
print(f"Scene keys/attrs: {image_queries[scene_id].keys() if isinstance(image_queries[scene_id], dict) else dir(image_queries[scene_id])}")

# Go one level deeper
val = image_queries[scene_id]
if isinstance(val, dict):
    first_key = list(val.keys())[0]

    print(f"First key: {first_key}") #nodes
    print(image_queries[scene_id]['nodes'][0])
    
    print(f"First value type: {type(val[first_key])}")
#     print(f"First value: {val[first_key]}")
# else:
#     print(f"Nodes: {val.nodes if hasattr(val, 'nodes') else 'no nodes'}")
# print("\nChecking Image query overlap...")
image_matches = []
image_labels_seen = set()   
for scene_id in tqdm(image_queries):
    graph = image_queries[scene_id]
    
    # Access nodes as dict or list
    if isinstance(graph, dict):
        nodes = graph.get('nodes', {})
    else:
        # SceneGraph object
        nodes = graph.nodes if hasattr(graph, 'nodes') else {}
    
    labels = []
    if isinstance(nodes, dict):
        node_iter = nodes.values()
    elif isinstance(nodes, list):
        node_iter = nodes
    else:
        node_iter = []

    for node in node_iter:
        if isinstance(node, dict):
            label = node.get('label', 'unknown').lower()
        else:
            label = node.label.lower() if hasattr(node, 'label') else str(node).lower()
        labels.append(label)
        image_labels_seen.add(label)
    
    if labels:
        match_rate = sum(1 for l in labels if l in dssg_labels) / len(labels)
        image_matches.append(match_rate)
print(f"Image query unique labels: {len(image_labels_seen)}")
print(f"Image query vocab overlap with 3DSSG: {np.mean(image_matches)*100:.1f}%")
print(f"Sample Image query labels: {list(image_labels_seen)[:15]}")
# ── Summary ────────────────────────────────────────────────
print("\n" + "="*50)
print("VOCAB OVERLAP SUMMARY")
print("="*50)
print(f"ScanScribe queries → 3DSSG: {np.mean(scanscribe_matches)*100:.1f}%")
# if image_matches:
#     print(f"Image queries → 3DSSG:      {np.mean(image_matches)*100:.1f}%")
print("\nIf ScanScribe >> Image, baseline benefits from vocabulary matching.")