from src.datasets.dual_scene_graph_dataset import DualSceneGraphDataset

ds = DualSceneGraphDataset("scene_graphs")
sample = ds[0]

print(sample["node_feats_src"].shape)
print(sample["geom_edges_src"][0].shape)
print(sample["text_edges_src"][0].shape)