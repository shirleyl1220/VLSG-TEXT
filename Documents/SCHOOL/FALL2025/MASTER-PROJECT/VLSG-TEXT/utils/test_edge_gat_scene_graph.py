import torch
from torch_geometric.data import Data
from aligner.networks.edge_gat import MultiGAT_Edge


def build_dummy_graph():
    """
    Builds a tiny fake graph with 5 nodes and 4 edges.
    Each edge has 8-dimensional edge attributes.
    """
    x = torch.randn(5, 17)              # 17-dim node features
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4]
    ], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), 8)  # 8-dim edge features
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def main():
    data = build_dummy_graph()

    model = MultiGAT_Edge(
        n_units=[17, 128, 256],
        n_heads=[2, 2],
        edge_dim=8,
        dropout=0.0
    )

    out = model(data.x, data.edge_index, data.edge_attr)
    print("\n===== MultiGAT_Edge Test =====")
    print("Node input:", data.x.shape)
    print("Edge index:", data.edge_index.shape)
    print("Edge attr:", data.edge_attr.shape)
    print("Output:", out.shape)
    print(out)


if __name__ == "__main__":
    main()