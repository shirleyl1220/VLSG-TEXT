import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv



class EdgeGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2, edge_dim=8, dropout=0.0):
        super().__init__()

        # IMPORTANT: project ANY edge_attr (geom=8, CLIP=512) to the GAT input dim
        self.edge_proj = nn.Linear(edge_dim, in_dim)

        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=in_dim,              # ← must match projected dimension!
            add_self_loops=False
        )

        self.mlp = nn.Linear(in_dim, out_dim * heads)

    def forward(self, x, edge_index, edge_attr, skip_edges=False):
        if skip_edges:
            return self.mlp(x)

        # Project edge features to correct dimension
        if edge_attr is not None:
            edge_attr = self.edge_proj(edge_attr)

        return self.gat(x, edge_index, edge_attr)


class MultiGAT_Edge(nn.Module):
    def __init__(self, n_units=[7, 128, 128], n_heads=[2, 2],
                 edge_dim=8, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(len(n_units) - 1):
            self.layers.append(
                EdgeGATLayer(
                    in_dim=n_units[i],
                    out_dim=n_units[i+1] // n_heads[i],
                    heads=n_heads[i],
                    edge_dim=edge_dim,      # CLIP = 512, geom = 8
                    dropout=dropout
                )
            )

    def forward(self, x, edge_index, edge_attr,
                edge_weight_factor=1.0,
                skip_edges_first_layer=False):

        for layer_i, layer in enumerate(self.layers):

            # First layer skips edges: pure node MLP
            if skip_edges_first_layer and layer_i == 0:
                x = layer(x, None, None, skip_edges=True)
                continue

            # Scale edge attributes if needed
            if edge_weight_factor != 1.0 and edge_attr is not None:
                edge_attr = edge_attr * edge_weight_factor

            # Run GAT layer
            x = layer(x, edge_index, edge_attr)

        return x

class MultiGAT_Edge(nn.Module):
    def __init__(self, n_units=[7, 128, 128], n_heads=[2, 2],
                 edge_dim=8, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(len(n_units) - 1):
            self.layers.append(
                EdgeGATLayer(
                    in_dim=n_units[i],
                    out_dim=n_units[i+1] // n_heads[i],
                    heads=n_heads[i],
                    edge_dim=edge_dim,
                    dropout=dropout
                )
            )

    def forward(self, x, edge_index, edge_attr,
                edge_weight_factor=1.0,
                skip_edges_first_layer=False):

        for layer_i, layer in enumerate(self.layers):

            # First layer: node-only feed-forward
            if skip_edges_first_layer and layer_i == 0:
                x = layer(x, None, None, skip_edges=True)
                continue

            # Scale edges
            if edge_weight_factor != 1.0 and edge_attr is not None:
                edge_attr = edge_attr * edge_weight_factor

            x = layer(x, edge_index, edge_attr)

        return x