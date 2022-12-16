import numpy as np
import torch
from torch_geometric.data import HeteroData


def create_torch_temporal_graph_from_df(graph, save=True, path_prefix="", feature_size=4):
    nodes = set(graph["node_1"].unique())
    nodes.update(set(graph["node_2"].unique()))
    nodes = list(nodes)
    print(f"number of nodes: {len(nodes)}")
    features = np.random.rand(len(nodes), feature_size)

    data = HeteroData()
    data["node"].x = torch.tensor(features, dtype=torch.float)
    for i, date_group in enumerate(np.sort(graph["date_group"].unique())):
        data["node", f"in_date_group_{date_group}", "node"].edge_index = torch.tensor(
            np.array([
                np.append(
                    graph[graph["date_group"] == date_group]["node_1"].values,
                    graph[graph["date_group"] == date_group]["node_2"].values),
                np.append(
                    graph[graph["date_group"] == date_group]["node_2"].values,
                    graph[graph["date_group"] == date_group]["node_1"].values)
            ]),
            dtype=torch.long,
        )
    if save:
        torch.save(data, path_prefix + "data/processed/hetero_data_multiplex.pt")
    return data
