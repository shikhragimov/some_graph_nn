import logging
import torch
from src.data.mock_data import get_temporal_mock_graph
from src.temporal_graph.build_graph import create_torch_temporal_graph_from_df
from src.models.dmgi_trainer import DMGITrainer

logging.basicConfig(level="INFO")
prefix = "dmgi"

graph = get_temporal_mock_graph()
graph = create_torch_temporal_graph_from_df(graph, save=False, path_prefix="../")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = DMGITrainer(data=graph, out_channels=10, conv_name="GCNConv", normalize_features=False, device=device)
model.train(epochs=100, learning_rate=0.005, weight_decay=0.00005, print_every_n_epoch=10)
model.save(path=f"../temporal_graphs/models/{prefix}_model.pt")
embeddings = model.get_embeddings()
torch.save(embeddings, f"../temporal_graphs/models/{prefix}_embeddings.pt")
