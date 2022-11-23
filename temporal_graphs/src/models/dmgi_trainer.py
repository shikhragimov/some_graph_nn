import logging
from typing import Optional
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import NormalizeFeatures
from temporal_graphs.src.models.dmgi import DMGI


LOGGER = logging.getLogger(__name__)


class DMGITrainer:
    """
    Trainer for DMGI
    """

    def __init__(
        self,
        data: HeteroData,
        out_channels: int,
        conv_name: Optional[str] = "GCNConv",
        normalize_features: Optional[bool] = False,
        dropout_probability: Optional[float] = 0.5,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        """
        Sets up parameters and DMGI model. Prepares the data for the model
        :param data: (torch_geometric.data.data.Data) graph
        :param out_channels: (int, optional) size of output layer. Default value is None
        :param conv_name: (str, optional) name of type of convolutions. Supported are GCNConv and SAGEConv
        :param normalize_features: (bool, optional) if True, normalizes node features. Default value is False
        :param dropout_probability: (float, optional) dropout probability
        :param device: (torch.device, optional) device on which objects will be allocated. Default is
        torch.device("cpu")
        """
        self.device = device
        self.data = data
        if normalize_features:
            self.normalize_features()
        self.data = self.data.to(str(self.device))
        self.x = self.data.node_stores[0].x  # it could be, but pycharm throw error
        # Expected type 'tuple[Union[str, tuple[str, str, str], tuple[str, str]]]', got 'str' instead
        self.edge_index = self.data.edge_index_dict.values()
        self.out_channels = out_channels
        self.conv_name = conv_name
        self.dropout_probability = dropout_probability
        self.model = self.setup_model()

    def normalize_features(self) -> None:
        """
        Row-normalizes the node features to sum-up to one.
        """
        self.data = NormalizeFeatures()(self.data)

    def setup_model(self) -> DMGI:
        """
        Creates GNN model based on the data type (weighted/unweighted).
        """
        model = DMGI(
            num_nodes=self.data["node"].num_nodes,
            in_channels=self.data["node"].x.size(-1),
            out_channels=self.out_channels,
            number_of_relations=len(self.data.edge_types),
            conv_name=self.conv_name
        ).to(self.device)
        return model

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Trains single epoch.
        :param optimizer: (torch.optim.Optimizer) optimizer
        :return: (float) loss
        """
        self.model.train()

        optimizer.zero_grad()
        pos_hs, neg_hs, summaries = self.model(self.x, self.edge_index, dropout_probability=self.dropout_probability)
        loss = self.model.loss(pos_hs, neg_hs, summaries)
        loss.backward()
        optimizer.step()

        return float(loss)

    def train(
        self,
        epochs: int,
        learning_rate: Optional[float] = 0.001,
        weight_decay: Optional[float] = 0.0,
        print_every_n_epoch: Optional[int] = 50,
    ) -> None:
        """
        Trains model.
        :param epochs: (int) number of epochs
        :param learning_rate: (float, optional) learning rate. Default value is 0.001
        :param weight_decay: (float, optional) Adam optimizer weight decay. Default value is 0.0
        :param print_every_n_epoch: (int, optional) maximum number of iterations. Default value is 1000
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(optimizer=optimizer)
            if epoch % print_every_n_epoch == 1:
                LOGGER.info(f"Epoch: {epoch:03d}, Loss: {loss:.3f}")

    @torch.no_grad()
    def get_embeddings(self) -> torch.Tensor:
        """
        Obtains generated node embeddings.
        :returns: (torch.Tensor) node embeddings
        """
        return self.model.Z

    def save(self, path: Optional[str] = "temporal_graphs/models/model.pt"):
        """
        Saves trained model.
        :param path: (str, optional) path with model name to save the trained model.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path: Optional[str] = "temporal_graphs/models/model.pt"):
        """
        Loads pretrained model.
        :param path: (str, optional): path of pretrained model.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
