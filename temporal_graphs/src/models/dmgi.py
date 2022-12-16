import torch
import torch_geometric.nn as nn


type_of_conv = {"GCNConv": nn.GCNConv,
                "SAGEConv": nn.SAGEConv}


class DMGI(torch.nn.Module):
    """
    Deep Multiplex Graph Infomax like in
        https://github.com/pcy1302/DMGI
        https://arxiv.org/abs/1911.06750
    """

    def __init__(
        self, num_nodes: int,
            in_channels: int,
            out_channels: int,
            number_of_relations: int,
            conv_name: str,
            alpha: float = 0.0001
    ) -> None:
        """
        Sets up parameters for DMGI model
        :param num_nodes: (int) number of input nodes
        :param in_channels: (int) number of input channels (features)
        :param out_channels: (int) number of output channels
        :param number_of_relations: (int) number of relations (types of edges)
        """
        super(DMGI, self).__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.number_of_relations = number_of_relations
        self.convs = torch.nn.ModuleList()
        self.conv_name = conv_name
        self.setup_conv_layers(self.conv_name)
        self.M = torch.nn.Bilinear(self.out_channels, self.out_channels, 1)  # a trainable scoring matrix
        self.Z = torch.nn.Parameter(
            torch.rand(self.num_nodes, self.out_channels, dtype=torch.float)
        )  # a trainable consensus embedding matrix
        self.alpha = alpha
        self.reset_parameters()

    def setup_conv_layers(self, conv_name) -> None:
        """
        Creates convolutional layers for each relation.
        """
        for _ in range(self.number_of_relations):
            self.convs.append(type_of_conv[conv_name](self.in_channels, self.out_channels))
        print(self.in_channels)
        print(self.out_channels)

    def reset_parameters(self) -> None:
        """
        Reset parameters and initialise them.
        :return:
        """
        for conv in self.convs:
            conv.reset_parameters()

        torch.nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.Z)

    @staticmethod
    def get_h(conv, h, e_index):
        h = conv(h, e_index)
        h = torch.nn.functional.relu(h)
        return h

    def forward(self, x: torch.Tensor, edge_index: dict.values, dropout_probability: float):
        """
        Feed Forward
        :param x: (torch.Tensor) inputs - features
        :param edge_index: (torch.Tensor) - values representing edges (for each relation
        2 x number of edges in these relations)
        :param dropout_probability: probability of dropout
        :return: pos_hs (positive relation-type specific node embedding matrix),
                 neg_hs (negative (corrupted) relation-type specific node embedding matrix),
                 summaries
        """
        pos_hs, neg_hs, summaries = [], [], []

        for conv, e_index in zip(self.convs, edge_index):
            pos_h = torch.nn.functional.dropout(x, p=dropout_probability, training=self.training)
            pos_h = DMGI.get_h(conv, pos_h, e_index)
            pos_hs.append(pos_h)

            summaries.append(torch.sigmoid(pos_h.mean(dim=0, keepdim=True)))  # readout

            # corrupted network
            neg_h = torch.nn.functional.dropout(x, p=dropout_probability, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_hs.append(DMGI.get_h(conv, neg_h, e_index))

        return pos_hs, neg_hs, summaries

    def loss(self, pos_hs, neg_hs, summaries) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        loss function
        :param pos_hs: positive representations of graph for different relation types after encoding
        :param neg_hs: corrupted representations of graph for different relation types after encoding
        :param summaries:
        :return:
        """
        # relation-type specific cross entropy
        loss_rel = torch.tensor([0.0])  # relation type loss
        for pos_h, neg_h, summary in zip(pos_hs, neg_hs, summaries):  # iterations per relation type
            summary = summary.expand_as(pos_h)

            loss_rel += -torch.log(torch.sigmoid(self.M(pos_h, summary)) + 1e-15).mean()
            loss_rel += -torch.log(1 - torch.sigmoid(self.M(neg_h, summary)) + 1e-15).mean()
            break

        # aggregation
        pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
        neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)

        # consensus regularisation
        pos_reg_loss = (self.Z - pos_mean).pow(2).sum()
        neg_reg_loss = (self.Z - neg_mean).pow(2).sum()
        loss_cr = self.alpha * (pos_reg_loss - neg_reg_loss)

        # return torch.sigmoid(self.M(pos_h, summary)).mean(),
        # 1-torch.sigmoid(self.M(neg_h, summary)).mean(), loss_rel+loss_cr

        return loss_rel, loss_cr, loss_rel+loss_cr
