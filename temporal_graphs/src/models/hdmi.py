import torch
import torch_geometric.nn as nn
from temporal_graphs.src.models.layers.descriminator import Discriminator, JointDiscriminator
from temporal_graphs.src.models.layers.attention import CombineAttention

type_of_conv = {"GCNConv": nn.GCNConv,
                "SAGEConv": nn.SAGEConv}


class HDMI(torch.nn.Module):
    """
    Deep Multiplex Graph Infomax like in
        https://github.com/baoyujing/HDMI
        https://arxiv.org/pdf/2102.07810
        http://tonghanghang.org/pdfs/www21_hdmi_slides.pdf
    """

    def __init__(
            self, num_nodes: int,
            in_channels: int,
            out_channels: int,
            number_of_relations: int,
            conv_name: str,
            alpha: float = 0.0001,
            l1: float = 0.01,
            l2: float = 0.01,
            l3: float = 0.01,
            l4: float = 0.01,
            l5: float = 0.01,
            l6: float = 0.01,
    ) -> None:
        """
        Sets up parameters for DMGI model
        :param num_nodes: (int) number of input nodes
        :param in_channels: (int) number of input channels (features)
        :param out_channels: (int) number of output channels
        :param number_of_relations: (int) number of relations (types of edges)
        """
        super(HDMI, self).__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.number_of_relations = number_of_relations
        self.convs = torch.nn.ModuleList()
        self.conv_name = conv_name
        self.setup_conv_layers(self.conv_name)
        self.Z = torch.nn.Parameter(
            torch.rand(self.num_nodes, self.out_channels, dtype=torch.float)
        )  # a trainable consensus embedding matrix
        self.alpha = alpha
        self.discriminator_i = Discriminator(self.out_channels)
        self.discriminator_e = Discriminator(self.out_channels)
        self.discriminator_j = JointDiscriminator(self.in_channels, self.out_channels)

        self.discriminator_i = Discriminator(self.out_channels)
        self.discriminator_e = Discriminator(self.out_channels)
        self.discriminator_j = JointDiscriminator(self.in_channels, self.out_channels)

        self.fusion_discriminator_i = Discriminator(self.out_channels)
        self.fusion_discriminator_e = Discriminator(self.out_channels)
        self.fusion_discriminator_j = JointDiscriminator(self.in_channels, self.out_channels)

        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5
        self.l6 = l6

        self.attention = CombineAttention(self.out_channels, self.number_of_relations)
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
        pos_hs, neg_hs, pos_seqs, neg_seqs, summaries = [], [], [], [], []

        for conv, e_index in zip(self.convs, edge_index):
            pos_h = torch.nn.functional.dropout(x, p=dropout_probability, training=self.training)
            pos_seqs = pos_h
            pos_h = HDMI.get_h(conv, pos_h, e_index)
            pos_hs.append(pos_h)

            summaries.append(torch.sigmoid(pos_h.mean(dim=0, keepdim=True)))  # readout

            # corrupted network
            neg_h = torch.nn.functional.dropout(x, p=dropout_probability, training=self.training)
            neg_seqs = neg_h
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_hs.append(HDMI.get_h(conv, neg_h, e_index))
            break

        return pos_hs, neg_hs, summaries, pos_seqs, neg_seqs

    def loss(self, pos_hs, neg_hs, pos_seqs, neg_seqs, summaries) -> \
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        loss function
        :param neg_seqs:
        :param pos_seqs:
        :param pos_hs: positive representations of graph for different relation types after encoding
        :param neg_hs: corrupted representations of graph for different relation types after encoding
        :param summaries:
        :return:
        """
        # discriminators with relation-type specific cross entropy
        logits_e_list = []
        logits_i_list = []
        logits_j_list = []
        for pos_h, neg_h, summary in zip(pos_hs, neg_hs, summaries):  # iterations per relation type
            summary = summary.expand_as(pos_h)
            logits_e_list.append(self.discriminator_i(pos_h, neg_h, summary))
            logits_i_list.append(self.discriminator_e(pos_h, neg_h, pos_seqs))
            logits_j_list.append(self.discriminator_j(pos_h, neg_h, summary, pos_seqs, neg_seqs))
            break
        logits_e = torch.mean(torch.stack(logits_e_list))
        logits_i = torch.mean(torch.stack(logits_i_list))
        logits_j = torch.mean(torch.stack(logits_j_list))

        summaries = torch.mean(summaries, 0)
        pos_attention = self.attention(pos_hs)
        neg_attention = self.attention(neg_hs)
        # fusion
        logits_i_fusion = self.fusion_discriminator_i(pos_attention, neg_attention, torch.mean(summaries, 0))
        logits_e_fusion = self.fusion_discriminator_e(pos_attention, neg_attention, pos_seqs)
        logits_j_fusion = self.fusion_discriminator_j(pos_attention, neg_attention, pos_seqs, neg_seqs)

        loss = self.l1 * logits_e + self.l2 * logits_i + self.l3 * logits_j + self.l4 * logits_e_fusion + \
            self.l5 * logits_i_fusion + self.l6 * logits_j_fusion
        return (logits_e, logits_i, logits_j,
                logits_e_fusion, logits_i_fusion, logits_j_fusion,
                loss)
