import torch


class Discriminator(torch.nn.Module):

    def __init__(
            self, out_channels
    ) -> None:
        super(Discriminator, self).__init__()
        self.M = torch.nn.Bilinear(out_channels, out_channels, 1)  # a trainable scoring matrix

    def forward(self, positive_h, negative_h, values):
        loss = - torch.log(torch.sigmoid(self.M(positive_h, values)) + 1e-15).mean()\
               - torch.log(1 - torch.sigmoid(self.M(negative_h, values)) + 1e-15).mean()
        return loss


class JointDiscriminator(torch.nn.Module):

    def __init__(
            self,  in_channels, out_channels
    ) -> None:
        super(JointDiscriminator, self).__init__()
        self.Mf = torch.nn.Linear(out_channels, out_channels)
        self.Ms = torch.nn.Linear(in_channels, out_channels)  # feature size
        self.Mz = torch.nn.Linear(out_channels, out_channels)
        self.Mj = torch.nn.Bilinear(out_channels, out_channels, 1)

    def forward(self, positive_h, negative_h, values, values_1, values_2):
        z_pos = torch.sigmoid(self.Mf(values_1))
        z_neg = torch.sigmoid(self.Mf(values_2))
        z = torch.sigmoid(self.Ms(values))

        z_pos = torch.cat([z, z_pos], dim=-1)
        z_neg = torch.cat([z, z_neg], dim=-1)

        z_pos = torch.sigmoid(self.Mz(z_pos))
        z_neg = torch.sigmoid(self.Mz(z_neg))

        z_pos = torch.sigmoid(self.Mj(z_pos, positive_h))
        z_neg = torch.sigmoid(self.Mj(z_neg, negative_h))

        loss = - torch.log(torch.sigmoid(z_pos) + 1e-15).mean() - torch.log(1 - torch.sigmoid(z_neg) + 1e-15).mean()

        return loss
