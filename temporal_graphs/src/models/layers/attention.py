import torch


class CombineAttention(torch.nn.Module):

    def __init__(
            self, out_channels, number_of_relations
    ) -> None:
        super(CombineAttention, self).__init__()
        self.out_channels = out_channels
        self.w_list = torch.nn.ModuleList(
            [torch.nn.Linear(out_channels, out_channels, bias=False) for _ in range(number_of_relations)])
        self.y_list = torch.nn.ModuleList([torch.nn.Linear(out_channels, 1) for _ in range(number_of_relations)])

    def froward(self, h_list):
        h_combine_list = []

        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)

        score = torch.cat(h_combine_list, -1)
        score = torch.tanh(score)
        score = torch.tanh(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h
