import torch
from torch import nn

class SL1Loss(nn.Module):
    def __init__(self, ohem=False, topk=0.6):
        super(SL1Loss, self).__init__()
        self.ohem = ohem
        self.topk = topk
        self.loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, inputs, targets, mask):
        loss = self.loss(inputs[mask], targets[mask])

        if self.ohem:
            num_hard_samples = int(self.topk * loss.numel())
            loss, _ = torch.topk(loss.flatten(), 
                                 num_hard_samples)

        return torch.mean(loss)

loss_dict = {'sl1': SL1Loss}