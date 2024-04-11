from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

# NOTE: used for training
class CEplusMSE(nn.Module):
    """
    Loss from MS-TCN paper. CrossEntropy + MSE
    https://arxiv.org/abs/1903.01945
    """

    def __init__(self, num_classes, alpha=0.17):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.classes = num_classes
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        :param logits:  [batch_size, seq_len, logits]
        :param targets: [batch_size, seq_len]
        """

        logits = einops.rearrange(logits, 'batch_size seq_len logits -> batch_size logits seq_len')
        loss = { }

        # Frame level classification
        loss['loss_ce'] = self.ce(
            einops.rearrange(logits, "batch_size logits seq_len -> (batch_size seq_len) logits"),
            einops.rearrange(targets, "batch_size seq_len -> (batch_size seq_len)")
        )

        # Neighbour frames should have similar values
        loss['loss_mse'] = torch.mean(torch.clamp(self.mse(
            F.log_softmax(logits[:, :, 1:], dim=1),
            F.log_softmax(logits.detach()[:, :, :-1], dim=1)
        ), min=0.0, max=160.0))

        loss['loss_total'] = loss['loss_ce'] + self.alpha * loss['loss_mse']
        return loss