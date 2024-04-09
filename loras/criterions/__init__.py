import torch

from .cemse import CEplusMSE
from .segmentation import MeanOverFramesAccuracy, EditDistance, F1Score

def calculate_metrics(logits, targets, prefix=None):

    mof = MeanOverFramesAccuracy()
    f1 = F1Score()
    edit = EditDistance(True)

    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)

    result = { 'mof': mof(predictions, targets), 'edit': edit(predictions, targets) }
    result.update(f1(predictions, targets))

    if prefix is not None:
        result = { f'{prefix}/{key}': val for key,val in result.items() }
    
    return result