import torch

from .cemse import CEplusMSE
from .segmentation import MeanOverFramesAccuracy, EditDistance, F1Score

def log_multi_result(result, logger_fn, mode):
    """ Log a multi result from losses or metrics
    """
    for category_name, category_data in result.items():
        prefix = f'{category_name}/{mode}'
        for metric_name, metric_value in category_data.items():
            logger_fn(f'{prefix}/{metric_name}', metric_value, on_epoch=True, on_step=False, prog_bar=False)


def calculate_multi_loss(logits, targets, categories):
    """ Calculate ce+mse multiloss between different target categories
            logits:     an array of tensors, each of the shape [batch_size, seq_len, logits]
            targets:    an array of tensors, each of the shape [batch_size, seq_len]
            categories: [('verb', 25), ('noun', 90)]
    """

    result = { }
    for category_name, _ in categories:
        result[category_name] = {}

    combined = 0.0
    for logit, target, category in zip(logits, targets, categories):
        category_name, num_classes = category

        loss = CEplusMSE(num_classes, alpha=0.17)
        category_result = loss(logit, target)
        result[category_name] = category_result
        
        # Accumulated loss between all categories
        combined += category_result['loss_total']

    return result, combined

def calculate_multi_metrics(logits, targets, categories):
    """ Calculate ce+mse multiloss between different target categories
    """

    result = {}
    for category_name, _ in categories:
        result[category_name] = {}

    for logit, target, category in zip(logits, targets, categories):
        category_name, _ = category
        category_result = calculate_metrics(logit, target)
        result[category_name] = category_result
        
    return result

def calculate_metrics(logits, targets):
    """ logits:  [batch_size, seq_len, logits]
        targets: [batch_size, seq_len]
    """

    mof = MeanOverFramesAccuracy()
    f1 = F1Score()
    edit = EditDistance(True)

    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)

    result = { 'mof': mof(predictions, targets), 'edit': edit(predictions, targets) }
    result.update(f1(predictions, targets))
    return result