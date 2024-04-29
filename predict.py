from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from loras.datasets import Assembly101Dataset
from loras.utils import load_args_and_config
from loras.model import LORAS
from tqdm import tqdm

import torch

config = load_args_and_config()
print(config)

dataset = Assembly101Dataset(mode='validation', config=config)
dataloader = DataLoader(dataset, 1, False)

model = LORAS(config, dataset.weights)
model.load_from_checkpoint('...')

with open('predictions.txt', 'rt') as f:
    for embeddings, poses, labels in tqdm(dataloader, total=len(dataset)):

        logits = model.forward(embeddings, poses)
        result = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)

        # Write predictions
        f.write(len(logits))
        f.writelines([
            ','.join(list(map(lambda x: str(int(x.item())), labels))),
            ','.join(list(map(lambda x: str(int(x.item())), result)))
        ])
