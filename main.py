import torch
import wandb
import os

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

from loras.model import LORAS, LORASFused
from loras.datasets import Assembly101
from loras.utils import load_args_and_config

def run_model():
    return 0.0

config = load_args_and_config()
print(config)

dataset = Assembly101(config)

if config.modality == 'fused':
    model = LORASFused(config)
else:
    model = LORAS(config)

# Stop training if the validation loss doesn't decrease
#early_stopping = EarlyStopping(monitor='val/loss', patience=25, mode='min')

logger = WandbLogger(name='LORAS', save_dir='runs')
trainer = Trainer(
    accumulate_grad_batches=config.accumulate_grad_batches,
    max_epochs=config.train_epochs,
#    callbacks=[early_stopping], 
    logger=logger
)

# Test framerate with test on a 2 frames tensor to simulate inference
#elapsed, fps = model.benchmark_framerate()
#print(f'elapsed ms for frame: {elapsed}, fps: {fps}')

trainer.fit(
    model, 
    train_dataloaders=dataset.train_dataloader(), 
    val_dataloaders=dataset.val_dataloader()
)

# Final evaluation to use with facebook ax
#predictions = trainer.predict(model, dataloaders=dataset.val_dataloader())

experiment_name = wandb.run.name
output_path = os.path.join('./runs/predictions', experiment_name)
os.makedirs(output_path)

with open(output_path + '/predictions.txt', 'wt') as f:

    # Header
    for category, num_classes in zip(config.categories, config.categories_num_classes):
        f.write(f'{category}:{num_classes}\n')

    f.write('================================')

    for frames, poses, targets in dataset.val_dataloader():
        assert(targets.shape[0] == 1)
        frames_count = targets.shape[1]

        f.write(f'{frames_count}\n')
        for i, label in enumerate(targets[0]):
            end = ',' if i != len(frames_count) else ''
            f.write(f'{label.item()}{end}')

        loss, outputs = model.predict_step((frames, poses, targets)) 
        for i, output in enumerate(outputs[0]):
            end = ',' if i != len(frames_count) else ''
            f.write(f'{output.item()}{end}')
