import torch

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

from loras.model import LORAS
from loras.datasets import Assembly101
from loras.utils import load_args_and_config

def run_model():
    return 0.0

config = load_args_and_config()
print(config)

dataset = Assembly101(config)
model = LORAS(config)

# Stop training if the validation loss doesn't decrease
early_stopping = EarlyStopping(monitor='val/loss', patience=20, mode='min')

logger = WandbLogger(name='LORAS', save_dir='runs')
trainer = Trainer(
    accumulate_grad_batches=config.accumulate_grad_batches,
    max_epochs=config.train_epochs,
    callbacks=[early_stopping], 
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
predictions = trainer.predict(model, dataloaders=dataset.val_dataloader())
final_score = sum(predictions) / len(predictions)
print(f'model score: {final_score}')
