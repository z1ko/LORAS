from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

from loras.model import LORAS
from loras.datasets import Assembly101
from loras.utils import load_args_and_config

config = load_args_and_config()
print(config)

dataset = Assembly101(config)
model = LORAS(config)


# Stop training if the validation loss doesn't decrease
early_stopping = EarlyStopping(monitor='val/loss_total', patience=3, mode='min')

logger = WandbLogger(name='LORAS', save_dir='runs')
trainer = Trainer(
    max_epochs=config.train_epochs,
    callbacks=[early_stopping], 
    logger=logger
)

trainer.fit(
    model, 
    train_dataloaders=dataset.train_dataloader(), 
    val_dataloaders=dataset.val_dataloader())