import torch

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

from loras.model import LORAS, LORASFused
from loras.datasets import Assembly101
from loras.utils import load_args_and_config, output_model_predictions

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

#torch.cuda.memory._record_memory_history()

trainer.fit(
    model,
    train_dataloaders=dataset.train_dataloader(),
    val_dataloaders=dataset.val_dataloader()
)

#torch.cuda.memory._dump_snapshot("run_memory_usage.pickle")

# Final evaluation to use with facebook ax
#predictions = trainer.predict(model, dataloaders=dataset.val_dataloader())

#predictions_path = './predictions/' + logger.experiment.id
#output_model_predictions(predictions_path, model, dataset.val_dataloader(), config)