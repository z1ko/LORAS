import torch
import torch.nn as nn
import lightning

from time import perf_counter_ns

from loras.model.temporal import LRUBlock
from loras.model.misc import GLU
from loras.criterions import CEplusMSE, calculate_metrics

class LORAS(lightning.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Training loss
        self.ce_plus_mse = CEplusMSE(config.classes, config.cemse_alpha)

        # Initial embeddings reduction
        self.input_module = nn.Sequential(
            nn.LayerNorm(config.frame_features),
            nn.Linear(config.frame_features, config.model_dim),
            #nn.Dropout(config.dropout),
        )

        self.temporal = LRUBlock(
            input_dim=config.model_dim, 
            output_dim=config.model_dim, 
            state_dim=config.temporal_state_dim, 
            layers_count=config.temporal_layers_count, 
            dropout=config.dropout
        )

        self.output_module = nn.Sequential(
            nn.Linear(config.model_dim, config.classes)
        )

    def forward(self, frames, _poses):
        x = self.input_module(frames)
        x = self.temporal(x)
        x = self.output_module(x)
        return x

    def training_step(self, batch, _batch_idx):
        frames, poses, targets = batch

        logits = self.forward(frames, poses)

        metrics = calculate_metrics(logits, targets, prefix='train')
        self.log_dict(metrics, on_step=False, on_epoch=True)

        loss = self.ce_plus_mse(logits, targets)
        loss = { f'train/{k}': v for k, v in loss.items() }
        self.log_dict(loss, on_step=False, on_epoch=True)

        self.log('train/loss', loss['train/loss_total'], prog_bar=True, on_step=False, on_epoch=True)
        return loss['train/loss_total']

    def validation_step(self, batch, _batch_idx):
        frames, poses, targets, = batch

        beg = perf_counter_ns()
        logits = self.forward(frames, poses)
        end = perf_counter_ns()

        # Framerate approximation
        # NOTE: works only because test_batch_size is 1
        assert(self.config.test_batch_size == 1)
        elapsed = (end - beg)
        self.log('val/elapsed(ms)', (elapsed * 1e-6) / targets.shape[-1], on_step=True)
        self.log('val/fps', targets.shape[-1] / (elapsed * 1e-9), on_step=True)

        metrics = calculate_metrics(logits, targets, prefix='val')
        self.log_dict(metrics, on_step=False, on_epoch=True)

        loss = self.ce_plus_mse(logits, targets)
        loss = { f'val/{k}': v for k, v in loss.items() }
        self.log_dict(loss, on_step=False, on_epoch=True)

        self.log('val/loss', loss['val/loss_total'], prog_bar=True, on_step=False, on_epoch=True)
        return loss['val/loss_total']

    def on_before_optimizer_step(self, optimizer):
        pass

    def configure_optimizers(self):
        params = list(self.parameters())
        optimizer = torch.optim.SGD(params=params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.config.scheduler_step, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}