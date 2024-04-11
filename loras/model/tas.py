import torch
import torch.nn as nn
import lightning

from time import perf_counter_ns

from loras.model.temporal import LRUBlock
from loras.criterions import CEplusMSE, calculate_multi_metrics, calculate_multi_loss, log_multi_result

def build_categories_head(config):
    
    categories = []
    for category_name, num_classes in zip(config.categories, config.categories_num_classes):
        categories.append(category_name, num_classes)

    heads = nn.ModuleList()
    for _, num_classes in categories:
        heads.append(nn.Linear(config.model_dim, num_classes))

    return categories, heads


class LORAS(lightning.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create output categories
        self.categories = []
        for category_name, num_classes in zip(config.categories, config.categories_num_classes):
            self.categories.append(category_name, num_classes)

        # Training loss
        self.ce_plus_mse = CEplusMSE(config.classes, config.cemse_alpha)

        # Initial embeddings reduction
        self.input_module = nn.Sequential(
            nn.Linear(config.frame_features, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.temporal = LRUBlock(
            input_dim=config.model_dim, 
            output_dim=config.model_dim, 
            state_dim=config.temporal_state_dim, 
            layers_count=config.temporal_layers_count, 
            dropout=config.dropout
        )

        # Different output heades for the categories
        self.output_heads = nn.ModuleList()
        for _, num_classes in zip(config.categories, config.categories_num_classes):
            self.output_heads.append(nn.Linear(config.model_dim, num_classes))


    def forward(self, frames, _poses):
        x = self.input_module(frames)
        x = self.temporal(x)

        # Calculate each category output
        outputs = []
        for i, head in enumerate(self.output_heads):
            outputs[i] = head(x)
        
        return outputs

    def training_step(self, batch, _batch_idx):
        frames, poses, targets = batch
        targets = torch.split(targets, split_size_or_sections=1, dim=-1)
        targets.squeeze_()

        # Output logits for each target category
        logits = self.forward(frames, poses)
        losses, combined_loss = calculate_multi_loss(logits, targets, self.categories)
        log_multi_result(losses, self.logger, 'train')

        self.log('train/loss', combined_loss, on_epoch=True, on_step=False, prog_bar=True)
        return combined_loss

    def validation_step(self, batch, _batch_idx):
        assert(self.config.test_batch_size == 1)
        
        frames, poses, targets, = batch
        targets = torch.split(targets, split_size_or_sections=1, dim=-1)
        targets.squeeze_()

        beg = perf_counter_ns()
        logits = self.forward(frames, poses)
        end = perf_counter_ns()
        elapsed = (end - beg)

        # Framerate approximation
        # NOTE: works only because test_batch_size is 1
        self.log('val/elapsed(ms)', (elapsed * 1e-6) / targets.shape[-1], on_step=False, on_epoch=True)
        self.log('val/fps', targets.shape[-1] / (elapsed * 1e-9), on_step=False, on_epoch=True)

        metrics = calculate_multi_metrics(logits, targets, self.categories)
        log_multi_result(metrics, self.logger, 'val')

        losses, combined_loss = calculate_multi_loss(logits, targets, self.categories)
        log_multi_result(losses, self.logger, 'val')
        return combined_loss

    def on_before_optimizer_step(self, optimizer):
        pass

    def configure_optimizers(self):
        params = list(self.parameters())
        optimizer = torch.optim.SGD(params=params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.config.scheduler_step, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    

class LORASFused(lightning.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Output module
        self.categories, self.output_heads = build_categories_head(config)

        # Initial pose data reduction
        self.pose_input_dim = config.pose_joint_features * config.pose_join_count
        self.pose_input_module = nn.Sequential(
            nn.Linear(self.pose_input_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Initial frames data reduction
        self.frame_input_module = nn.Sequential(
            nn.Linear(config.frame_features, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Final merger between poses and frames
        self.final_mixer = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

    def forward(self, frames, poses):

        # TODO: temporal frequency reduction
        f = self.frame_input_module(frames)

        p = self.pose_input_module(poses)
        p = self.temporal(p)
        
        x = torch.cat([p, f], dim=-1)
        x = self.final_mixer(x)

        # Calculate each category output
        outputs = []
        for i, head in enumerate(self.output_heads):
            outputs[i] = head(x)
        
        return outputs