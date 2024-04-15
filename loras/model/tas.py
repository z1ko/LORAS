import torch
import torch.nn as nn
import lightning
import einops

from time import perf_counter_ns

from loras.model.temporal import LRUBlock, S4DBlock, MambaBlock
from loras.criterions import calculate_multi_metrics, calculate_multi_loss, log_multi_result

def build_categories_head(config):
    
    categories = []
    for category_name, num_classes in zip(config.categories, config.categories_num_classes):
        categories.append((category_name, num_classes))

    heads = nn.ModuleList()
    for _, num_classes in categories:
        heads.append(nn.Linear(config.model_dim, num_classes))

    return categories, heads

def build_temporal_model(config):
    if config.temporal_model == 'lru':
        return LRUBlock(
            input_dim=config.model_dim, 
            output_dim=config.model_dim, 
            state_dim=config.temporal_state_dim, 
            layers_count=config.temporal_layers_count, 
            dropout=config.dropout,
            phase_max=config.lru_max_phase,
            r_min=config.lru_min_radius,
            r_max=config.lru_max_radius,
        )
    elif config.temporal_model == 's4d':
        return S4DBlock(
            input_dim=config.model_dim, 
            output_dim=config.model_dim, 
            state_dim=config.temporal_state_dim, 
            layers_count=config.temporal_layers_count, 
            dropout=config.dropout
        )
    elif config.temporal_model == 'mamba':
        return MambaBlock(
            input_dim=config.model_dim, 
            output_dim=config.model_dim, 
            state_dim=config.temporal_state_dim, 
            layers_count=config.temporal_layers_count,
            expand=config.mamba_expand_factor,
            conv_size=config.mamba_conv_size
        )
    else:
        raise NotADirectoryError()

class input_module(nn.Module):
    def __init__(self, config, modality):
        super().__init__()
        self.modality = modality

        self.input_features = 0 
        if self.modality == 'embeddings':
            self.input_features = config.frame_features
        elif self.modality == 'poses':
            self.input_features = config.pose_joint_features * config.pose_joint_count
        else:
            raise NotImplementedError()

        self.model = nn.Sequential(
            nn.Linear(self.input_features, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        if self.modality == 'poses':
            x = einops.rearrange(x, 'B T J F -> B T (J F)')
        return self.model(x)

class LORASBase(lightning.LightningModule):
    """ Base class for all LORAS
    """

    def __init__(self, config):
        super().__init__()
        self.modality = config.modality
        self.bg_weight = config.background_class_weight
        self.alpha = config.cemse_alpha

    def split_targets_between_categories(self, targets):
        targets_verb, targets_noun = torch.split(targets, split_size_or_sections=1, dim=-1)
        targets_verb = torch.squeeze(targets_verb, dim=-1)
        targets_noun = torch.squeeze(targets_noun, dim=-1)
        return [ targets_verb, targets_noun ]
    
    def training_step(self, batch, _batch_idx):
        frames, poses, targets = batch
        targets = self.split_targets_between_categories(targets)

        # Output logits for each target category
        logits = self.forward(frames, poses)
        losses, combined_loss = calculate_multi_loss(logits, targets, self.categories, self.alpha, self.bg_weight)
        log_multi_result(losses, self.log, 'train')

        self.log('train/loss', combined_loss, on_epoch=True, on_step=False, prog_bar=True)
        return combined_loss

    def validation_step(self, batch, _batch_idx):
        assert(self.config.test_batch_size == 1)
        frames, poses, targets = batch
        frames_count = targets.shape[1]

        targets = self.split_targets_between_categories(targets)

        beg = perf_counter_ns()
        logits = self.forward(frames, poses)
        end = perf_counter_ns()

        # Framerate approximation
        # NOTE: works only because test_batch_size is 1
        elapsed_ms = (end - beg) * 1e-6
        self.log('val/elapsed(ms)', elapsed_ms / frames_count, on_step=False, on_epoch=True)
        self.log('val/fps', frames_count / (elapsed_ms * 1e-3), on_step=False, on_epoch=True)

        metrics = calculate_multi_metrics(logits, targets, self.categories)
        log_multi_result(metrics, self.log, 'val')

        losses, combined_loss = calculate_multi_loss(logits, targets, self.categories, self.alpha, self.bg_weight)
        log_multi_result(losses, self.log, 'val')

        self.log('val/loss', combined_loss, on_epoch=True, on_step=False, prog_bar=True)
        return combined_loss

    def predict_step(self, batch, _batch_idx):
        assert(self.config.test_batch_size == 1)
        frames, poses, targets = batch

        logits = self.forward(frames, poses)
        
        results = []
        for i in range(len(logits)):
            probabilities = torch.softmax(logits[i], dim=-1)
            results.append(torch.argmax(probabilities, dim=-1))

        targets = self.split_targets_between_categories(targets)
        losses, combined_loss = calculate_multi_loss(logits, targets, self.categories, self.alpha, self.bg_weight)

        return combined_loss, results

    def on_before_optimizer_step(self, optimizer):
        pass

    def configure_optimizers(self):
        params = list(self.parameters())
        optimizer = torch.optim.SGD(params=params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.config.scheduler_step, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

class LORAS(LORASBase):
    def __init__(self, config):
        super().__init__(config)

        # Store hyperparameters
        self.save_hyperparameters(vars(config))
        self.inference = False
        self.modality = config.modality
        self.config = config

        # Create output categories and heads
        self.categories, self.output_heads = build_categories_head(config)
        self.input_module = input_module(config, config.modality)
        self.temporal = build_temporal_model(config)

    def initialize_inference(self):
        self.temporal.initialize_inference()
        self.inference = True

    def forward_with_state(self, frames, _poses, state):
        x = self.input_module(frames)
        x, state = self.temporal.forward_with_state(x, state)

        outputs = []
        for head in self.output_heads:
            outputs.append(head(x))

        return outputs, state

    def forward(self, frames, poses):
        
        if self.modality == 'embeddings':
            x = self.input_module(frames)
        elif self.modality == 'poses':
            x = self.input_module(poses)
        else:
            raise NotImplementedError()
        
        x = self.temporal(x)

        # Calculate each category output
        outputs = []
        for head in self.output_heads:
            outputs.append(head(x))
        
        return outputs
    

class LORASFused(LORASBase):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Output module
        self.categories, self.output_heads = build_categories_head(config)

        self.input_embeddings = input_module(config, 'embeddings')
        self.input_poses = input_module(config, 'poses')

        self.temporal = LRUBlock(
            input_dim=config.model_dim, 
            output_dim=config.model_dim, 
            state_dim=config.temporal_state_dim, 
            layers_count=config.temporal_layers_count, 
            dropout=config.dropout,
            phase_max=config.lru_max_phase,
            r_min=config.lru_min_radius,
            r_max=config.lru_max_radius,
        )

        # Final merger between poses and frames
        self.final_mixer = nn.Sequential(
            nn.LayerNorm(config.model_dim * 2),
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

    def forward(self, frames, poses):

        # TODO: temporal frequency reduction
        f = self.input_embeddings(frames)

        p = self.input_poses(poses)
        p = self.temporal(p)
        
        x = torch.cat([p, f], dim=-1)
        x = self.final_mixer(x)

        # Calculate each category output
        outputs = []
        for head in self.output_heads:
            outputs.append(head(x))
        
        return outputs