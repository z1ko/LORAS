from loras.utils import load_args_and_config
from loras.model.tas import LORAS
from loras.datasets.assembly import Assembly101Dataset

import torch
from torch.profiler import ProfilerActivity

config = load_args_and_config('configs/memory.yaml')
print(config)

dataset = Assembly101Dataset('validation', config)
model = LORAS(config)

model.initialize_inference()
state = model.create_state()

with torch.no_grad():
    with torch.profiler.profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        embeddings, poses, labels = next(iter(dataset))
        with torch.profiler.record_function('model inference'):
            model.forward_with_state(poses[:1, ...], state)

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))