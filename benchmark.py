import torch
import argparse

from loras.utils import load_args_and_config
from loras.model.tas import LORAS
from time import perf_counter_ns

REPETITIONS = 1000

def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--chain', action='store_true')
    parser.add_argument('--model_size', type=int)
    parser.add_argument('--temporal_state_dim', type=int)
    parser.add_argument('--temporal_layers_count', type=int)


config = load_args_and_config('configs/benchmark.yaml', setup=setup_parser)
if not config.chain:
    print(config)

torch.cuda.reset_max_memory_allocated()
model = LORAS(config, None)
model = model.cuda()
model.initialize_inference()
memory = torch.cuda.max_memory_allocated()

model_size_megabytes = memory / 1e6
if not config.chain:
    print('model_size_allocated: ', model_size_megabytes, '(MB)')

# Simulate a performant system with float32 operations
# state is 1024 complex, we can use 2048 float values

state = torch.zeros((config.temporal_layers_count, config.temporal_state_dim), dtype=torch.complex64)
state = state.cuda()

state_size_kilobytes = state.numel() * 8 / 1e3
if not config.chain:
    print('state size: ', state_size_kilobytes, '(KB)')

if config.modality == 'poses':
    x = torch.zeros((42, 3), dtype=torch.float32)
else:
    x = torch.zeros((2048,), dtype=torch.float32)
x = x.cuda()

elapsed_total = 0.0
fps_total = 0.0

for i in range(REPETITIONS):

    beg = perf_counter_ns()
    model.forward_with_state(x, state)
    end = perf_counter_ns()

    elapsed_ms = ((end - beg) * 1e-6)
    elapsed_total += elapsed_ms
    fps_total += 1.0 / (elapsed_ms * 1e-3)

elapsed_total /= float(REPETITIONS)
fps_total /= float(REPETITIONS)

if not config.chain:
    print(f'elapsed: {elapsed_total} (ms)')
    print(f'fps: {fps_total}')

if config.chain:
    print(f'{config.modality} {config.temporal_state_dim} {config.temporal_layers_count} {config.model_dim} {model_size_megabytes} {state_size_kilobytes} {elapsed_total} {fps_total}')