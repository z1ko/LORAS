import torch
from time import perf_counter_ns

# Simulate a performant system with float32 operations
# state is 1024 complex, we can use 2048 float values

L_diag = torch.zeros(2048, dtype=torch.float32)
B_norm = torch.zeros((2048, 2048), dtype=torch.float32)
C = torch.zeros((2048, 2048), dtype=torch.float32)

state = torch.zeros(2048, dtype=torch.float32)
x = torch.zeros(2048, dtype=torch.float32)

elapsed_total = 0.0
fps_total = 0.0

for i in range(1000):
    beg = perf_counter_ns()
    y = torch.zeros_like(x)
    state = L_diag * state + B_norm @ x
    y = (C @ state).real
    end = perf_counter_ns()

    elapsed_ms = ((end - beg) * 1e-6)
    elapsed_total += elapsed_ms
    fps_total += 1.0 / (elapsed_ms * 1e-3)

elapsed_total /= 1000.0
fps_total /= 1000.0

print(elapsed_total, fps_total)