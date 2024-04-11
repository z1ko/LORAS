import torch
import torch.nn as nn
import einops as ein
import math

# NOTE: Should use custom kernels
from loras.model.acceleration import associative_scan, binary_operator_diag
from loras.model.misc.glu import GLU

class LRU(nn.Module):
    """ Implementation of a Linear Recurrent Unit (LRU)
        https://arxiv.org/pdf/2303.06349.pdf
    """

    def __init__(
        self,
        state_dim,                  # The state dimension is the same as the input dimension and output dimension
        r_min=0.8,                  # Min. radius in the complex plane
        r_max=0.99,                 # Max. radius in the complex plane
        phase_max=math.pi * 2,      # Phase in the form of [0, phase_max]
        **kwargs
    ):
        super().__init__()
        self.inference = False

        self.state_dim = state_dim
        self.state = torch.complex(torch.zeros(state_dim), torch.zeros(state_dim))

        # Input to output, skip connection, implemented in the block
        # self.D = nn.Parameter(torch.randn([state_dim, state_dim]) / math.sqrt(state_dim))

        # Diagonal state matrix parameters
        u1 = torch.rand(state_dim)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (r_max + r_min) * (r_max - r_min) + r_min**2)))
        u2 = torch.rand(state_dim)
        self.theta_log = nn.Parameter(torch.log(phase_max * u2))

        # Diagonal state matrix and normalization factor
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))))

        # Input to state matrix
        B_re = torch.randn([state_dim, state_dim]) / math.sqrt(2 * state_dim)
        B_im = torch.randn([state_dim, state_dim]) / math.sqrt(2 * state_dim)
        self.B = nn.Parameter(torch.complex(B_re, B_im))

        # State to output matrix
        C_re = torch.randn([state_dim, state_dim]) / math.sqrt(state_dim)
        C_im = torch.randn([state_dim, state_dim]) / math.sqrt(state_dim)
        self.C = nn.Parameter(torch.complex(C_re, C_im))

    def forward(self, x):  # (B, L, F)
        self.state = self.state.to(self.B.device)

        # Istantiate diagonal state matrix
        L_mod = torch.exp(-torch.exp(self.nu_log))
        L_re = L_mod * torch.cos(torch.exp(self.theta_log))
        L_im = L_mod * torch.sin(torch.exp(self.theta_log))
        L_diag = torch.complex(L_re, L_im).to(self.B.device)

        # Istantiate normalization factor
        G_norm = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        B_norm = self.B * G_norm

        L_elems = L_diag.tile(x.shape[1], 1)
        B_elems = x.to(B_norm.dtype) @ B_norm.T

        def inner_state_fn(B_seq):
            return associative_scan(binary_operator_diag, (L_elems, B_seq))[1]

        inner_states = torch.vmap(inner_state_fn)(B_elems)
        return (inner_states @ self.C.T).real

    def initialize_inference(self):
        self.inference = True

        L_mod = torch.exp(-torch.exp(self.nu_log))
        L_re = L_mod * torch.cos(torch.exp(self.theta_log))
        L_im = L_mod * torch.sin(torch.exp(self.theta_log))

        self.L_diag_cache = torch.complex(L_re, L_im).to(self.B.device)
        G_norm = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        self.B_norm_cache = self.B * G_norm

    def forward_with_state(self, x, state):  # F, S dove F = S
        assert(self.inference)
        y = torch.zeros_like(x)
        state = self.L_diag_cache * state + self.B_norm_cache @ x.to(dtype=self.B.dtype)
        y = (self.C @ state).real
        return y, state

class LRUBlock(nn.Module):
    """LRU block with gated linear unit as channel mixer, dropout and normalization
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        state_dim,
        layers_count,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.inference = False

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.layers_count = layers_count
        self.dropout = dropout

        self.input_proj = nn.Linear(self.input_dim, self.state_dim)
        self.output_proj = nn.Linear(self.state_dim, self.output_dim)

        self.norms = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.glus = nn.ModuleList()

        # Create all layers
        for _ in range(self.layers_count):
            self.norms.append(nn.LayerNorm(self.state_dim))
            self.layers.append(LRU(self.state_dim, **kwargs))
            self.glus.append(GLU(self.state_dim, self.dropout))

    def initialize_inference(self):
        self.inference = True
        for lru in self.layers:
            lru.initialize_inference()

    def forward(self, x):
        x = self.input_proj(x)

        for norm, lru, glu in zip(self.norms, self.layers, self.glus):
            residual = x
            x = norm(x)
            x = lru(x)
            x = glu(x)
            x = x + residual

        x = self.output_proj(x)
        return x


    def forward_with_state(self, x, state):
        assert(self.inference)

        x = self.input_proj(x)
        
        for norm, lru, glu in zip(self.norms, self.layers, self.glus):
            residual = x
            x = norm(x)
            x, state = lru.forward_with_state(x, state)
            x = glu(x)
            x = x + residual

        x = self.output_proj(x)
        return x, state