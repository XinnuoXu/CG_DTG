import json
import torch
from torch import nn
from torch.nn import init
from sparsemax import Sparsemax

class WeightedAttention(nn.Module):
    def __init__(self, dim, eps = 1e-8, softmax_dim = 1, weighted_mean_dim = 2):
        super().__init__()
        self.norm_input = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.eps = eps
        self.scale = dim ** -0.5
        self.softmax_dim = softmax_dim
        self.weighted_mean_dim = weighted_mean_dim

    def forward(self, inputs, context):

        inputs = self.norm_input(inputs)
        context = self.norm_context(context)

        q = self.to_q(inputs)
        k = self.to_k(context)
        v = self.to_v(context)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim = self.softmax_dim) + self.eps
        attn = attn / attn.sum(dim = self.weighted_mean_dim, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)
        return updates

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class GatedResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.fn = fn
    def forward(self, *args):
        inputs = args[0]
        b, _, d = inputs.shape

        updates = self.fn(*args)

        inputs = self.gru(
            updates.reshape(-1, d),
            inputs.reshape(-1, d)
        )
        return inputs.reshape(b, -1, d)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = max(dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)

class SlotAttentionExperimental(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        scale = dim ** -0.5
        self.num_slots = num_slots
        self.iters = iters

        self.norm_inputs = nn.LayerNorm(dim)

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.slots_to_inputs_attn = GatedResidual(dim, WeightedAttention(dim, eps = eps))
        self.slots_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))

        self.inputs_to_slots_attn = GatedResidual(dim, WeightedAttention(dim, eps = eps, softmax_dim = 2, weighted_mean_dim = 1))
        self.inputs_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))

    def forward(self, inputs, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_inputs(inputs)

        for _ in range(self.iters):
            slots = self.slots_to_inputs_attn(slots, inputs)
            slots = self.slots_ff(slots)

            inputs = self.inputs_to_slots_attn(inputs, slots)
            inputs = self.inputs_ff(inputs)

        return slots, inputs


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim, eps=1e-6)
        self.norm_slots  = nn.LayerNorm(dim, eps=1e-6)
        self.norm_pre_ff = nn.LayerNorm(dim, eps=1e-6)

        #self.max_function = nn.Softmax(dim=1)
        self.max_function = Sparsemax(dim=1)

    def forward(self, inputs, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        
        #inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn(mu.shape, device = device)

        for _ in range(self.iters):
            slots_prev = slots

            #slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = self.max_function(dots) + self.eps

            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            #slots = slots + self.mlp(self.norm_pre_ff(slots))

        q = self.to_q(slots)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = self.max_function(dots)

        return slots, attn, dots



class SoftKMeans(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        #self.max_function = Sparsemax(dim=1)
        #self.max_function = Sparsemax(dim=-1)
        self.max_function = nn.Softmax(dim=-1)

    def forward(self, inputs, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        #mu = torch.mean(inputs, dim=1).unsqueeze(1).repeat(1, n_s, 1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn(mu.shape, device = device)

        debug_obj = {'inputs':inputs.tolist(), 'slots':slots.tolist()}
        print (json.dumps(debug_obj))

        for _ in range(self.iters):
            #dots = torch.einsum('bid,bjd->bij', slots, inputs) * self.scale
            slots = slots.contiguous()
            dist = torch.cdist(slots, inputs, p=2)
            scores = -dist ** 2
            attn = self.max_function(scores) + self.eps
            #attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', inputs, attn)
            slots = updates

            debug_obj = {'inputs':inputs.tolist(), 'slots':slots.tolist()}
            print (json.dumps(debug_obj))

        dots = torch.einsum('bid,bjd->bij', slots, inputs) * self.scale
        #attn = self.max_function(dots)

        return slots, attn, dots
