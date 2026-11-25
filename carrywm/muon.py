"""SuperMimic-aligned Muon + Adam hybrid optimizer for V7.3-Ultra.

Hidden 2D linear layer weights use Muon (Newton-Schulz orthogonalization with
RMS-to-AdamW LR scaling), while input/output linear layers, biases, and LayerNorm
parameters use Adam. Falls back to an exact pure-PyTorch Muon implementation
when running on PyTorch builds earlier than 2.9 (e.g. 2.7.0 on 3090).
"""
from collections.abc import MutableMapping
import math

import torch
import torch.nn as nn
from torch.optim import Optimizer


EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    if ns_steps >= 100:
        raise ValueError("ns_steps must be less than 100")
    if grad.ndim != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("ns_coefficients must have exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if transposed:
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    a_dim, b_dim = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1.0, a_dim / b_dim))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(a_dim, b_dim))
    else:
        raise ValueError(f"Unsupported adjust_lr_fn {adjust_lr_fn!r}")
    return float(lr) * ratio


class _CompatMuon(Optimizer):
    """Exact pure-PyTorch implementation of PyTorch 2.9+ torch.optim.Muon."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ):
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= float(lr):
            raise ValueError(f"Learning rate should be >= 0, got {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0, got {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight_decay should be >= 0, got {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"Unsupported adjust_lr_fn {adjust_lr_fn!r}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon only supports 2D parameters, got shape {tuple(p.shape)}")

    def _init_group(self, group: MutableMapping):
        params_with_grad, grads, bufs = [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            params_with_grad.append(p)
            grads.append(p.grad)
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )
            bufs.append(state["momentum_buffer"])
        return params_with_grad, grads, bufs

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr_val = group["lr"].item() if torch.is_tensor(group["lr"]) else float(group["lr"])
            wd = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_coeffs = tuple(group["ns_coefficients"])
            ns_steps = int(group["ns_steps"])
            eps = float(group["eps"])
            adjust_lr_fn = group["adjust_lr_fn"]
            params_with_grad, grads, bufs = self._init_group(group)
            for p, grad, buf in zip(params_with_grad, grads, bufs):
                buf.lerp_(grad, 1.0 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf
                ortho = _zeropower_via_newtonschulz(update, ns_coeffs, ns_steps, eps)
                adjusted_lr = _adjust_lr(lr_val, adjust_lr_fn, p.shape)
                p.mul_(1.0 - lr_val * wd)
                p.add_(ortho.to(dtype=p.dtype), alpha=-adjusted_lr)
        return loss


Muon = getattr(torch.optim, "Muon", _CompatMuon)


def hidden_linear_weights(module: nn.Module) -> list[torch.nn.Parameter]:
    """Return interior 2D Linear weights in ``module`` while keeping input/output heads in Adam."""
    weights: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    if hasattr(module, "trunk") and isinstance(module.trunk, nn.Sequential) and hasattr(module, "robot"):
        # WorldModel splits its trunk from separate head linears (robot, box, contact), so
        # only trunk[0] is the input layer and all subsequent trunk Linears are hidden layers.
        trunk_linears = [m for m in module.trunk if isinstance(m, nn.Linear)]
        candidates = trunk_linears[1:]
        for layer in candidates:
            w = layer.weight
            if w.requires_grad and w.ndim == 2 and id(w) not in seen:
                weights.append(w)
                seen.add(id(w))
        return weights

    for seq in module.modules():
        if isinstance(seq, nn.Sequential):
            linears = [m for m in seq if isinstance(m, nn.Linear)]
            for layer in linears[1:-1]:
                w = layer.weight
                if w.requires_grad and w.ndim == 2 and id(w) not in seen:
                    weights.append(w)
                    seen.add(id(w))
    return weights


class _MergedStateView:
    """Read-only combined mapping view over Adam and Muon optimizer state dicts."""

    def __init__(self, adam: Optimizer, muon: Optimizer):
        self._adam = adam
        self._muon = muon

    def __getitem__(self, key):
        if key in self._muon.state:
            return self._muon.state[key]
        return self._adam.state[key]

    def __contains__(self, key):
        return key in self._adam.state or key in self._muon.state

    def __len__(self):
        return len(self._adam.state) + len(self._muon.state)

    def __iter__(self):
        yield from self._adam.state
        yield from self._muon.state

    def __bool__(self):
        return bool(self._adam.state) or bool(self._muon.state)


class HybridMuonAdam:
    """Single optimizer facade pairing Adam (first/last/1D params) with Muon (hidden matrices)."""

    def __init__(self, adam: Optimizer, muon: Optimizer):
        self.adam = adam
        self.muon = muon
        self._state_view = _MergedStateView(adam, muon)

    @property
    def param_groups(self):
        return [*self.adam.param_groups, *self.muon.param_groups]

    @property
    def state(self):
        return self._state_view

    def zero_grad(self, set_to_none: bool = True):
        self.adam.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.adam.step(closure)
        self.muon.step()
        return loss

    def state_dict(self):
        return {"optimizer": "muon", "adam": self.adam.state_dict(), "muon": self.muon.state_dict()}

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict) or state_dict.get("optimizer") != "muon":
            raise ValueError("Cannot load non-Muon optimizer checkpoint into HybridMuonAdam")
        self.adam.load_state_dict(state_dict["adam"])
        self.muon.load_state_dict(state_dict["muon"])

    def register_step_pre_hook(self, hook):
        return self.adam.register_step_pre_hook(hook)

    def register_step_post_hook(self, hook):
        return self.muon.register_step_post_hook(hook)


def build_module_optimizer(
    module: nn.Module,
    lr: float,
    *,
    betas: tuple[float, float] = (0.9, 0.999),
    optimizer: str = "adam",
    muon_weight_decay: float = 0.1,
    adam_kwargs: dict | None = None,
):
    """Build per-module optimizer: plain Adam or SuperMimic-aligned HybridMuonAdam."""
    extra = dict(adam_kwargs or {})
    if optimizer == "adam":
        return torch.optim.Adam(module.parameters(), lr=lr, betas=betas, **extra)
    if optimizer != "muon":
        raise ValueError(f"Unknown optimizer {optimizer!r}")
    hidden = hidden_linear_weights(module)
    if not hidden:
        raise ValueError(f"No interior hidden 2D Linear weights found in {type(module).__name__}")
    hidden_ids = {id(p) for p in hidden}
    rest = [p for p in module.parameters() if p.requires_grad and id(p) not in hidden_ids]
    adam = torch.optim.Adam(rest, lr=lr, betas=betas, **extra)
    muon_opt = Muon(
        hidden,
        lr=lr,
        weight_decay=float(muon_weight_decay),
        adjust_lr_fn="match_rms_adamw",
    )
    return HybridMuonAdam(adam, muon_opt)
