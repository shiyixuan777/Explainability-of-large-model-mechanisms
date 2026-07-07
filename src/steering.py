from __future__ import annotations

import torch


def make_add_direction_hook(direction: torch.Tensor, alpha: float):
    def hook(resid, _hook):
        resid[:, -1, :] = resid[:, -1, :] + alpha * direction.to(resid.device)
        return resid

    return hook


@torch.no_grad()
def run_steered_logits(model, tokens: torch.Tensor, layer: int, direction: torch.Tensor, alpha: float):
    hook_name = f"blocks.{layer}.hook_resid_post"
    return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_add_direction_hook(direction, alpha))])
