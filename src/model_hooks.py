from __future__ import annotations

from typing import Iterable

import torch


def load_hooked_transformer(model_name: str):
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(model_name, device="cpu", dtype=torch.float32)


def make_prompts(statements: Iterable[str], template: str) -> list[str]:
    return [template.format(statement=statement) for statement in statements]


@torch.no_grad()
def collect_resid_post_by_layer(model, prompts: list[str]) -> dict[int, torch.Tensor]:
    tokens = model.to_tokens(prompts, prepend_bos=True)
    _, cache = model.run_with_cache(tokens)
    result: dict[int, torch.Tensor] = {}
    for layer in range(model.cfg.n_layers):
        activations = cache[f"blocks.{layer}.hook_resid_post"]
        result[layer] = activations[:, -1, :].detach().cpu()
    return result


def token_id(model, text: str) -> int:
    ids = model.to_single_token(text)
    return int(ids)
