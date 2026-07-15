from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from src.dataset import filter_dataset, load_dataset
from src.model_hooks import load_hooked_transformer, make_prompts
from src.truth_direction import logit_diff


@dataclass(frozen=True)
class ReadoutSetting:
    prompt_name: str
    verbalizer: str
    true_token: str
    false_token: str
    zero_shot_template: str
    few_shot_template: str


SETTINGS = [
    ReadoutSetting(
        prompt_name="answer_true_false",
        verbalizer="lower_true_false",
        true_token=" true",
        false_token=" false",
        zero_shot_template="Statement: {statement}\nAnswer true or false:",
        few_shot_template=(
            "Statement: The capital of France is Paris.\nAnswer true or false: true\n"
            "Statement: The capital of France is Rome.\nAnswer true or false: false\n"
            "Statement: {statement}\nAnswer true or false:"
        ),
    ),
    ReadoutSetting(
        prompt_name="statement_is",
        verbalizer="lower_true_false",
        true_token=" true",
        false_token=" false",
        zero_shot_template="Statement: {statement}\nThe statement is",
        few_shot_template=(
            "Statement: The capital of France is Paris.\nThe statement is true.\n"
            "Statement: The capital of France is Rome.\nThe statement is false.\n"
            "Statement: {statement}\nThe statement is"
        ),
    ),
    ReadoutSetting(
        prompt_name="answer_True_False",
        verbalizer="title_true_false",
        true_token=" True",
        false_token=" False",
        zero_shot_template="Statement: {statement}\nAnswer True or False:",
        few_shot_template=(
            "Statement: The capital of France is Paris.\nAnswer True or False: True\n"
            "Statement: The capital of France is Rome.\nAnswer True or False: False\n"
            "Statement: {statement}\nAnswer True or False:"
        ),
    ),
    ReadoutSetting(
        prompt_name="answer_yes_no",
        verbalizer="yes_no",
        true_token=" yes",
        false_token=" no",
        zero_shot_template="Statement: {statement}\nIs this statement true? Answer yes or no:",
        few_shot_template=(
            "Statement: The capital of France is Paris.\nIs this statement true? Answer yes or no: yes\n"
            "Statement: The capital of France is Rome.\nIs this statement true? Answer yes or no: no\n"
            "Statement: {statement}\nIs this statement true? Answer yes or no:"
        ),
    ),
    ReadoutSetting(
        prompt_name="statement_correct",
        verbalizer="correct_incorrect",
        true_token=" correct",
        false_token=" incorrect",
        zero_shot_template="Statement: {statement}\nThe statement is",
        few_shot_template=(
            "Statement: The capital of France is Paris.\nThe statement is correct.\n"
            "Statement: The capital of France is Rome.\nThe statement is incorrect.\n"
            "Statement: {statement}\nThe statement is"
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["all", "capital"],
        help="Domains to evaluate. Use 'all' for the full filtered dataset.",
    )
    parser.add_argument("--out", default="figures/output_readout_baselines.csv")
    return parser.parse_args()


def single_token_id(model, text: str) -> tuple[int | None, list[int]]:
    token_ids = model.tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        return None, token_ids
    return int(token_ids[0]), [int(token_ids[0])]


def evaluate_readout(
    model,
    data: pd.DataFrame,
    domain_name: str,
    setting: ReadoutSetting,
    *,
    shots: int,
) -> dict[str, object]:
    template = setting.zero_shot_template if shots == 0 else setting.few_shot_template
    true_token_id, true_token_ids = single_token_id(model, setting.true_token)
    false_token_id, false_token_ids = single_token_id(model, setting.false_token)

    base_row: dict[str, object] = {
        "model": model.cfg.model_name,
        "domain": domain_name,
        "prompt_name": setting.prompt_name,
        "verbalizer": setting.verbalizer,
        "shots": shots,
        "true_token_text": setting.true_token,
        "false_token_text": setting.false_token,
        "true_token_ids": " ".join(str(item) for item in true_token_ids),
        "false_token_ids": " ".join(str(item) for item in false_token_ids),
        "single_token_readout": true_token_id is not None and false_token_id is not None,
        "n_rows": len(data),
        "n_true": int((data["label"] == 1).sum()),
        "n_false": int((data["label"] == 0).sum()),
    }

    if true_token_id is None or false_token_id is None:
        return {
            **base_row,
            "accuracy_from_logit_sign": float("nan"),
            "auc": float("nan"),
            "separability_auc": float("nan"),
            "predicted_true_rate": float("nan"),
            "mean_true_minus_false_logit_diff": float("nan"),
            "mean_diff_on_true_examples": float("nan"),
            "mean_diff_on_false_examples": float("nan"),
            "std_true_minus_false_logit_diff": float("nan"),
        }

    prompts = make_prompts(data["statement"].tolist(), template)
    labels = data["label"].to_numpy().astype(int)
    tokens = model.to_tokens(prompts, prepend_bos=True)
    with torch.no_grad():
        logits = model(tokens)
    diffs = logit_diff(logits, true_token_id, false_token_id).detach().cpu().numpy()
    preds = (diffs > 0).astype(int)
    auc = float(roc_auc_score(labels, diffs))

    return {
        **base_row,
        "accuracy_from_logit_sign": float(accuracy_score(labels, preds)),
        "auc": auc,
        "separability_auc": max(auc, 1.0 - auc),
        "predicted_true_rate": float(preds.mean()),
        "mean_true_minus_false_logit_diff": float(diffs.mean()),
        "mean_diff_on_true_examples": float(diffs[labels == 1].mean()),
        "mean_diff_on_false_examples": float(diffs[labels == 0].mean()),
        "std_true_minus_false_logit_diff": float(diffs.std()),
    }


def main() -> None:
    args = parse_args()
    full_data = filter_dataset(load_dataset(args.data), language=args.language)
    model = load_hooked_transformer(args.model)

    rows: list[dict[str, object]] = []
    for domain in args.domains:
        if domain == "all":
            data = full_data
            domain_name = "all"
        else:
            data = filter_dataset(full_data, domain=domain)
            domain_name = domain
        print(f"Evaluating output readout baselines on domain={domain_name}, rows={len(data)}")
        for setting in SETTINGS:
            for shots in [0, 2]:
                row = evaluate_readout(model, data, domain_name, setting, shots=shots)
                rows.append(row)
                print(
                    f"domain={domain_name:<8} prompt={setting.prompt_name:<20} "
                    f"verbalizer={setting.verbalizer:<18} shots={shots} "
                    f"acc={row['accuracy_from_logit_sign']:.3f} "
                    f"auc={row['auc']:.3f} pred_true={row['predicted_true_rate']:.3f}"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved output readout baselines to {out_path}")


if __name__ == "__main__":
    main()
