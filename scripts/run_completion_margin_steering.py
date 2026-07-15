from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts
from src.truth_direction import probe_direction


CAPITAL_RE = re.compile(r"^The capital of (?P<country>.+?) is (?P<capital>.+?)\.$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/capital_balanced.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default="capital_balanced")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--alphas", nargs="+", type=float, default=[-4, -2, -1, 0, 1, 2, 4])
    parser.add_argument(
        "--position-mode",
        choices=["all", "prompt-final-only", "completion-internal-only"],
        default="all",
        help="Residual positions where the steering vector is injected during teacher-forced completion scoring.",
    )
    parser.add_argument("--out-details", default="figures/completion_margin_steering_details.csv")
    parser.add_argument("--out-summary", default="figures/completion_margin_steering_summary.csv")
    parser.add_argument(
        "--from-details",
        action="store_true",
        help="Rebuild the summary from --out-details without rerunning the model.",
    )
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def parse_capital_statement(statement: str) -> tuple[str, str]:
    match = CAPITAL_RE.match(statement)
    if match is None:
        raise ValueError(f"Cannot parse capital statement: {statement}")
    return match.group("country"), match.group("capital")


def build_country_frame(data: pd.DataFrame) -> pd.DataFrame:
    parsed = data["statement"].map(parse_capital_statement)
    data = data.copy()
    data["country"] = [item[0] for item in parsed]
    data["stated_capital"] = [item[1] for item in parsed]

    true_rows = data[data["label"] == 1].drop_duplicates("country")
    false_rows = data[data["label"] == 0].drop_duplicates("country")
    true_capitals = true_rows.set_index("country")["stated_capital"].to_dict()
    false_capitals = false_rows.set_index("country")["stated_capital"].to_dict()
    pair_ids = true_rows.set_index("country")["pair_id"].to_dict()
    missing = sorted((set(data["country"]) - set(true_capitals)) | (set(data["country"]) - set(false_capitals)))
    if missing:
        raise ValueError(f"Countries missing true/false capital rows: {missing}")

    rows = []
    for country in sorted(true_capitals):
        rows.append(
            {
                "country": country,
                "pair_id": pair_ids[country],
                "correct_capital": true_capitals[country],
                "false_capital": false_capitals[country],
                "completion_prompt": f"The capital of {country} is",
            }
        )
    return pd.DataFrame(rows)


def make_position_steering_hook(direction: torch.Tensor, alpha: float, spans: list[tuple[int, int]]):
    def hook(resid, hook):
        if alpha != 0:
            direction_on_device = direction.to(resid.device)
            for batch_idx, (start_pos, end_pos) in enumerate(spans):
                resid[batch_idx, start_pos:end_pos, :] = resid[batch_idx, start_pos:end_pos, :] + (
                    alpha * direction_on_device
                )
        return resid

    return hook


def make_steering_spans(
    prompt_lens: list[int],
    full_lens: list[int],
    *,
    position_mode: str,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for prompt_len, full_len in zip(prompt_lens, full_lens):
        if position_mode == "all":
            spans.append((prompt_len - 1, full_len - 1))
        elif position_mode == "prompt-final-only":
            spans.append((prompt_len - 1, prompt_len))
        elif position_mode == "completion-internal-only":
            spans.append((prompt_len, full_len - 1))
        else:
            raise ValueError(f"Unknown position mode: {position_mode}")
    return spans


@torch.no_grad()
def completion_logprobs(
    model,
    prompts: list[str],
    completions: list[str],
    *,
    direction: torch.Tensor | None = None,
    alpha: float = 0.0,
    layer: int | None = None,
    position_mode: str = "all",
    batch_size: int = 16,
) -> list[tuple[float, int]]:
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")

    results: list[tuple[float, int]] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_completions = completions[start : start + batch_size]
        prompt_lens = [int(model.to_tokens(prompt, prepend_bos=True).shape[1]) for prompt in batch_prompts]
        full_texts = [prompt + completion for prompt, completion in zip(batch_prompts, batch_completions)]
        full_lens = [int(model.to_tokens(text, prepend_bos=True).shape[1]) for text in full_texts]
        completion_token_counts = [full_len - prompt_len for prompt_len, full_len in zip(prompt_lens, full_lens)]
        bad = [completion for completion, count in zip(batch_completions, completion_token_counts) if count <= 0]
        if bad:
            raise ValueError(f"Completion produced no new tokens: {bad[0]!r}")

        full_tokens = model.to_tokens(full_texts, prepend_bos=True)
        spans = make_steering_spans(prompt_lens, full_lens, position_mode=position_mode)

        if direction is not None and alpha != 0:
            if layer is None:
                raise ValueError("layer must be set when direction is used")
            hook_name = f"blocks.{layer}.hook_resid_post"
            logits = model.run_with_hooks(
                full_tokens,
                fwd_hooks=[(hook_name, make_position_steering_hook(direction, alpha, spans))],
            )
        else:
            logits = model(full_tokens)

        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        for batch_idx, (prompt_len, full_len, completion_tokens) in enumerate(
            zip(prompt_lens, full_lens, completion_token_counts)
        ):
            total = 0.0
            for token_pos in range(prompt_len, full_len):
                token_id = int(full_tokens[batch_idx, token_pos])
                total += float(log_probs[batch_idx, token_pos - 1, token_id])
            results.append((total, completion_tokens))
    return results


def make_directions(train_activations: torch.Tensor, train_labels: np.ndarray, seed: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    learned = probe_direction(train_activations, train_labels)

    random_np = rng.normal(size=train_activations.shape[1])
    random_direction = torch.tensor(random_np, dtype=train_activations.dtype)
    random_direction = random_direction / (random_direction.norm() + 1e-8)

    permuted_labels = rng.permutation(train_labels)
    label_permutation = probe_direction(train_activations, permuted_labels)
    return {
        "learned_probe": learned,
        "random_direction": random_direction,
        "label_permutation": label_permutation,
    }


def block_bootstrap_summary(
    frame: pd.DataFrame,
    *,
    n_samples: int,
    seed: int,
    score_col: str,
    delta_col: str,
    preference_col: str,
) -> dict[str, float | int]:
    groups = np.array(sorted(frame["pair_id"].unique()))
    if len(groups) == 0 or n_samples <= 0:
        return {
            f"{delta_col}_ci_low": float("nan"),
            f"{delta_col}_ci_high": float("nan"),
            f"{preference_col}_ci_low": float("nan"),
            f"{preference_col}_ci_high": float("nan"),
            "bootstrap_samples": 0,
        }

    block_row_counts: list[int] = []
    block_delta_sums: list[float] = []
    block_preference_sums: list[float] = []
    block_exact_scores: list[float] = []
    for group in groups:
        block = frame[frame["pair_id"] == group]
        block_row_counts.append(int(len(block)))
        block_delta_sums.append(float(block[delta_col].sum()))
        block_preference_sums.append(float(block[preference_col].sum()))
        block_exact_scores.append(float((block[score_col] > 0).all()))

    row_counts = np.array(block_row_counts)
    delta_sums = np.array(block_delta_sums)
    preference_sums = np.array(block_preference_sums)
    block_exact_scores_array = np.array(block_exact_scores)

    rng = np.random.default_rng(seed)
    delta_values: list[float] = []
    preference_values: list[float] = []
    block_exact_values: list[float] = []
    for _ in range(n_samples):
        sampled_indices = rng.integers(0, len(groups), size=len(groups))
        sampled_row_count = int(row_counts[sampled_indices].sum())
        delta_values.append(float(delta_sums[sampled_indices].sum() / sampled_row_count))
        preference_values.append(float(preference_sums[sampled_indices].sum() / sampled_row_count))
        block_exact_values.append(float(block_exact_scores_array[sampled_indices].mean()))

    return {
        f"{delta_col}_ci_low": float(np.quantile(delta_values, 0.025)),
        f"{delta_col}_ci_high": float(np.quantile(delta_values, 0.975)),
        f"{preference_col}_ci_low": float(np.quantile(preference_values, 0.025)),
        f"{preference_col}_ci_high": float(np.quantile(preference_values, 0.975)),
        "block_exact_accuracy_ci_low": float(np.quantile(block_exact_values, 0.025)),
        "block_exact_accuracy_ci_high": float(np.quantile(block_exact_values, 0.975)),
        "bootstrap_samples": len(delta_values),
    }


def summarize_group(
    frame: pd.DataFrame,
    *,
    group_name: str,
    direction_type: str,
    alpha: float,
    layer: int,
    seed: int,
    n_bootstrap: int,
    position_mode: str = "all",
) -> dict[str, object]:
    block_correct_counts = frame.groupby("pair_id")["prefers_correct_avg_token"].sum()
    block_exact_accuracy = float((block_correct_counts == 2).mean())
    block_one_correct_rate = float((block_correct_counts == 1).mean())
    block_zero_correct_rate = float((block_correct_counts == 0).mean())
    bootstrap = block_bootstrap_summary(
        frame,
        n_samples=n_bootstrap,
        seed=seed,
        score_col="avg_token_margin",
        delta_col="delta_avg_token_margin",
        preference_col="prefers_correct_avg_token",
    )
    return {
        "direction_type": direction_type,
        "position_mode": position_mode,
        "layer": layer,
        "seed": seed,
        "split": group_name,
        "alpha": alpha,
        "countries": int(len(frame)),
        "blocks": int(frame["pair_id"].nunique()),
        "mean_total_margin": float(frame["total_margin"].mean()),
        "mean_avg_token_margin": float(frame["avg_token_margin"].mean()),
        "mean_delta_total_margin": float(frame["delta_total_margin"].mean()),
        "mean_delta_avg_token_margin": float(frame["delta_avg_token_margin"].mean()),
        "pairwise_total_accuracy": float(frame["prefers_correct_total"].mean()),
        "pairwise_avg_token_accuracy": float(frame["prefers_correct_avg_token"].mean()),
        "block_exact_avg_token_accuracy": block_exact_accuracy,
        "block_one_correct_avg_token_rate": block_one_correct_rate,
        "block_zero_correct_avg_token_rate": block_zero_correct_rate,
        "ci_unit": "pair_id_block",
        **bootstrap,
    }


def rebuild_summary_from_details(args: argparse.Namespace) -> None:
    details_path = Path(args.out_details)
    if not details_path.exists():
        raise FileNotFoundError(f"Cannot rebuild summary because details file does not exist: {details_path}")

    details = pd.read_csv(details_path)
    summary: list[dict[str, object]] = []
    if "position_mode" not in details.columns:
        details["position_mode"] = "all"

    for (position_mode, direction_type, alpha), alpha_frame in details.groupby(
        ["position_mode", "direction_type", "alpha"],
        sort=True,
    ):
        layer = int(alpha_frame["layer"].iloc[0]) if "layer" in alpha_frame.columns else args.layer
        seed = int(alpha_frame["seed"].iloc[0]) if "seed" in alpha_frame.columns else args.seed
        for split_name, split_frame in [
            ("all_countries", alpha_frame.copy()),
            ("heldout_countries", alpha_frame[alpha_frame["split"] == "test"].copy()),
        ]:
            summary.append(
                summarize_group(
                    split_frame,
                    group_name=split_name,
                    direction_type=direction_type,
                    alpha=float(alpha),
                    layer=layer,
                    seed=seed,
                    n_bootstrap=args.bootstrap_samples,
                    position_mode=str(position_mode),
                )
            )

    summary_out = Path(args.out_summary)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(summary_out, index=False)
    print(f"Rebuilt completion-margin steering summary from {details_path}")
    print(f"Saved completion-margin steering summary to {summary_out}")


def main() -> None:
    args = parse_args()
    if args.from_details:
        rebuild_summary_from_details(args)
        return

    data = filter_dataset(load_dataset(args.data), language=args.language, domain=args.domain).copy()
    country_data = build_country_frame(data)

    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy().astype(int)
    groups = data["pair_id"].to_numpy()

    model = load_hooked_transformer(args.model)
    activations = collect_resid_post_by_layer(model, prompts)[args.layer]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(activations.numpy(), labels, groups=groups))
    train_pair_ids = set(data.iloc[train_idx]["pair_id"].tolist())
    test_pair_ids = set(data.iloc[test_idx]["pair_id"].tolist())
    country_data["split"] = np.where(country_data["pair_id"].isin(test_pair_ids), "test", "train")

    directions = make_directions(activations[train_idx], labels[train_idx], args.seed)
    directions = {name: direction.to(model.cfg.device) for name, direction in directions.items()}

    baseline_rows: dict[str, dict[str, float | int]] = {}
    details: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []

    for direction_type, direction in directions.items():
        for alpha in args.alphas:
            alpha_rows: list[dict[str, object]] = []
            completion_prompts = country_data["completion_prompt"].tolist()
            correct_completions = [f" {capital}" for capital in country_data["correct_capital"].tolist()]
            false_completions = [f" {capital}" for capital in country_data["false_capital"].tolist()]
            correct_scores = completion_logprobs(
                model,
                completion_prompts,
                correct_completions,
                direction=direction,
                alpha=alpha,
                layer=args.layer,
                position_mode=args.position_mode,
            )
            false_scores = completion_logprobs(
                model,
                completion_prompts,
                false_completions,
                direction=direction,
                alpha=alpha,
                layer=args.layer,
                position_mode=args.position_mode,
            )
            for row_index, row in enumerate(country_data.itertuples(index=False)):
                correct_total, correct_tokens = correct_scores[row_index]
                false_total, false_tokens = false_scores[row_index]
                correct_avg = correct_total / correct_tokens
                false_avg = false_total / false_tokens
                total_margin = correct_total - false_total
                avg_token_margin = correct_avg - false_avg

                key = f"{row.country}::{direction_type}"
                if alpha == 0:
                    baseline_rows[key] = {
                        "baseline_total_margin": total_margin,
                        "baseline_avg_token_margin": avg_token_margin,
                    }
                baseline = baseline_rows.get(key)
                if baseline is None:
                    # Alphas are normally sorted by the caller, but make the row robust.
                    base_correct_score = completion_logprobs(
                        model,
                        [row.completion_prompt],
                        [f" {row.correct_capital}"],
                    )[0]
                    base_false_score = completion_logprobs(
                        model,
                        [row.completion_prompt],
                        [f" {row.false_capital}"],
                    )[0]
                    base_correct_total, base_correct_tokens = base_correct_score
                    base_false_total, base_false_tokens = base_false_score
                    baseline = {
                        "baseline_total_margin": base_correct_total - base_false_total,
                        "baseline_avg_token_margin": (base_correct_total / base_correct_tokens)
                        - (base_false_total / base_false_tokens),
                    }
                    baseline_rows[key] = baseline

                alpha_rows.append(
                    {
                        "direction_type": direction_type,
                        "position_mode": args.position_mode,
                        "layer": args.layer,
                        "seed": args.seed,
                        "alpha": alpha,
                        "split": row.split,
                        "pair_id": row.pair_id,
                        "country": row.country,
                        "correct_capital": row.correct_capital,
                        "false_capital": row.false_capital,
                        "correct_total_logprob": correct_total,
                        "false_total_logprob": false_total,
                        "correct_avg_token_logprob": correct_avg,
                        "false_avg_token_logprob": false_avg,
                        "correct_completion_tokens": correct_tokens,
                        "false_completion_tokens": false_tokens,
                        "total_margin": total_margin,
                        "avg_token_margin": avg_token_margin,
                        "baseline_total_margin": baseline["baseline_total_margin"],
                        "baseline_avg_token_margin": baseline["baseline_avg_token_margin"],
                        "delta_total_margin": total_margin - baseline["baseline_total_margin"],
                        "delta_avg_token_margin": avg_token_margin - baseline["baseline_avg_token_margin"],
                        "prefers_correct_total": int(total_margin > 0),
                        "prefers_correct_avg_token": int(avg_token_margin > 0),
                    }
                )

            alpha_frame = pd.DataFrame(alpha_rows)
            details.extend(alpha_rows)
            for split_name, split_frame in [
                ("all_countries", alpha_frame),
                ("heldout_countries", alpha_frame[alpha_frame["split"] == "test"].copy()),
            ]:
                summary.append(
                    summarize_group(
                        split_frame,
                        group_name=split_name,
                        direction_type=direction_type,
                        alpha=alpha,
                        layer=args.layer,
                        seed=args.seed,
                        n_bootstrap=args.bootstrap_samples,
                        position_mode=args.position_mode,
                    )
                )
            heldout = alpha_frame[alpha_frame["split"] == "test"]
            print(
                f"control={direction_type:<17} alpha={alpha:+.2f} "
                f"heldout_delta_avg={heldout['delta_avg_token_margin'].mean():+.4f} "
                f"heldout_pair_acc={heldout['prefers_correct_avg_token'].mean():.3f}"
            )

    details_out = Path(args.out_details)
    summary_out = Path(args.out_summary)
    details_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(details).to_csv(details_out, index=False)
    pd.DataFrame(summary).to_csv(summary_out, index=False)
    print(f"Saved completion-margin steering details to {details_out}")
    print(f"Saved completion-margin steering summary to {summary_out}")


if __name__ == "__main__":
    main()
