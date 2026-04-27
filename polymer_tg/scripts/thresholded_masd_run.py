from __future__ import annotations

"""Runner for exploratory thresholded-MASD training and evaluation sweeps."""

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polymer_tg.scripts.mainline_run as masd_run
from polymer_tg.scripts.mainline_eval import paired_stats, reduction_interpretation, summary_stats, summarize_payload_metrics
from train.experiment_overrides import temporary_experiment_overrides
from train.full_train import DEVICE, build_model, diagnostic_config, load_artifacts, make_loader, prepare_seed_tensors
from train.msce_stage import ensure_msce_features, train_msce_stage
from train.rcmf_stage import train_rcmf_external_focus_stage, train_rcmf_stage


RUN_NAME = "thresholded_masd_20260407"
RUN_DIR = ROOT / "outputs" / "exp" / "diagnostics" / RUN_NAME
CACHE_PATH = RUN_DIR / "cache.pt"
SCREEN_SEEDS = (0, 1)
FINAL_SEEDS = (0, 1, 2, 3, 4)

TAU_GRID = (0.75, 0.80, 0.85)
BOUND_GRID = (1.5, 2.0, 3.0)
MEAN_LAMBDA_GRID = (0.01, 0.02)
SPARSE_LAMBDA_GRID = (1.0e-3, 5.0e-3)


def fmt_ci(low: float, high: float, *, unit: str = "K") -> str:
    suffix = f" {unit}" if unit else ""
    return f"[{low:.4f}, {high:.4f}]{suffix}"


def sign_consistency(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    pos = int(np.sum(arr > 1e-8))
    neg = int(np.sum(arr < -1e-8))
    zero = int(arr.size - pos - neg)
    return f"{pos}/{arr.size} positive, {neg}/{arr.size} negative, {zero}/{arr.size} zero"


def load_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {"base": {}, "screen": {}, "final": {}}
    return torch.load(CACHE_PATH, map_location="cpu")


def save_cache(cache: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE_PATH)


def row_by_name(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    for row in rows:
        if str(row["model_name"]) == model_name:
            return row
    raise KeyError(model_name)


def current_stage_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return masd_run.current_stage_row(rows)


def candidate_grid() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for tau, bound_k, mean_lambda, sparse_lambda in itertools.product(
        TAU_GRID,
        BOUND_GRID,
        MEAN_LAMBDA_GRID,
        SPARSE_LAMBDA_GRID,
    ):
        label = f"tm_tau{int(round(tau*100)):02d}_b{str(bound_k).replace('.', 'p')}_lm{str(mean_lambda).replace('.', 'p')}_ls{str(sparse_lambda).replace('.', 'p')}"
        items.append(
            {
                "label": label,
                "tau": float(tau),
                "bound_k": float(bound_k),
                "mean_lambda": float(mean_lambda),
                "sparse_lambda": float(sparse_lambda),
                "gamma": 8.0,
            }
        )
    return items


def masd_overrides(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "thresholded_masd_enabled": True,
        "thresholded_masd_bound_k": float(candidate["bound_k"]),
        "thresholded_masd_tau": float(candidate["tau"]),
        "thresholded_masd_gamma": float(candidate["gamma"]),
        "thresholded_masd_mean_lambda": float(candidate["mean_lambda"]),
        "thresholded_masd_sparse_lambda": float(candidate["sparse_lambda"]),
        "thresholded_masd_focus_only": True,
    }


def payload_activation_stats(payload: dict[str, Any]) -> dict[str, float]:
    gate = np.asarray(payload.get("masd_thresholded_gate", []), dtype=np.float64).reshape(-1)
    applied = np.asarray(payload.get("masd_applied_delta", []), dtype=np.float64).reshape(-1)
    thresholded_delta = np.asarray(payload.get("masd_thresholded_delta", []), dtype=np.float64).reshape(-1)
    if gate.size == 0:
        return {
            "activation_rate": float("nan"),
            "mean_gate": float("nan"),
            "mean_correction": float("nan"),
            "mean_abs_correction": float("nan"),
            "mean_thresholded_delta": float("nan"),
        }
    return {
        "activation_rate": float(np.mean(gate >= 0.5)),
        "mean_gate": float(gate.mean()),
        "mean_correction": float(applied.mean()) if applied.size else float("nan"),
        "mean_abs_correction": float(np.mean(np.abs(applied))) if applied.size else float("nan"),
        "mean_thresholded_delta": float(thresholded_delta.mean()) if thresholded_delta.size else float("nan"),
    }


def ensure_base_seed(
    *,
    cache: dict[str, Any],
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
    seed: int,
    epoch_log: list[float],
) -> dict[str, Any]:
    key = str(int(seed))
    if key in cache["base"]:
        return cache["base"][key]

    split = masd_run.ensure_protocol_split(splits, dataset, seed=seed)
    seed_tensors = prepare_seed_tensors(features, split["train"], dataset)
    baseline_model, msce_model = train_msce_stage(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        repeat_id=seed,
    )
    minimal_rcmf = train_rcmf_stage(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        repeat_id=seed,
        repair_model=msce_model,
    )
    current_rcmf = train_rcmf_external_focus_stage(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        repeat_id=seed,
        minimal_rcmf=minimal_rcmf,
    )
    current_full_model = masd_run.train_masd_current_student(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        seed=seed,
        current_rcmf=current_rcmf,
        mode=masd_run.CURRENT_MODE,
        selection_policy="tailfix",
        epoch_log=epoch_log,
    )

    primary_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)

    rows: list[dict[str, Any]] = []
    bundle: dict[str, Any] = {"seed": int(seed)}
    stages = [
        ("strongest_baseline", baseline_model),
        ("strongest_baseline_plus_mspce", msce_model),
        ("strongest_baseline_plus_mspce_rcmf", current_rcmf),
        (masd_run.CURRENT_STAGE_NAME, current_full_model),
    ]
    baseline_metrics: dict[str, float] | None = None
    prev_stage: dict[str, float] | None = None
    for idx, (name, model) in enumerate(stages):
        need_payload = True
        clean_metrics, clean_payload = masd_run.evaluate_stage(
            model,
            primary_loader,
            seed_tensors,
            variant="clean",
            noise_seed=seed * 601 + idx * 10 + 1,
            return_payload=need_payload,
        )
        noisy_metrics, noisy_payload = masd_run.evaluate_stage(
            model,
            primary_loader,
            seed_tensors,
            variant="noisy",
            noise_seed=seed * 601 + idx * 10 + 2,
            return_payload=need_payload,
        )
        external_metrics, external_payload = masd_run.evaluate_stage(
            model,
            external_loader,
            seed_tensors,
            variant="clean",
            noise_seed=seed * 601 + idx * 10 + 3,
            return_payload=need_payload,
        )
        stage_metrics = {
            "primary_clean": float(clean_metrics["mae_k"]),
            "primary_noisy": float(noisy_metrics["mae_k"]),
            "primary_hard_subgroup": float(clean_metrics["hard_subgroup_mae_k"]),
            "external_holdout": float(external_metrics["mae_k"]),
            "external_hard_subgroup": float(external_metrics["hard_subgroup_mae_k"]),
        }
        if baseline_metrics is None:
            baseline_metrics = stage_metrics
        rows.append(
            {
                "seed": int(seed),
                "model_name": name,
                **stage_metrics,
                "delta_vs_strongest_baseline_primary_clean": float(stage_metrics["primary_clean"] - baseline_metrics["primary_clean"]),
                "delta_vs_strongest_baseline_primary_hard_subgroup": float(stage_metrics["primary_hard_subgroup"] - baseline_metrics["primary_hard_subgroup"]),
                "delta_vs_strongest_baseline_external_holdout": float(stage_metrics["external_holdout"] - baseline_metrics["external_holdout"]),
                "delta_vs_previous_primary_clean": 0.0 if prev_stage is None else float(stage_metrics["primary_clean"] - prev_stage["primary_clean"]),
                "delta_vs_previous_primary_hard_subgroup": 0.0 if prev_stage is None else float(stage_metrics["primary_hard_subgroup"] - prev_stage["primary_hard_subgroup"]),
                "delta_vs_previous_external_holdout": 0.0 if prev_stage is None else float(stage_metrics["external_holdout"] - prev_stage["external_holdout"]),
            }
        )
        prev_stage = stage_metrics
        if name == "strongest_baseline":
            bundle["baseline_primary_clean"] = clean_payload
            bundle["baseline_external"] = external_payload
        elif name == "strongest_baseline_plus_mspce":
            bundle["mspce_primary_clean"] = clean_payload
            bundle["mspce_external"] = external_payload
        elif name == "strongest_baseline_plus_mspce_rcmf":
            bundle["rcmf_primary_clean"] = clean_payload
            bundle["rcmf_external"] = external_payload
        else:
            bundle["masd_primary_clean"] = clean_payload
            bundle["masd_primary_noisy"] = noisy_payload
            bundle["masd_external"] = external_payload

    val_clean, val_clean_payload = masd_run.evaluate_stage(
        current_full_model,
        val_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 811 + 1,
        return_payload=True,
    )
    val_external, val_external_payload = masd_run.evaluate_stage(
        current_full_model,
        external_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 811 + 2,
        return_payload=True,
    )

    payload = {
        "seed": int(seed),
        "split": split,
        "seed_tensors": seed_tensors,
        "rows": rows,
        "bundle": bundle,
        "current_full_state": {k: v.detach().cpu() for k, v in current_full_model.state_dict().items()},
        "val_clean": val_clean,
        "val_external": val_external,
        "val_clean_payload": val_clean_payload,
        "val_external_payload": val_external_payload,
    }
    cache["base"][key] = payload
    save_cache(cache)
    return payload


def masd_focus_score(
    *,
    val_clean: dict[str, float],
    val_external: dict[str, float],
    activation_stats: dict[str, float],
) -> float:
    activation_rate = float(activation_stats["activation_rate"])
    mean_correction = abs(float(activation_stats["mean_correction"]))
    return (
        float(val_external["mae_k"])
        + 0.45 * float(val_clean["mae_k"])
        + 0.12 * float(val_clean["hard_subgroup_mae_k"])
        + 1.20 * max(0.0, activation_rate - 0.25)
        + 0.80 * max(0.0, 0.10 - activation_rate)
        + 0.40 * mean_correction
    )


def train_thresholded_candidate(
    *,
    base_payload: dict[str, Any],
    config: Any,
    seed: int,
    candidate: dict[str, Any],
    epoch_log: list[float],
) -> tuple[nn.Module, dict[str, Any]]:
    from torch import nn  # local import to keep file-level imports small

    split = base_payload["split"]
    seed_tensors = base_payload["seed_tensors"]
    overrides = masd_overrides(candidate)
    with temporary_experiment_overrides(label=str(candidate["label"]), **overrides):
        student = build_model(masd_run.CURRENT_MODE, seed_tensors, config)
    student.load_state_dict(base_payload["current_full_state"])
    student.to(DEVICE)

    train_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=True)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)

    best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    init_val_clean, init_val_payload = masd_run.evaluate_stage(
        student,
        val_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1201 + 1,
        return_payload=True,
    )
    init_val_external, _ = masd_run.evaluate_stage(
        student,
        external_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1201 + 2,
        return_payload=False,
    )
    best_aux = payload_activation_stats(init_val_payload)
    best_score = masd_focus_score(
        val_clean=init_val_clean,
        val_external=init_val_external,
        activation_stats=best_aux,
    )

    phase_specs = (
        (2, 1.2e-5, 1.8),
        (2, 4.0e-6, 2.4),
    )
    phase_index = 0
    for epochs, lr, wd_scale in phase_specs:
        masd_run.set_masd_trainable(student, stage="C")
        optimizer = torch.optim.AdamW(
            [param for param in student.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=config.weight_decay * wd_scale,
        )
        for _ in range(epochs):
            tick = time.time()
            student.train()
            for batch in train_loader:
                batch = masd_run._to_device(batch)
                out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                second_out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                loss, _loss_terms = masd_run.masd_current_loss(
                    out,
                    batch["y"],
                    model=student,
                    mode=masd_run.CURRENT_MODE,
                    stage="C",
                    second_out=second_out,
                    cluster_support=batch["cluster_support"],
                    chemistry_multihot=batch["chemistry_multihot"],
                    other_subcluster_code=batch["other_subcluster_code"],
                    stage_progress=1.0,
                    dro_cap=0.34,
                    dro_temperature=0.11,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=2.5)
                optimizer.step()
            epoch_log.append(float(time.time() - tick))
            phase_index += 1
            val_clean, val_payload = masd_run.evaluate_stage(
                student,
                val_loader,
                seed_tensors,
                variant="clean",
                noise_seed=seed * 1201 + 10 + phase_index,
                return_payload=True,
            )
            val_external, _ = masd_run.evaluate_stage(
                student,
                external_loader,
                seed_tensors,
                variant="clean",
                noise_seed=seed * 1201 + 30 + phase_index,
                return_payload=False,
            )
            aux = payload_activation_stats(val_payload)
            score = masd_focus_score(
                val_clean=val_clean,
                val_external=val_external,
                activation_stats=aux,
            )
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
                best_aux = aux
    student.load_state_dict(best_state)
    student.to(DEVICE)
    return student, {
        "best_score": float(best_score),
        "best_aux": best_aux,
    }


def run_thresholded_candidate(
    *,
    cache: dict[str, Any],
    base_payload: dict[str, Any],
    config: Any,
    seed: int,
    candidate: dict[str, Any],
    cluster_masks: dict[str, np.ndarray],
    epoch_log: list[float],
    store_group: str,
) -> dict[str, Any]:
    key = f"{candidate['label']}::seed{seed}"
    if key in cache[store_group]:
        return cache[store_group][key]

    student, fit_meta = train_thresholded_candidate(
        base_payload=base_payload,
        config=config,
        seed=seed,
        candidate=candidate,
        epoch_log=epoch_log,
    )
    seed_tensors = base_payload["seed_tensors"]
    split = base_payload["split"]
    primary_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)

    clean_metrics, clean_payload = masd_run.evaluate_stage(
        student,
        primary_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1401 + 1,
        return_payload=True,
    )
    external_metrics, external_payload = masd_run.evaluate_stage(
        student,
        external_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1401 + 3,
        return_payload=True,
    )
    base_current = current_stage_row(base_payload["rows"])
    baseline_row = row_by_name(base_payload["rows"], "strongest_baseline")
    cluster_row = masd_run.external_cluster_reduction_row(
        baseline_external_payload=base_payload["bundle"]["baseline_external"],
        candidate_external_payload=external_payload,
        cluster_masks=cluster_masks,
    )
    other_delta_vs_baseline = float(cluster_row.get("cluster_other_mae_reduction_k", float("nan")))
    current_other_row = masd_run.external_cluster_reduction_row(
        baseline_external_payload=base_payload["bundle"]["masd_external"],
        candidate_external_payload=external_payload,
        cluster_masks=cluster_masks,
    )
    other_delta_vs_current = float(current_other_row.get("cluster_other_mae_reduction_k", float("nan")))

    payload = {
        "label": str(candidate["label"]),
        "seed": int(seed),
        "candidate": dict(candidate),
        "fit_meta": fit_meta,
        "metrics": {
            "main_delta_vs_current_full": float(base_current["primary_clean"] - clean_metrics["mae_k"]),
            "hard_delta_vs_current_full": float(base_current["primary_hard_subgroup"] - clean_metrics["hard_subgroup_mae_k"]),
            "external_delta_vs_current_full": float(base_current["external_holdout"] - external_metrics["mae_k"]),
            "main_delta_vs_baseline": float(baseline_row["primary_clean"] - clean_metrics["mae_k"]),
            "hard_delta_vs_baseline": float(baseline_row["primary_hard_subgroup"] - clean_metrics["hard_subgroup_mae_k"]),
            "external_delta_vs_baseline": float(baseline_row["external_holdout"] - external_metrics["mae_k"]),
            "candidate_main": float(clean_metrics["mae_k"]),
            "candidate_hard": float(clean_metrics["hard_subgroup_mae_k"]),
            "candidate_external": float(external_metrics["mae_k"]),
            "other_delta_vs_current_full": float(other_delta_vs_current),
            "other_delta_vs_baseline": float(other_delta_vs_baseline),
        },
        "primary_payload": clean_payload,
        "external_payload": external_payload,
        "activation_primary": payload_activation_stats(clean_payload),
        "activation_external": payload_activation_stats(external_payload),
    }
    cache[store_group][key] = payload
    save_cache(cache)
    return payload


def screen_summary_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, group in pd.DataFrame(records).groupby("label", sort=False):
        main_values = group["main_delta_vs_current_full"].tolist()
        hard_values = group["hard_delta_vs_current_full"].tolist()
        external_values = group["external_delta_vs_current_full"].tolist()
        other_values = group["other_delta_vs_current_full"].tolist()
        activation_values = group["activation_rate_main"].tolist()
        correction_values = group["mean_correction_main"].tolist()
        row = {
            "label": str(label),
            "tau": float(group["tau"].iloc[0]),
            "bound_k": float(group["bound_k"].iloc[0]),
            "lambda_mean": float(group["lambda_mean"].iloc[0]),
            "lambda_sparse": float(group["lambda_sparse"].iloc[0]),
            "num_seeds": int(group["seed"].nunique()),
            "main_delta_mean": float(np.mean(main_values)),
            "hard_delta_mean": float(np.mean(hard_values)),
            "external_delta_mean": float(np.mean(external_values)),
            "other_delta_mean": float(np.mean(other_values)),
            "activation_rate_mean": float(np.mean(activation_values)),
            "mean_correction_mean": float(np.mean(correction_values)),
            "main_sign_consistency": sign_consistency(main_values),
            "hard_sign_consistency": sign_consistency(hard_values),
            "external_sign_consistency": sign_consistency(external_values),
            "other_sign_consistency": sign_consistency(other_values),
        }
        row["pass_screen"] = bool(
            row["main_delta_mean"] >= -0.15
            and row["external_delta_mean"] > 0.30
            and row["hard_delta_mean"] >= -0.30
            and row["other_delta_mean"] >= -1.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def choose_screen_winner(screen_df: pd.DataFrame) -> dict[str, Any]:
    ranked = screen_df.sort_values(
        by=["pass_screen", "external_delta_mean", "main_delta_mean", "hard_delta_mean", "other_delta_mean", "activation_rate_mean"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    label = str(ranked.iloc[0]["label"])
    return next(item for item in candidate_grid() if item["label"] == label)


def aggregate_5seed_rows(
    *,
    base_results: dict[int, dict[str, Any]],
    best_results: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    current_primary_payloads = [base_results[seed]["bundle"]["masd_primary_clean"] for seed in sorted(base_results)]
    best_primary_payloads = [best_results[seed]["primary_payload"] for seed in sorted(best_results)]
    current_main = summarize_payload_metrics(current_primary_payloads)
    best_main = summarize_payload_metrics(best_primary_payloads)
    current_hard_values = [current_stage_row(base_results[seed]["rows"])["primary_hard_subgroup"] for seed in sorted(base_results)]
    best_hard_values = [best_results[seed]["metrics"]["candidate_hard"] for seed in sorted(best_results)]
    current_external_values = [current_stage_row(base_results[seed]["rows"])["external_holdout"] for seed in sorted(base_results)]
    best_external_values = [best_results[seed]["metrics"]["candidate_external"] for seed in sorted(best_results)]

    main_diff = np.asarray(current_main["mae_values"], dtype=np.float64) - np.asarray(best_main["mae_values"], dtype=np.float64)
    hard_diff = np.asarray(current_hard_values, dtype=np.float64) - np.asarray(best_hard_values, dtype=np.float64)
    external_diff = np.asarray(current_external_values, dtype=np.float64) - np.asarray(best_external_values, dtype=np.float64)

    main_stats = paired_stats(main_diff)
    hard_stats = paired_stats(hard_diff)
    external_stats = paired_stats(external_diff)

    rows = [
        {
            "row_type": "model",
            "name": "current_full",
            "main_mae_mean_k": float(current_main["mae"]["mean"]),
            "main_ci95_low_k": float(current_main["mae"]["ci95_low"]),
            "main_ci95_high_k": float(current_main["mae"]["ci95_high"]),
            "hard_mae_mean_k": float(summary_stats(current_hard_values)["mean"]),
            "hard_ci95_low_k": float(summary_stats(current_hard_values)["ci95_low"]),
            "hard_ci95_high_k": float(summary_stats(current_hard_values)["ci95_high"]),
            "external_mae_mean_k": float(summary_stats(current_external_values)["mean"]),
            "external_ci95_low_k": float(summary_stats(current_external_values)["ci95_low"]),
            "external_ci95_high_k": float(summary_stats(current_external_values)["ci95_high"]),
        },
        {
            "row_type": "model",
            "name": "thresholded_masd_best",
            "main_mae_mean_k": float(best_main["mae"]["mean"]),
            "main_ci95_low_k": float(best_main["mae"]["ci95_low"]),
            "main_ci95_high_k": float(best_main["mae"]["ci95_high"]),
            "hard_mae_mean_k": float(summary_stats(best_hard_values)["mean"]),
            "hard_ci95_low_k": float(summary_stats(best_hard_values)["ci95_low"]),
            "hard_ci95_high_k": float(summary_stats(best_hard_values)["ci95_high"]),
            "external_mae_mean_k": float(summary_stats(best_external_values)["mean"]),
            "external_ci95_low_k": float(summary_stats(best_external_values)["ci95_low"]),
            "external_ci95_high_k": float(summary_stats(best_external_values)["ci95_high"]),
        },
        {
            "row_type": "comparison",
            "name": "current_full_vs_thresholded_masd_best",
            "main_delta_mean_k": float(main_stats["mean"]),
            "main_delta_ci95_low_k": float(main_stats["ci95_low"]),
            "main_delta_ci95_high_k": float(main_stats["ci95_high"]),
            "main_t_pvalue": float(main_stats["t_pvalue"]),
            "main_perm_pvalue": float(main_stats["perm_pvalue"]),
            "main_sign_consistency": sign_consistency(main_diff.tolist()),
            "hard_delta_mean_k": float(hard_stats["mean"]),
            "hard_delta_ci95_low_k": float(hard_stats["ci95_low"]),
            "hard_delta_ci95_high_k": float(hard_stats["ci95_high"]),
            "hard_t_pvalue": float(hard_stats["t_pvalue"]),
            "hard_perm_pvalue": float(hard_stats["perm_pvalue"]),
            "hard_sign_consistency": sign_consistency(hard_diff.tolist()),
            "external_delta_mean_k": float(external_stats["mean"]),
            "external_delta_ci95_low_k": float(external_stats["ci95_low"]),
            "external_delta_ci95_high_k": float(external_stats["ci95_high"]),
            "external_t_pvalue": float(external_stats["t_pvalue"]),
            "external_perm_pvalue": float(external_stats["perm_pvalue"]),
            "external_sign_consistency": sign_consistency(external_diff.tolist()),
        },
    ]
    return pd.DataFrame(rows)


def cluster_df(
    *,
    base_results: dict[int, dict[str, Any]],
    best_results: dict[int, dict[str, Any]],
    cluster_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_name in masd_run.CHEMISTRY_CLUSTER_ORDER:
        mask = cluster_masks.get(cluster_name)
        if mask is None or not bool(mask.any()):
            continue
        current_values = []
        best_values = []
        for seed in sorted(base_results):
            current_err = np.asarray(base_results[seed]["bundle"]["masd_external"]["error"], dtype=np.float64).reshape(-1)
            best_err = np.asarray(best_results[seed]["external_payload"]["error"], dtype=np.float64).reshape(-1)
            current_values.append(float(current_err[mask].mean()))
            best_values.append(float(best_err[mask].mean()))
        diff = np.asarray(current_values, dtype=np.float64) - np.asarray(best_values, dtype=np.float64)
        stats = paired_stats(diff)
        rows.append(
            {
                "cluster_name": cluster_name,
                "sample_count": int(mask.sum()),
                "current_full_mae_mean_k": float(summary_stats(current_values)["mean"]),
                "thresholded_masd_mae_mean_k": float(summary_stats(best_values)["mean"]),
                "delta_vs_current_full_k": float(stats["mean"]),
                "ci95_low_k": float(stats["ci95_low"]),
                "ci95_high_k": float(stats["ci95_high"]),
                "t_pvalue": float(stats["t_pvalue"]),
                "perm_pvalue": float(stats["perm_pvalue"]),
                "interpretation": reduction_interpretation(stats),
            }
        )
    return pd.DataFrame(rows)


def activation_df(best_results: dict[int, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in sorted(best_results):
        main_stats = best_results[seed]["activation_primary"]
        ext_stats = best_results[seed]["activation_external"]
        rows.append(
            {
                "seed": int(seed),
                "main_activation_rate": float(main_stats["activation_rate"]),
                "main_mean_gate": float(main_stats["mean_gate"]),
                "main_mean_correction": float(main_stats["mean_correction"]),
                "main_mean_abs_correction": float(main_stats["mean_abs_correction"]),
                "external_activation_rate": float(ext_stats["activation_rate"]),
                "external_mean_gate": float(ext_stats["mean_gate"]),
                "external_mean_correction": float(ext_stats["mean_correction"]),
                "external_mean_abs_correction": float(ext_stats["mean_abs_correction"]),
            }
        )
    agg = {
        "seed": "mean",
        "main_activation_rate": float(np.mean([row["main_activation_rate"] for row in rows])),
        "main_mean_gate": float(np.mean([row["main_mean_gate"] for row in rows])),
        "main_mean_correction": float(np.mean([row["main_mean_correction"] for row in rows])),
        "main_mean_abs_correction": float(np.mean([row["main_mean_abs_correction"] for row in rows])),
        "external_activation_rate": float(np.mean([row["external_activation_rate"] for row in rows])),
        "external_mean_gate": float(np.mean([row["external_mean_gate"] for row in rows])),
        "external_mean_correction": float(np.mean([row["external_mean_correction"] for row in rows])),
        "external_mean_abs_correction": float(np.mean([row["external_mean_abs_correction"] for row in rows])),
    }
    rows.append(agg)
    return pd.DataFrame(rows)


def write_outputs(
    *,
    screen_df: pd.DataFrame,
    best_candidate: dict[str, Any],
    five_seed_df: pd.DataFrame,
    cluster_table: pd.DataFrame,
    activation_table: pd.DataFrame,
) -> None:
    screen_df.to_csv(RUN_DIR / "thresholded_masd_screen.csv", index=False)
    five_seed_df.to_csv(RUN_DIR / "thresholded_masd_5seed.csv", index=False)
    cluster_table.to_csv(RUN_DIR / "thresholded_masd_cluster.csv", index=False)
    activation_table.to_csv(RUN_DIR / "thresholded_masd_activation.csv", index=False)

    best_screen_row = screen_df.loc[screen_df["label"] == best_candidate["label"]].iloc[0]
    compare_row = five_seed_df.loc[five_seed_df["row_type"] == "comparison"].iloc[0]
    other_row = cluster_table.loc[cluster_table["cluster_name"] == "other"].iloc[0]
    activation_mean = activation_table.loc[activation_table["seed"] == "mean"].iloc[0]

    screen_lines = [
        "# Thresholded MASD screen",
        "",
        f"- Screen winner: `{best_candidate['label']}` with tau={best_candidate['tau']:.2f}, b={best_candidate['bound_k']:.1f}, lambda_mean={best_candidate['mean_lambda']:.3f}, lambda_sparse={best_candidate['sparse_lambda']:.4f}.",
        f"- Two-seed mean deltas vs `current_full`: main {float(best_screen_row['main_delta_mean']):+.4f} K, hard {float(best_screen_row['hard_delta_mean']):+.4f} K, external {float(best_screen_row['external_delta_mean']):+.4f} K, other {float(best_screen_row['other_delta_mean']):+.4f} K.",
        f"- Two-seed activation rate mean: {float(best_screen_row['activation_rate_mean']):.4f}; mean correction: {float(best_screen_row['mean_correction_mean']):+.4f} K.",
    ]
    (RUN_DIR / "thresholded_masd_screen.md").write_text("\n".join(screen_lines) + "\n", encoding="utf-8")

    five_lines = [
        "# Thresholded MASD 5-seed confirmation",
        "",
        f"- Main delta vs `current_full`: {float(compare_row['main_delta_mean_k']):+.4f} K, 95% CI {fmt_ci(float(compare_row['main_delta_ci95_low_k']), float(compare_row['main_delta_ci95_high_k']))}.",
        f"- Hard delta vs `current_full`: {float(compare_row['hard_delta_mean_k']):+.4f} K, 95% CI {fmt_ci(float(compare_row['hard_delta_ci95_low_k']), float(compare_row['hard_delta_ci95_high_k']))}.",
        f"- External delta vs `current_full`: {float(compare_row['external_delta_mean_k']):+.4f} K, 95% CI {fmt_ci(float(compare_row['external_delta_ci95_low_k']), float(compare_row['external_delta_ci95_high_k']))}.",
        f"- Sign consistency: main {compare_row['main_sign_consistency']}; hard {compare_row['hard_sign_consistency']}; external {compare_row['external_sign_consistency']}.",
    ]
    (RUN_DIR / "thresholded_masd_5seed.md").write_text("\n".join(five_lines) + "\n", encoding="utf-8")

    cluster_lines = ["# Thresholded MASD clusters", ""]
    for cluster_name in masd_run.CHEMISTRY_CLUSTER_ORDER:
        row = cluster_table.loc[cluster_table["cluster_name"] == cluster_name]
        if row.empty:
            cluster_lines.append(f"- `{cluster_name}`: not sampled in external holdout.")
            continue
        item = row.iloc[0]
        cluster_lines.append(
            f"- `{cluster_name}`: delta vs `current_full` {float(item['delta_vs_current_full_k']):+.4f} K, 95% CI {fmt_ci(float(item['ci95_low_k']), float(item['ci95_high_k']))}, interpretation = {item['interpretation']}."
        )
    (RUN_DIR / "thresholded_masd_cluster.md").write_text("\n".join(cluster_lines) + "\n", encoding="utf-8")

    activation_lines = [
        "# Thresholded MASD activation",
        "",
        f"- Main activation rate mean: {float(activation_mean['main_activation_rate']):.4f}; main mean gate: {float(activation_mean['main_mean_gate']):.4f}.",
        f"- Main mean correction: {float(activation_mean['main_mean_correction']):+.4f} K; mean abs correction: {float(activation_mean['main_mean_abs_correction']):.4f} K.",
        f"- External activation rate mean: {float(activation_mean['external_activation_rate']):.4f}; external mean gate: {float(activation_mean['external_mean_gate']):.4f}.",
        f"- External mean correction: {float(activation_mean['external_mean_correction']):+.4f} K; mean abs correction: {float(activation_mean['external_mean_abs_correction']):.4f} K.",
    ]
    (RUN_DIR / "thresholded_masd_activation.md").write_text("\n".join(activation_lines) + "\n", encoding="utf-8")

    replaceable = bool(
        float(compare_row["main_delta_mean_k"]) >= -0.15
        and float(compare_row["external_delta_mean_k"]) > 0.30
        and float(compare_row["hard_delta_mean_k"]) >= -0.30
        and float(other_row["delta_vs_current_full_k"]) >= -1.0
    )
    advisor_lines = [
        "# Thresholded MASD advisor summary",
        "",
        f"- Replacement verdict: {'yes' if replaceable else 'no'}; keep `current_full` if strict replacement is required.",
        f"- Most credible gain source: {'external holdout' if float(compare_row['external_delta_mean_k']) >= max(float(compare_row['main_delta_mean_k']), float(compare_row['hard_delta_mean_k'])) else ('hard subgroup' if float(compare_row['hard_delta_mean_k']) >= float(compare_row['main_delta_mean_k']) else 'main test')}.",
        f"- `other` remains {'the main risk' if float(other_row['delta_vs_current_full_k']) < 0.0 else 'under control'} with delta {float(other_row['delta_vs_current_full_k']):+.4f} K.",
        f"- MASD sparsity check: main activation rate {float(activation_mean['main_activation_rate']):.4f}, external activation rate {float(activation_mean['external_activation_rate']):.4f}, main mean correction {float(activation_mean['main_mean_correction']):+.4f} K.",
        f"- PR-writing readiness: {'improved' if replaceable or float(compare_row['external_delta_mean_k']) > 0.30 else 'not materially improved'} relative to `current_full`.",
    ]
    (RUN_DIR / "thresholded_masd_summary_for_advisor.md").write_text("\n".join(advisor_lines) + "\n", encoding="utf-8")


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    ensure_msce_features()
    masd_run.ensure_gpu()
    dataset, features, splits = load_artifacts()
    masd_run.CHEMISTRY_TAG_LOOKUP = masd_run.build_chemistry_tag_lookup(dataset)
    cluster_masks = masd_run.external_cluster_masks(dataset)
    config = diagnostic_config()
    epoch_log: list[float] = []

    base_screen = {
        seed: ensure_base_seed(
            cache=cache,
            dataset=dataset,
            features=features,
            splits=splits,
            config=config,
            seed=seed,
            epoch_log=epoch_log,
        )
        for seed in SCREEN_SEEDS
    }

    screen_records: list[dict[str, Any]] = []
    candidates = candidate_grid()
    for candidate in candidates:
        for seed in SCREEN_SEEDS:
            result = run_thresholded_candidate(
                cache=cache,
                base_payload=base_screen[seed],
                config=config,
                seed=seed,
                candidate=candidate,
                cluster_masks=cluster_masks,
                epoch_log=epoch_log,
                store_group="screen",
            )
            screen_records.append(
                {
                    "label": str(candidate["label"]),
                    "seed": int(seed),
                    "tau": float(candidate["tau"]),
                    "bound_k": float(candidate["bound_k"]),
                    "lambda_mean": float(candidate["mean_lambda"]),
                    "lambda_sparse": float(candidate["sparse_lambda"]),
                    **result["metrics"],
                    "activation_rate_main": float(result["activation_primary"]["activation_rate"]),
                    "mean_correction_main": float(result["activation_primary"]["mean_correction"]),
                }
            )
    screen_df = screen_summary_df(screen_records)
    best_candidate = choose_screen_winner(screen_df)

    base_final = {
        seed: ensure_base_seed(
            cache=cache,
            dataset=dataset,
            features=features,
            splits=splits,
            config=config,
            seed=seed,
            epoch_log=epoch_log,
        )
        for seed in FINAL_SEEDS
    }
    best_final = {
        seed: run_thresholded_candidate(
            cache=cache,
            base_payload=base_final[seed],
            config=config,
            seed=seed,
            candidate=best_candidate,
            cluster_masks=cluster_masks,
            epoch_log=epoch_log,
            store_group="final",
        )
        for seed in FINAL_SEEDS
    }

    five_seed_df = aggregate_5seed_rows(base_results=base_final, best_results=best_final)
    cluster_table = cluster_df(base_results=base_final, best_results=best_final, cluster_masks=cluster_masks)
    activation_table = activation_df(best_final)
    write_outputs(
        screen_df=screen_df.sort_values(by=["pass_screen", "external_delta_mean", "main_delta_mean"], ascending=[False, False, False]).reset_index(drop=True),
        best_candidate=best_candidate,
        five_seed_df=five_seed_df,
        cluster_table=cluster_table,
        activation_table=activation_table,
    )
    summary = {
        "best_candidate": best_candidate,
        "epoch_time_mean_sec": float(np.mean(epoch_log)) if epoch_log else float("nan"),
    }
    (RUN_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
