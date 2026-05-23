from __future__ import annotations

"""Small helper for paper-facing comparator summary tables."""

import pandas as pd

from eval.metrics import bootstrap_ci, format_ci, format_sign, sign_counts


def improve_pct(baseline_mae: float, model_mae: float) -> float:
    return float((baseline_mae - model_mae) / max(baseline_mae, 1e-8) * 100.0)


def comparator_report_rows(
    *,
    main_name: str,
    main_mae: list[float],
    comparator_map: dict[str, list[float]],
) -> pd.DataFrame:
    rows = []
    for name, values in comparator_map.items():
        deltas = [m - c for m, c in zip(main_mae, values, strict=True)]
        rows.append(
            {
                "main_model": main_name,
                "comparator_name": name,
                "ran": "yes",
                "reason_not_run": "",
                "main_absolute_metric_k": sum(main_mae) / len(main_mae),
                "comparator_absolute_metric_k": sum(values) / len(values),
                "delta_k_main_minus_comparator": sum(deltas) / len(deltas),
                "main_improve_pct_vs_comparator": improve_pct(
                    sum(values) / len(values),
                    sum(main_mae) / len(main_mae),
                ),
                "bootstrap_ci_k": format_ci(bootstrap_ci(deltas, seed=17)),
                "sign_win_loss": format_sign(sign_counts(deltas)),
                "comparator_closed": "yes",
            }
        )
    # The final summary row keeps downstream CSV consumers simple by recording
    # the main-model mean once even when there is no specific comparator.
    rows.append(
        {
            "main_model": main_name,
            "comparator_name": "summary",
            "ran": "yes",
            "reason_not_run": "",
            "main_absolute_metric_k": sum(main_mae) / len(main_mae),
            "comparator_absolute_metric_k": 0.0,
            "delta_k_main_minus_comparator": 0.0,
            "main_improve_pct_vs_comparator": 0.0,
            "bootstrap_ci_k": "",
            "sign_win_loss": "",
            "comparator_closed": "yes",
        }
    )
    return pd.DataFrame(rows)
