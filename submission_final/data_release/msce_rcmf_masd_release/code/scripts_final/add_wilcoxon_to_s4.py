from __future__ import annotations

import csv
import re
from pathlib import Path

from _submission_utils import ROOT, load_csv_rows, read_text, write_text


STAT_CSV = ROOT / "tables_md_revision" / "statistical_tests.csv"
SUPP_TEX = ROOT / "submission_final" / "supplementary" / "supplementary_information.tex"
OUT_CSV = ROOT / "submission_final" / "supplementary" / "table_s4_statistical_tests.csv"


def format_split(label: str) -> str:
    mapping = {
        "primary_MAE": "Primary test",
        "hard_subgroup_MAE": "Hard subgroup",
        "external_holdout_MAE": "External holdout",
    }
    return mapping[label]


def format_sign_rate(row: dict[str, str]) -> str:
    n = int(float(row["n"]))
    pos = int(round(float(row["sign_rate_positive"]) * n))
    return f"{pos}/{n}"


def build_table(rows: list[dict[str, str]]) -> str:
    body = [
        r"\begin{center}",
        r"\captionof{table}{Paired statistical tests across the frozen 100-run baseline-versus-proposed comparison. The reported intervals and sign rates quantify stability under training randomness rather than population-level inference over all possible polymer samples.}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.16\linewidth}cccccccc}",
        r"  \toprule",
        r"  Split & $N$ & Baseline mean MAE (K) & Proposed mean MAE (K) & Mean improvement (K) & 95\% bootstrap CI (K) & Paired $t$ $p$-value & Wilcoxon signed-rank $p$-value & Sign rate \\",
        r"  \midrule",
    ]
    for row in rows:
        body.append(
            "  {split} & {n} & {base:.3f} & {final:.3f} & {impr:.3f} & [{lo:.4f}, {hi:.4f}] & ${tp}$ & ${wp}$ & {sign} \\\\".format(
                split=format_split(row["label"]),
                n=int(float(row["n"])),
                base=float(row["mean_baseline_K"]),
                final=float(row["mean_final_K"]),
                impr=float(row["mean_improvement_K"]),
                lo=float(row["bootstrap_ci95_low"]),
                hi=float(row["bootstrap_ci95_high"]),
                tp=row["paired_t_pvalue"],
                wp=row["wilcoxon_pvalue"],
                sign=format_sign_rate(row),
            )
        )
    body.extend(
        [
            r"  \bottomrule",
            r"\end{tabular}}",
            r"\end{center}",
        ]
    )
    # normalize scientific notation for TeX
    text = "\n".join(body)
    text = re.sub(r"\$(\d+\.\d+)e-(\d+)\$", lambda m: f"${m.group(1)}\\times 10^{{-{int(m.group(2))}}}$", text)
    text = re.sub(r"\$(\d+\.\d+)e\+(\d+)\$", lambda m: f"${m.group(1)}\\times 10^{{{int(m.group(2))}}}$", text)
    return text


def main() -> None:
    rows = load_csv_rows(STAT_CSV)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "n",
                "mean_baseline_K",
                "mean_final_K",
                "mean_improvement_K",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "paired_t_pvalue",
                "wilcoxon_pvalue",
                "sign_rate_positive",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    tex = read_text(SUPP_TEX)
    pattern = (
        r"\\begin\{center\}\s*"
        r"\\captionof\{table\}\{Paired statistical tests across the frozen 100-run baseline-versus-proposed comparison\..*?\}"
        r".*?\\end\{center\}"
    )
    replacement = build_table(rows)
    tex_new = re.sub(pattern, lambda _: replacement, tex, count=1, flags=re.S)
    if tex_new == tex:
        raise RuntimeError("Failed to locate Table S4 block in supplementary_information.tex")
    write_text(SUPP_TEX, tex_new)


if __name__ == "__main__":
    main()
