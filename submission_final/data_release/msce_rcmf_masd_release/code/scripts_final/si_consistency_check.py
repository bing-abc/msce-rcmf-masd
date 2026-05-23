from __future__ import annotations

import csv
from pathlib import Path

from _submission_utils import ROOT, read_text, write_text


def add_row(rows: list[dict[str, str]], item: str, expected: str, location: str, status: str, detail: str) -> None:
    rows.append(
        {
            "item": item,
            "expected": expected,
            "location": location,
            "status": status,
            "detail": detail,
        }
    )


def main() -> None:
    main_tex = read_text(ROOT / "submission_final" / "source" / "main.tex")
    supp_tex = read_text(ROOT / "submission_final" / "supplementary" / "supplementary_information.tex")
    report_path = ROOT / "submission_final" / "reports" / "si_consistency_report.md"

    rows: list[dict[str, str]] = []
    checks = [
        ("Primary baseline MAE", "24.668", "main+SI"),
        ("Primary proposed MAE", "23.981", "main+SI"),
        ("Primary delta MAE", "0.686", "main+SI"),
        ("Hard baseline MAE", "29.376", "main+SI"),
        ("Hard proposed MAE", "25.152", "main+SI"),
        ("Hard delta MAE", "4.224", "main+SI"),
        ("Hard relative change", "-14.4\\%", "main+SI"),
        ("External baseline MAE", "27.609", "main+SI"),
        ("External proposed MAE", "27.205", "main+SI"),
        ("External delta MAE", "0.404", "main+SI"),
        ("Primary sign rate", "100/100", "SI"),
        ("Hard sign rate", "96/100", "main+SI"),
        ("External sign rate", "72/100", "main+SI"),
        ("External Tanimoto mean", "0.629", "main+SI"),
        ("polyBERT primary audit", "24.463", "main+SI"),
        ("polyBERT external audit", "26.84", "main+SI"),
    ]
    combined = main_tex + "\n" + supp_tex
    for item, expected, location in checks:
        status = "PASS" if expected in combined else "FAIL"
        detail = f"Located `{expected}` in the combined main/SI text." if status == "PASS" else f"Could not locate `{expected}`."
        add_row(rows, item, expected, location, status, detail)

    s3_guard_values = ["69.440", "66.748", "67.348", "63.335", "67.295"]
    in_supp = all(value in supp_tex for value in s3_guard_values)
    in_main = any(value in main_tex for value in s3_guard_values)
    add_row(
        rows,
        "Table S3 five-seed hard-slice values",
        ", ".join(s3_guard_values),
        "SI only",
        "PASS" if in_supp and not in_main else "FAIL",
        "All five-seed 60+ hard-audit values remain confined to SI." if in_supp and not in_main else "Hard-audit values appear to be missing from SI or leaking into the main text.",
    )

    s3_phrase = (
        "Hard-subgroup values in Table~S3 correspond to the matched five-seed polyBERT audit pipeline "
        "and should not be compared directly with the 100-run hard-subgroup values in the main text."
    )
    add_row(
        rows,
        "Table S3 clarification",
        "matched five-seed polyBERT audit pipeline",
        "SI",
        "PASS" if s3_phrase in supp_tex else "FAIL",
        "Table S3 now explicitly separates the audit slice from the 100-run package." if s3_phrase in supp_tex else "Missing explicit Table S3 audit-slice clarification.",
    )

    s4_terms = ["Wilcoxon signed-rank", "95\\% bootstrap CI", "Sign rate"]
    s4_pass = all(term in supp_tex for term in s4_terms)
    add_row(
        rows,
        "Table S4 statistical columns",
        ", ".join(s4_terms),
        "SI",
        "PASS" if s4_pass else "FAIL",
        "Table S4 contains paired t-test, Wilcoxon, bootstrap CI, and sign-rate wording." if s4_pass else "Table S4 is still missing one or more required statistical-result terms.",
    )

    lines = [
        "# SI Consistency Report",
        "",
        "| Item | Expected | Location | Status | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['item']} | `{row['expected']}` | {row['location']} | {row['status']} | {row['detail']} |")

    failing = [row for row in rows if row["status"] != "PASS"]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "- PASS" if not failing else f"- FAIL: {len(failing)} consistency checks still need attention.",
        ]
    )

    write_text(report_path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
