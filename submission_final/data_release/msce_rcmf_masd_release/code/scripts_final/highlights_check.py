from __future__ import annotations

from _submission_utils import ROOT, markdown_table, read_text, write_text


def main() -> None:
    highlights_path = ROOT / "submission_final" / "Highlights.txt"
    report_path = ROOT / "submission_final" / "reports" / "highlights_check.md"

    lines = [line.strip() for line in read_text(highlights_path).splitlines() if line.strip()]
    rows = [[str(idx), str(len(line)), "PASS" if len(line) <= 85 else "FAIL", line] for idx, line in enumerate(lines, start=1)]

    verdict = "PASS" if 3 <= len(lines) <= 5 and all(len(line) <= 85 for line in lines) else "FAIL"
    report_lines = [
        "# Highlights Check",
        "",
        f"- Highlight count: `{len(lines)}`",
        f"- Verdict: `{verdict}`",
        "",
        markdown_table(["Line", "Chars", "Status", "Text"], rows),
        "",
    ]
    if not (3 <= len(lines) <= 5):
        report_lines.append("- Count check failed: Materials & Design expects 3 to 5 highlights.")
    if any(len(line) > 85 for line in lines):
        report_lines.append("- Character-limit check failed: each highlight must be 85 characters or fewer including spaces.")

    write_text(report_path, "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
