from __future__ import annotations

import hashlib
from pathlib import Path

from _submission_utils import (
    ROOT,
    bibitem_order,
    extract_pdf_text,
    first_nonempty_line,
    markdown_table,
    normalize_ws,
    pdf_section_block,
    read_text,
    write_text,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_blocks(pdf_text: str) -> dict[str, str]:
    return {
        "title": first_nonempty_line(pdf_text),
        "abstract": normalize_ws(pdf_section_block(pdf_text, "Abstract", ["Keywords", "1. Introduction", "Introduction"])),
        "table2_caption": normalize_ws(
            pdf_section_block(pdf_text, "Table 2", ["Table 3", "Figure 2", "3.3", "3.2"])
        ),
        "figure2_caption": normalize_ws(
            pdf_section_block(pdf_text, "Figure 2", ["Figure 3", "Table 3", "3.3", "3.4"])
        ),
        "data_availability": normalize_ws(
            pdf_section_block(
                pdf_text,
                "Data Availability",
                [
                    "Declaration of Competing Interest",
                    "Funding",
                    "Acknowledgements",
                ],
            )
        ),
        "ai_declaration": normalize_ws(
            pdf_section_block(
                pdf_text,
                "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process",
                ["References", "[1]", "1. "],
            )
        ),
    }


def main() -> None:
    report_path = ROOT / "submission_final" / "reports" / "pdf_version_consistency_report.md"
    source_dir = ROOT / "submission_final" / "source"

    pdfs = {
        "Manuscript.pdf": source_dir / "Manuscript.pdf",
        "main.pdf": source_dir / "main.pdf",
        "full_materials_design_draft.pdf": ROOT / "manuscript_md_revision_files" / "full_materials_design_draft.pdf",
    }
    bbls = {
        "Manuscript.pdf": source_dir / "main.bbl",
        "main.pdf": source_dir / "main.bbl",
        "full_materials_design_draft.pdf": ROOT / "manuscript_md_revision_files" / "full_materials_design_draft.bbl",
    }

    texts = {name: extract_pdf_text(path) for name, path in pdfs.items()}
    hashes = {name: sha256(path) for name, path in pdfs.items()}
    blocks = {name: extract_blocks(text) for name, text in texts.items()}
    ref_orders = {name: bibitem_order(read_text(path)) for name, path in bbls.items()}

    section_names = ["title", "abstract", "table2_caption", "figure2_caption", "data_availability", "ai_declaration"]
    rows = []
    diffs: list[str] = []
    base_name = "Manuscript.pdf"
    for key in section_names:
        base_text = blocks[base_name][key]
        status = "MATCH"
        compared = []
        for other_name in ("main.pdf", "full_materials_design_draft.pdf"):
            same = base_text == blocks[other_name][key]
            compared.append(f"{other_name}: {'same' if same else 'different'}")
            if not same:
                status = "DIFF"
                diffs.append(
                    f"## Difference in {key}\n- `{base_name}`: {base_text[:500] or '[empty]'}\n- `{other_name}`: {blocks[other_name][key][:500] or '[empty]'}\n"
                )
        rows.append([key, status, "; ".join(compared)])

    ref_count_rows = []
    for name in pdfs:
        ref_count_rows.append([name, str(len(ref_orders[name])), hashes[name][:16]])
    ref_order_match = ref_orders[base_name] == ref_orders["main.pdf"] == ref_orders["full_materials_design_draft.pdf"]

    identical_hashes = len(set(hashes.values())) == 1
    canonical = source_dir / "Manuscript.pdf"

    lines = [
        "# PDF Version Consistency Report",
        "",
        f"- Canonical final manuscript PDF for submission: `{canonical}`",
        f"- SHA-256 identical across the three checked PDFs: `{identical_hashes}`",
        f"- Reference order identical across associated `.bbl` files: `{ref_order_match}`",
        "",
        "## Checked PDFs",
        "",
        markdown_table(["PDF", "Reference count from .bbl", "SHA-256 prefix"], ref_count_rows),
        "",
        "## Section-Level Consistency",
        "",
        markdown_table(["Section", "Status", "Comparison"], rows),
        "",
    ]

    if identical_hashes and ref_order_match and all(row[1] == "MATCH" for row in rows):
        lines.extend(
            [
                "## Verdict",
                "",
                "All three manuscript PDFs are textually consistent on the checked submission-critical sections. Only `Manuscript.pdf` should be retained in the final upload package; `main.pdf` and `full_materials_design_draft.pdf` should remain outside `TO_UPLOAD` as working or mirrored copies.",
            ]
        )
    else:
        lines.extend(
            [
                "## Verdict",
                "",
                "The manuscript PDF variants are not fully consistent. The latest `main.tex`-compiled `Manuscript.pdf` should be treated as authoritative, and the differences below must not be carried into the final upload package.",
                "",
            ]
        )
        lines.extend(diffs)

    write_text(report_path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
