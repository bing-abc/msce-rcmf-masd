from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from _submission_utils import (
    ROOT,
    bibitem_order,
    contains_type3,
    count_words,
    extract_caption_for_label,
    extract_pdf_text,
    find_citation_keys_in_order,
    find_lines_with_terms,
    markdown_table,
    normalize_ws,
    pdf_font_types,
    png_dimensions,
    read_text,
    ref_labels_in_order,
    ref_mentions_in_order,
    table_column_specs,
    tex_environment,
    tex_section_block,
    write_text,
)


OVERCLAIM_TERMS = [
    "universal generalization",
    "material discovery",
    "guarantees",
    "fully reliable",
    "de novo discovery",
    "broad external utility",
    "state-of-the-art",
]


def add_issue(issues: list[dict[str, str]], level: str, category: str, item: str, status: str, detail: str, recommendation: str) -> None:
    issues.append(
        {
            "priority": level,
            "category": category,
            "item": item,
            "status": status,
            "detail": detail,
            "recommendation": recommendation,
        }
    )


def parse_bib_file_stats(text: str) -> tuple[int, int]:
    entries = re.findall(r"@\w+\{", text)
    conference_like = re.findall(r"\bbooktitle\s*=", text, flags=re.I)
    arxiv_like = re.findall(r"arxiv|preprint", text, flags=re.I)
    return len(entries), len(conference_like) + len(arxiv_like)


def main() -> None:
    base = ROOT / "submission_final"
    source_dir = base / "source"
    figures_dir = base / "figures"
    supp_dir = base / "supplementary"
    reports_dir = base / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    manuscript_pdf = source_dir / "Manuscript.pdf"
    main_tex = source_dir / "main.tex"
    main_bbl = source_dir / "main.bbl"
    references_bib = source_dir / "references.bib"
    extra_bib = source_dir / "extra.bib"
    highlights = base / "Highlights.txt"
    ga_png = base / "Graphical_Abstract.png"
    supp_tex = supp_dir / "supplementary_information.tex"
    supp_pdf = supp_dir / "supplementary_information.pdf"

    tex_text = read_text(main_tex)
    supp_text = read_text(supp_tex)
    bbl_text = read_text(main_bbl)
    ref_text = read_text(references_bib) + "\n" + read_text(extra_bib)
    abstract = normalize_ws(tex_environment(tex_text, "abstract"))
    abstract_words = count_words(abstract)
    cite_keys = find_citation_keys_in_order(tex_text)
    bib_keys = bibitem_order(bbl_text)
    supp_cites = find_citation_keys_in_order(supp_text)
    pdf_text = extract_pdf_text(manuscript_pdf)

    issues: list[dict[str, str]] = []

    # A. Abstract
    abstract_has_refs = bool(re.search(r"\\cite[a-zA-Z*]*\{", tex_environment(tex_text, "abstract")))
    overclaim_hits = find_lines_with_terms(abstract, OVERCLAIM_TERMS)
    if abstract_words > 200:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Abstract",
            "Word limit",
            "FAIL",
            f"Abstract has {abstract_words} words.",
            "Reduce abstract to 200 words or fewer.",
        )
    else:
        add_issue(
            issues,
            "OPTIONAL",
            "Abstract",
            "Word limit",
            "PASS",
            f"Abstract has {abstract_words} words.",
            "No action required.",
        )
    if abstract_has_refs:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Abstract",
            "References in abstract",
            "FAIL",
            "Abstract contains a citation command.",
            "Remove references from the abstract.",
        )
    if overclaim_hits:
        detail = "; ".join(f"{term}: {line}" for _, term, line in overclaim_hits)
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Abstract",
            "Overclaim language",
            "FAIL",
            detail,
            "Replace overclaim language with bounded, evidence-supported wording.",
        )

    # B. Highlights
    highlight_lines = [line.strip() for line in read_text(highlights).splitlines() if line.strip()]
    highlight_rows = [[str(len(line)), line] for line in highlight_lines]
    if not (3 <= len(highlight_lines) <= 5):
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Highlights",
            "Highlight count",
            "FAIL",
            f"Found {len(highlight_lines)} highlight lines.",
            "Keep 3–5 highlight lines.",
        )
    if any(len(line) > 85 for line in highlight_lines):
        too_long = [line for line in highlight_lines if len(line) > 85]
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Highlights",
            "Character limit",
            "FAIL",
            " | ".join(f"{len(line)} chars: {line}" for line in too_long),
            "Reduce each highlight to 85 characters or fewer including spaces.",
        )

    # C. Graphical abstract
    if not ga_png.exists():
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Graphical abstract",
            "File presence",
            "FAIL",
            "Graphical abstract image is missing.",
            "Provide a separate graphical abstract file.",
        )
        ga_dims = (0, 0)
    else:
        ga_dims = png_dimensions(ga_png)
        if ga_dims[0] < 1328 or ga_dims[1] < 531:
            add_issue(
                issues,
                "MUST FIX BEFORE SUBMISSION",
                "Graphical abstract",
                "Minimum dimensions",
                "FAIL",
                f"Graphical abstract dimensions are {ga_dims[0]}x{ga_dims[1]}.",
                "Re-export the graphical abstract above the minimum size requirement.",
            )
    ga_builder = ROOT / "scripts_final" / "rebuild_graphical_abstract.py"
    if not ga_builder.exists():
        ga_builder = ROOT / "scripts_md_revision" / "build_graphical_abstract.py"
    ga_builder_text = read_text(ga_builder) if ga_builder.exists() else ""
    ga_text_blocks = re.findall(r'ax\.text\([^,]+,[^,]+,\s*"([^"]+)"', ga_builder_text)
    if any("Graphical Abstract" in block for block in ga_text_blocks):
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Graphical abstract",
            "Forbidden in-image heading",
            "FAIL",
            "Builder script contains the phrase 'Graphical Abstract'.",
            "Remove the literal 'Graphical Abstract' heading from the artwork.",
        )
    ga_word_count = sum(count_words(block.replace("\\n", " ")) for block in ga_text_blocks)
    if ga_word_count > 45:
        add_issue(
            issues,
            "STRONGLY RECOMMENDED",
            "Graphical abstract",
            "Text density",
            "WARN",
            f"Graphical abstract builder contains roughly {ga_word_count} words across text blocks.",
            "Keep the graphical abstract concise and readable at small size.",
        )

    # D. References
    missing_from_bbl = [key for key in cite_keys if key not in bib_keys]
    unused_bib = [key for key in bib_keys if key not in cite_keys]
    appearance_vs_bbl = cite_keys == bib_keys[: len(cite_keys)]
    if missing_from_bbl:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "References",
            "Missing reference entry",
            "FAIL",
            ", ".join(missing_from_bbl),
            "Ensure every cited key appears in the bibliography list.",
        )
    if unused_bib:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "References",
            "Unused bibliography entries",
            "FAIL",
            ", ".join(unused_bib),
            "Remove bibliography entries that are not cited in the manuscript.",
        )
    if not appearance_vs_bbl:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "References",
            "Order of appearance",
            "FAIL",
            "Bibliography order does not match first citation order.",
            "Re-run BibTeX and ensure numeric references follow first appearance order.",
        )
    if not abstract_has_refs:
        add_issue(
            issues,
            "OPTIONAL",
            "References",
            "Abstract references",
            "PASS",
            "Abstract contains no citations.",
            "No action required.",
        )
    total_bib_entries, conference_arxiv_like = parse_bib_file_stats(ref_text)
    ratio = conference_arxiv_like / total_bib_entries if total_bib_entries else 0.0
    if ratio > 0.35:
        add_issue(
            issues,
            "STRONGLY RECOMMENDED",
            "References",
            "Conference/arXiv ratio",
            "WARN",
            f"Detected {conference_arxiv_like} conference/arXiv-like entries out of {total_bib_entries} bibliography entries.",
            "Prefer journal and materials-facing anchors where possible.",
        )

    # E. Tables
    table_labels = ref_labels_in_order(tex_text, "tab:")
    table_refs = ref_mentions_in_order(tex_text, "tab:")
    if table_refs != table_labels:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Tables",
            "Citation order",
            "FAIL",
            f"Table labels: {table_labels}; first refs: {table_refs}",
            "Cite Table 1–4 in the same order that they appear.",
        )
    captions_ok = all(extract_caption_for_label(tex_text, label) for label in table_labels)
    if not captions_ok:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Tables",
            "Captions",
            "FAIL",
            "One or more main-text tables are missing captions.",
            "Provide captions for all main-text tables.",
        )
    if any("|" in spec for spec in table_column_specs(tex_text)):
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Tables",
            "Vertical rules",
            "FAIL",
            "Tabular column specification contains vertical rules.",
            "Remove vertical table rules for journal style compliance.",
        )
    table2_caption = extract_caption_for_label(tex_text, "tab:overall_performance")
    if "five-seed sequence-baseline audit" not in table2_caption or "100-run statistical package" not in table2_caption:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Tables",
            "Table 2 mixed-run explanation",
            "FAIL",
            table2_caption,
            "Clarify that polyBERT is a five-seed audit and is not pooled with the 100-run package.",
        )
    table3_caption = extract_caption_for_label(tex_text, "tab:hard_external")
    if "Sign rate" not in table3_caption:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Tables",
            "Table 3 sign-rate definition",
            "FAIL",
            table3_caption,
            "Define sign rate explicitly in the Table 3 caption.",
        )
    table4_caption = extract_caption_for_label(tex_text, "tab:design_implications")
    if any(term in table4_caption.lower() for term in ["discovery", "state-of-the-art"]):
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Tables",
            "Table 4 overclaim",
            "FAIL",
            table4_caption,
            "Keep Table 4 retrospective and bounded; avoid discovery claims.",
        )

    # F. Figures
    figure_labels = ref_labels_in_order(tex_text, "fig:")
    figure_refs = ref_mentions_in_order(tex_text, "fig:")
    if figure_refs != figure_labels:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Figures",
            "Citation order",
            "FAIL",
            f"Figure labels: {figure_labels}; first refs: {figure_refs}",
            "Cite Figure 1–5 in the same order that they appear.",
        )
    figure_missing = []
    for idx in range(1, 6):
        if not (figures_dir / f"Figure_{idx}.pdf").exists():
            figure_missing.append(f"Figure_{idx}.pdf")
    if figure_missing:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Figures",
            "Separate files",
            "FAIL",
            ", ".join(figure_missing),
            "Provide each main figure as a separate file.",
        )
    for idx in range(1, 6):
        fig_pdf = figures_dir / f"Figure_{idx}.pdf"
        if fig_pdf.exists() and contains_type3(fig_pdf):
            add_issue(
                issues,
                "MUST FIX BEFORE SUBMISSION",
                "Figures",
                f"Type 3 fonts in Figure_{idx}",
                "FAIL",
                ", ".join(pdf_font_types(fig_pdf)),
                "Re-export the figure with Type 42 or embedded outline fonts.",
            )

    # G. Supplementary information
    supp_placeholders = re.findall(r"PENDING|TODO|Table 1:\s*Supplementary Table S1|Figure 1:\s*Supplementary Figure S1|Table 1:\s*\*|Table 2:\s*\*", supp_text)
    if supp_placeholders:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Supplementary Information",
            "Legacy placeholders",
            "FAIL",
            ", ".join(sorted(set(supp_placeholders))),
            "Remove placeholder or legacy caption text from the supplementary file.",
        )
    if "matched five-seed baseline slice" not in supp_text:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Supplementary Information",
            "Table S3 hard-subgroup clarification",
            "FAIL",
            "Table S3 wording does not clearly separate the matched five-seed audit from the 100-run package.",
            "Explain explicitly that Table S3 hard-subgroup values are audit-slice values and not the 100-run hard-subgroup package.",
        )
    if "Wilcoxon signed-rank" in supp_text and "Wilcoxon" not in supp_text:
        pass
    has_wilcoxon_column = "Wilcoxon signed-rank" in supp_text or "Wilcoxon" in supp_text
    s4_header_has_wilcoxon = bool(re.search(r"Wilcoxon", supp_text))
    if "Wilcoxon signed-rank tests" in supp_text and not s4_header_has_wilcoxon:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Supplementary Information",
            "Table S4 Wilcoxon results",
            "FAIL",
            "Supplementary methods claim Wilcoxon signed-rank tests, but Table S4 does not show a Wilcoxon column.",
            "Add the Wilcoxon p-value column to Table S4 or remove the manuscript claim.",
        )
    # numeric consistency spot-checks
    for needle in ["24.668", "23.981", "29.376", "25.152", "27.609", "27.205", "0.629", "24.463", "26.84"]:
        if needle not in supp_text and needle not in pdf_text:
            add_issue(
                issues,
                "STRONGLY RECOMMENDED",
                "Supplementary Information",
                "Numeric traceability",
                "WARN",
                f"Could not locate expected value `{needle}` in main/SI text during the quick scan.",
                "Check that key metrics remain traceable between the manuscript and SI.",
            )

    # H. Declarations
    declaration_blocks = {
        "Data Availability": tex_section_block(tex_text, "Data Availability"),
        "Declaration of Competing Interest": tex_section_block(tex_text, "Declaration of Competing Interest"),
        "Funding": tex_section_block(tex_text, "Funding"),
        "Acknowledgements": tex_section_block(tex_text, "Acknowledgements"),
        "CRediT authorship contribution statement": tex_section_block(tex_text, "CRediT authorship contribution statement"),
        "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process": tex_section_block(
            tex_text, "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process"
        ),
    }
    for name, block in declaration_blocks.items():
        if not block:
            add_issue(
                issues,
                "MUST FIX BEFORE SUBMISSION",
                "Declarations",
                name,
                "FAIL",
                f"Section `{name}` is missing.",
                "Add the missing declaration section.",
            )
    ai_block = declaration_blocks["Declaration of generative AI and AI-assisted technologies in the manuscript preparation process"]
    if "OpenAI ChatGPT" not in ai_block or "Anthropic Claude Code" not in ai_block:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Declarations",
            "AI tools disclosure",
            "FAIL",
            ai_block,
            "List the actual AI tools used in manuscript preparation.",
        )
    if "No generative AI tool was used to create or modify scientific images, raw data, or experimental results." not in ai_block:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Declarations",
            "AI artwork/raw-data restriction",
            "FAIL",
            ai_block,
            "State explicitly that no generative AI was used for scientific images, raw data, or experimental results.",
        )
    data_block = declaration_blocks["Data Availability"]
    required_data_terms = [
        "github.com",
        "split",
        "reproducibility",
        "result exports",
        "PolyMetriX",
        "Mendeley",
        "Figshare",
    ]
    missing_terms = [term for term in required_data_terms if term.lower() not in data_block.lower()]
    if missing_terms:
        add_issue(
            issues,
            "MUST FIX BEFORE SUBMISSION",
            "Declarations",
            "Data Availability completeness",
            "FAIL",
            ", ".join(missing_terms),
            "Expand Data Availability to include repository, processed splits, result exports, and upstream data accessions.",
        )
    if "zenodo doi" not in data_block.lower() and "[insert doi]" not in data_block.lower() and "doi-linked archival release" in data_block.lower():
        add_issue(
            issues,
            "STRONGLY RECOMMENDED",
            "Declarations",
            "Project DOI",
            "WARN",
            "Data Availability still relies on a future DOI-linked archival release rather than an already minted Zenodo/Mendeley DOI.",
            "Mint a DOI-backed archival release before submission if possible.",
        )

    # Write CSV
    csv_path = reports_dir / "md_compliance_check.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["priority", "category", "item", "status", "detail", "recommendation"])
        writer.writeheader()
        writer.writerows(issues)

    # Write markdown
    grouped = {"MUST FIX BEFORE SUBMISSION": [], "STRONGLY RECOMMENDED": [], "OPTIONAL": []}
    for issue in issues:
        grouped[issue["priority"]].append(issue)

    lines = [
        "# Materials & Design Compliance Check",
        "",
        f"- Abstract word count: `{abstract_words}`",
        f"- Highlights count: `{len(highlight_lines)}`",
        f"- Graphical abstract dimensions: `{ga_dims[0]} x {ga_dims[1]}`",
        f"- Bibliography entries in main.bbl: `{len(bib_keys)}`",
        "",
        "## Highlight Character Counts",
        "",
        markdown_table(["Chars", "Highlight"], highlight_rows),
        "",
    ]
    for heading in ("MUST FIX BEFORE SUBMISSION", "STRONGLY RECOMMENDED", "OPTIONAL"):
        lines.append(f"## {heading}")
        lines.append("")
        if not grouped[heading]:
            lines.append("- None.")
            lines.append("")
            continue
        for issue in grouped[heading]:
            lines.append(f"- `{issue['category']} / {issue['item']}`: {issue['status']}. {issue['detail']}")
            lines.append(f"  Recommendation: {issue['recommendation']}")
        lines.append("")

    write_text(reports_dir / "md_compliance_check.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
