from __future__ import annotations

import csv
import re

from _submission_utils import (
    ROOT,
    bibitem_order,
    contains_type3,
    count_words,
    find_citation_keys_in_order,
    markdown_table,
    png_dimensions,
    read_text,
    ref_labels_in_order,
    ref_mentions_in_order,
    tex_environment,
    tex_section_block,
    write_text,
)


def add_check(rows: list[dict[str, str]], item: str, status: str, detail: str) -> None:
    rows.append({"item": item, "status": status, "detail": detail})


def in_order(labels: list[str], mentions: list[str]) -> bool:
    mentioned_labels = [label for label in labels if label in mentions]
    return mentioned_labels == mentions


def only_negated_de_novo(text: str) -> bool:
    lowered = text.lower()
    for match in re.finditer(r"de novo discovery", lowered):
        window = lowered[max(0, match.start() - 12) : match.end()]
        if "not de novo discovery" not in window:
            return False
    return True


def main() -> None:
    base = ROOT / "submission_final"
    upload_dir = base / "TO_UPLOAD"
    source_dir = base / "source"
    supp_dir = base / "supplementary"
    reports_dir = base / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    main_tex = read_text(source_dir / "main.tex")
    supp_tex = read_text(supp_dir / "supplementary_information.tex")
    highlights_text = read_text(base / "Highlights.txt")
    cover_md_path = base / "cover_letter" / "cover_letter_materials_design.md"
    cover_md = read_text(cover_md_path) if cover_md_path.exists() else ""
    main_bbl = read_text(source_dir / "main.bbl")

    rows: list[dict[str, str]] = []

    combined_submission_text = "\n".join([main_tex, supp_tex, highlights_text, cover_md])
    bad_tokens = [token for token in ["PENDING", "TODO"] if token in combined_submission_text]
    add_check(
        rows,
        "No PENDING/TODO in manuscript, SI, or highlights",
        "PASS" if not bad_tokens else "FAIL",
        "No placeholder tokens detected." if not bad_tokens else f"Detected placeholder token(s): {', '.join(sorted(set(bad_tokens)))}.",
    )

    old_files = [name for name in ["main.pdf", "full_materials_design_draft.pdf"] if (upload_dir / name).exists()]
    add_check(
        rows,
        "No old filenames in submission package",
        "PASS" if not old_files else "FAIL",
        "No stale draft PDFs present in TO_UPLOAD." if not old_files else f"Found stale upload file(s): {', '.join(old_files)}.",
    )

    manuscript_candidates = [name for name in ["Manuscript.pdf", "main.pdf", "full_materials_design_draft.pdf"] if (upload_dir / name).exists()]
    add_check(
        rows,
        "Only one manuscript PDF in upload package",
        "PASS" if manuscript_candidates == ["Manuscript.pdf"] else "FAIL",
        f"Detected manuscript-like PDFs: {', '.join(manuscript_candidates) if manuscript_candidates else 'none'}.",
    )

    abstract_words = count_words(tex_environment(main_tex, "abstract"))
    add_check(
        rows,
        "Abstract <= 200 words",
        "PASS" if abstract_words <= 200 else "FAIL",
        f"Abstract word count = {abstract_words}.",
    )

    highlight_lines = [line.strip() for line in highlights_text.splitlines() if line.strip()]
    highlight_ok = 3 <= len(highlight_lines) <= 5 and all(len(line) <= 85 for line in highlight_lines)
    add_check(
        rows,
        "Highlights count and character limits",
        "PASS" if highlight_ok else "FAIL",
        "; ".join(f"{len(line)} chars: {line}" for line in highlight_lines),
    )

    main_table_labels = ref_labels_in_order(main_tex, "tab:")
    main_table_mentions = ref_mentions_in_order(main_tex, "tab:")
    main_fig_labels = ref_labels_in_order(main_tex, "fig:")
    main_fig_mentions = ref_mentions_in_order(main_tex, "fig:")
    supp_table_labels = ref_labels_in_order(supp_tex, "tab:")
    supp_table_mentions = ref_mentions_in_order(supp_tex, "tab:")
    supp_fig_labels = ref_labels_in_order(supp_tex, "fig:")
    supp_fig_mentions = ref_mentions_in_order(supp_tex, "fig:")
    citation_order_ok = all(
        [
            in_order(main_table_labels, main_table_mentions),
            in_order(main_fig_labels, main_fig_mentions),
            in_order(supp_table_labels, supp_table_mentions),
            in_order(supp_fig_labels, supp_fig_mentions),
        ]
    )
    add_check(
        rows,
        "All tables and figures cited in order",
        "PASS" if citation_order_ok else "FAIL",
        "Main and supplementary table/figure mentions match source order." if citation_order_ok else "Detected a citation-order mismatch in main text or SI.",
    )

    cite_keys = find_citation_keys_in_order(main_tex)
    bib_keys = bibitem_order(main_bbl)
    ref_match = cite_keys == bib_keys[: len(cite_keys)] and len(cite_keys) == len(bib_keys)
    add_check(
        rows,
        "References cited/list match",
        "PASS" if ref_match else "FAIL",
        f"First-appearance citations = {len(cite_keys)}; bibliography entries in main.bbl = {len(bib_keys)}.",
    )

    phrase = "confirms generalizability"
    add_check(
        rows,
        "No 'confirms generalizability' claim",
        "PASS" if phrase not in combined_submission_text.lower() else "FAIL",
        "Forbidden phrase absent." if phrase not in combined_submission_text.lower() else "Forbidden phrase detected.",
    )

    de_novo_ok = only_negated_de_novo(combined_submission_text)
    add_check(
        rows,
        "No positive 'de novo discovery' claim",
        "PASS" if de_novo_ok else "FAIL",
        "No positive 'de novo discovery' phrasing detected." if de_novo_ok else "Found a non-negated 'de novo discovery' phrase.",
    )

    rcmf_bad = bool(
        re.search(r"RCMF.{0,80}(standalone|independent).{0,50}(gain|accuracy|improv)", main_tex, flags=re.I | re.S)
        and "not support a standalone claim for RCMF" not in main_tex
    )
    add_check(
        rows,
        "RCMF not described as standalone accuracy gain",
        "PASS" if not rcmf_bad else "FAIL",
        "RCMF is framed as auxiliary/limited rather than as a standalone accuracy booster." if not rcmf_bad else "RCMF appears to be described as a standalone accuracy gain.",
    )

    polybert_external_ack = "slightly lower than the proposed model" in main_tex or "slightly better on the external average" in main_tex
    add_check(
        rows,
        "polyBERT external superiority acknowledged",
        "PASS" if polybert_external_ack else "FAIL",
        "Main text acknowledges the stronger polyBERT external average." if polybert_external_ack else "Main text does not clearly acknowledge the stronger polyBERT external average.",
    )

    add_check(
        rows,
        "Hard subgroup described as baseline-defined",
        "PASS" if "baseline-defined" in main_tex else "FAIL",
        "Baseline-defined hard-subgroup language is present." if "baseline-defined" in main_tex else "Missing baseline-defined hard-subgroup wording.",
    )

    s4_csv = supp_dir / "table_s4_statistical_tests.csv"
    s4_header = s4_csv.read_text(encoding="utf-8").splitlines()[0].lower() if s4_csv.exists() else ""
    has_wilcoxon = "wilcoxon" in s4_header
    main_claims_wilcoxon = "Wilcoxon" in main_tex
    add_check(
        rows,
        "Table S4 contains Wilcoxon or manuscript does not claim it",
        "PASS" if has_wilcoxon or not main_claims_wilcoxon else "FAIL",
        "Wilcoxon evidence is aligned between manuscript and Table S4." if has_wilcoxon or not main_claims_wilcoxon else "Main text claims Wilcoxon testing, but Table S4 lacks it.",
    )

    s3_guard = "should not be compared directly with the 100-run hard-subgroup values in the main text" in supp_tex
    add_check(
        rows,
        "Table S3 hard-subgroup audit not mixed with 100-run package",
        "PASS" if s3_guard else "FAIL",
        "Table S3 contains the required five-seed audit clarification." if s3_guard else "Table S3 lacks the hard-subgroup audit clarification.",
    )

    ga_pdf = upload_dir / "Graphical_Abstract.pdf"
    ga_png = upload_dir / "Graphical_Abstract.png"
    ga_ok = ga_pdf.exists() and ga_png.exists()
    ga_dims = png_dimensions(ga_png) if ga_png.exists() else (0, 0)
    ga_ok = ga_ok and ga_dims[0] >= 1328 and ga_dims[1] >= 531
    add_check(
        rows,
        "Graphical abstract exists and dimensions pass",
        "PASS" if ga_ok else "FAIL",
        f"Graphical abstract PNG dimensions = {ga_dims[0]} x {ga_dims[1]} px.",
    )

    figure_paths = [upload_dir / f"Figure_{idx}.pdf" for idx in range(1, 6)]
    figures_exist = all(path.exists() for path in figure_paths)
    add_check(
        rows,
        "Figures exist as separate files",
        "PASS" if figures_exist else "FAIL",
        "All Figure_1-Figure_5 PDFs are present." if figures_exist else "One or more main-figure PDFs are missing.",
    )

    figure_font_pass = all(not contains_type3(path) for path in figure_paths if path.exists())
    add_check(
        rows,
        "No Type 3 fonts in figure PDFs",
        "PASS" if figure_font_pass else "FAIL",
        "No Type 3 fonts detected in Figure_1-Figure_5 PDFs." if figure_font_pass else "At least one figure PDF still contains Type 3 fonts.",
    )

    required_source_data = [
        "Figure_1_source.csv",
        "Figure_2_source.csv",
        "Figure_3_source.csv",
        "Figure_4_source.csv",
        "Figure_5_source.csv",
        "Supplementary_Figure_S1_source.csv",
        "Supplementary_Figure_S2_source.csv",
        "Supplementary_Figure_S3_source.csv",
    ]
    source_data_ok = all((base / "source_data" / name).exists() for name in required_source_data) and (upload_dir / "Source_Data.zip").exists()
    add_check(
        rows,
        "Source data files exist",
        "PASS" if source_data_ok else "FAIL",
        "Required figure source-data files and Source_Data.zip are present." if source_data_ok else "Missing one or more required source-data files or Source_Data.zip.",
    )

    ai_block = tex_section_block(main_tex, "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process")
    ai_ok = all(term in ai_block for term in ["OpenAI ChatGPT", "Anthropic Claude Code"]) and "No generative AI tool was used to create or modify scientific images, raw data, or experimental results." in ai_block
    add_check(
        rows,
        "AI declaration includes ChatGPT/Claude and no AI artwork/raw data/results",
        "PASS" if ai_ok else "FAIL",
        "AI declaration contains the required tool names and exclusions." if ai_ok else "AI declaration is incomplete or missing required exclusions.",
    )

    data_block = tex_section_block(main_tex, "Data Availability")
    has_github = "github.com/bing-abc/msce-rcmf-masd" in data_block
    has_minted_doi = bool(re.search(r"(zenodo|mendeley data).{0,80}10\.\d{4,9}/", data_block, flags=re.I | re.S))
    has_unresolved_warning = "provided upon acceptance" in data_block.lower() or "prepared for zenodo/mendeley data" in data_block.lower()
    data_status = "PASS" if has_github and (has_minted_doi or has_unresolved_warning) else "FAIL"
    if data_status == "PASS" and not has_minted_doi:
        data_status = "WARN"
    add_check(
        rows,
        "Data Availability has GitHub and DOI or explicit unresolved warning",
        data_status,
        "GitHub link present; DOI already minted." if has_github and has_minted_doi else "GitHub link present; archival DOI still unresolved and only warned in text." if has_github and has_unresolved_warning else "Missing GitHub link or DOI/unresolved-warning language.",
    )

    report_md = reports_dir / "final_submission_audit.md"
    report_csv = reports_dir / "final_submission_audit.csv"

    with report_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    fail_count = sum(1 for row in rows if row["status"] == "FAIL")
    warn_count = sum(1 for row in rows if row["status"] == "WARN")
    pass_count = sum(1 for row in rows if row["status"] == "PASS")
    table_rows = [[row["item"], row["status"], row["detail"]] for row in rows]
    report_lines = [
        "# Final Submission Audit",
        "",
        f"- PASS: `{pass_count}`",
        f"- WARN: `{warn_count}`",
        f"- FAIL: `{fail_count}`",
        "",
        markdown_table(["Check", "Status", "Detail"], table_rows),
        "",
        "## Verdict",
        "",
        "- PASS" if fail_count == 0 and warn_count == 0 else "- PASS WITH WARNINGS" if fail_count == 0 else "- FAIL",
    ]
    write_text(report_md, "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
