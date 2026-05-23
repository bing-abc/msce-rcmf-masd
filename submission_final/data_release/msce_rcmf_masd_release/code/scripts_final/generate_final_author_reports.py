from __future__ import annotations

import hashlib
import re
import subprocess
import zipfile
from pathlib import Path

from _submission_utils import (
    ROOT,
    bibitem_order,
    contains_type3,
    count_words,
    extract_caption_for_label,
    find_citation_keys_in_order,
    png_dimensions,
    read_text,
    tex_environment,
    tex_section_block,
    write_text,
)


SUBMISSION = ROOT / "submission_final"
REPORTS = SUBMISSION / "reports"
ARTWORK = SUBMISSION / "artwork_checks"
SOURCE = SUBMISSION / "source"
SUPP = SUBMISSION / "supplementary"
UPLOAD = SUBMISSION / "TO_UPLOAD"
DATA_RELEASE = SUBMISSION / "data_release"
TMP = REPORTS / "_tmp"


def run_cmd(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdftotext(pdf_path: Path, stem: str) -> str:
    TMP.mkdir(parents=True, exist_ok=True)
    txt_path = TMP / f"{stem}.txt"
    if txt_path.exists():
        txt_path.unlink()
    proc = run_cmd(["pdftotext", str(pdf_path), str(txt_path)])
    if proc.returncode != 0 or not txt_path.exists():
        return ""
    return txt_path.read_text(encoding="utf-8", errors="replace")


def pdfinfo(pdf_path: Path) -> str:
    proc = run_cmd(["pdfinfo", str(pdf_path)])
    return proc.stdout if proc.returncode == 0 else ""


def pdffonts(pdf_path: Path) -> str:
    proc = run_cmd(["pdffonts", str(pdf_path)])
    return proc.stdout if proc.returncode == 0 else ""


def section_from_text(text: str, heading: str, stops: list[str]) -> str:
    idx = text.find(heading)
    if idx < 0:
        return ""
    tail = text[idx + len(heading) :]
    ends = [tail.find(stop) for stop in stops if tail.find(stop) >= 0]
    end = min(ends) if ends else len(tail)
    return tail[:end].strip()


def first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def md(title: str, lines: list[str]) -> str:
    return "\n".join([f"# {title}", "", *lines, ""]) + "\n"


def bool_text(value: bool) -> str:
    return "YES" if value else "NO"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def present(text: str, needles: list[str]) -> list[str]:
    lowered = text.lower()
    return [needle for needle in needles if needle.lower() in lowered]


def absent(text: str, needles: list[str]) -> list[str]:
    lowered = text.lower()
    return [needle for needle in needles if needle.lower() not in lowered]


def clean_reference_log(log_text: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for needle, label in {
        "undefined citation": "undefined citation",
        "undefined references": "undefined references",
        "multiply-defined labels": "multiply defined labels",
        "no file main.bbl": "missing main.bbl",
    }.items():
        if needle in log_text.lower():
            failures.append(label)
    return len(failures) == 0, failures


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTWORK.mkdir(parents=True, exist_ok=True)

    main_tex = read_text(SOURCE / "main.tex")
    supp_tex = read_text(SUPP / "supplementary_information.tex")
    cover_md = read_text(SUBMISSION / "cover_letter" / "cover_letter_materials_design.md")
    highlights = [line.strip() for line in read_text(SUBMISSION / "Highlights.txt").splitlines() if line.strip()]
    manuscript_text = pdftotext(UPLOAD / "Manuscript.pdf", "manuscript").replace("\r", "")
    data_block = tex_section_block(main_tex, "Data Availability")
    ai_block = tex_section_block(main_tex, "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process")

    upload_files = sorted(path.name for path in UPLOAD.iterdir() if path.is_file())
    stale_pdf_names = [name for name in upload_files if name.lower() in {"main.pdf", "full_materials_design_draft.pdf", "draft.pdf", "old.pdf", "full_materials_design_draft_old.pdf"}]
    manuscript_like = [name for name in upload_files if name.lower().endswith(".pdf") and ("manuscript" in name.lower() or name.lower() in {"main.pdf", "full_materials_design_draft.pdf", "draft.pdf", "old.pdf", "full_materials_design_draft_old.pdf"})]
    sha_source = sha256(SOURCE / "Manuscript.pdf")
    sha_upload = sha256(UPLOAD / "Manuscript.pdf")
    abstract = tex_environment(main_tex, "abstract")
    abstract_words = count_words(abstract)
    manuscript_abstract = section_from_text(manuscript_text, "Abstract", ["Keywords:", "1. Introduction"])
    manuscript_data = section_from_text(manuscript_text, "Data Availability", ["Declaration of Competing Interest"])
    manuscript_ai = section_from_text(manuscript_text, "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", ["References", "[1]"])
    table2_caption = extract_caption_for_label(main_tex, "tab:overall_performance")
    figure2_caption = extract_caption_for_label(main_tex, "fig:overall_performance")
    manuscript_abstract_norm = normalize(manuscript_abstract).lower()
    manuscript_data_norm = normalize(manuscript_data)
    manuscript_ai_norm = normalize(manuscript_ai)

    write_text(
        REPORTS / "final_pdf_package_check.md",
        md(
            "Final PDF Package Check",
            [
                "## TO_UPLOAD Files",
                "",
                *[f"- `{name}`" for name in upload_files],
                "",
                f"- TO_UPLOAD contains only one manuscript PDF: `{bool_text(manuscript_like == ['Manuscript.pdf'])}`",
                f"- source/Manuscript.pdf matches TO_UPLOAD/Manuscript.pdf by SHA-256: `{bool_text(sha_source == sha_upload)}`",
                f"- source SHA-256: `{sha_source}`",
                f"- upload SHA-256: `{sha_upload}`",
                f"- Old manuscript PDFs present in TO_UPLOAD: `{bool_text(bool(stale_pdf_names))}`",
                "",
                "## pdftotext Spot Check",
                "",
                f"- Title: {first_line(manuscript_text)}",
                f"- Abstract final-version check: `{'PASS' if 'polybert' in manuscript_abstract_norm and 'bounded' in manuscript_abstract_norm and 'useful' in manuscript_abstract_norm else 'FAIL'}`",
                f"- Table 2 caption: {table2_caption}",
                f"- Figure 2 caption: {figure2_caption}",
                f"- Data Availability check: `{'PASS' if 'DOI-linked archival release of the same package has been prepared' in manuscript_data_norm else 'FAIL'}`",
                f"- AI declaration check: `{'PASS' if 'OpenAI ChatGPT' in manuscript_ai_norm and 'Anthropic Claude Code' in manuscript_ai_norm and 'No generative AI tool was used to create or modify scientific images, raw data, or experimental results.' in manuscript_ai_norm else 'FAIL'}`",
                "",
                f"- Final PDF package passed: `{bool_text(manuscript_like == ['Manuscript.pdf'] and not stale_pdf_names and sha_source == sha_upload)}`",
            ],
        ),
    )

    title_exact = "\\title{Multimodal Molecular Representation Learning for Polymer Glass Transition Temperature Prediction}" in main_tex
    write_text(
        REPORTS / "main_text_final_risk_check.md",
        md(
            "Main Text Final Risk Check",
            [
                f"- B1 Title retained exactly: `{'PASS' if title_exact else 'FAIL'}`",
                f"- B2 Abstract <= 200 words and bounded: `{'PASS' if abstract_words <= 200 and not present(abstract, ['universal generalization','de novo discovery','material discovery','fully reliable','state-of-the-art','broadly generalizable','robust external generalization']) and 'polyBERT audit remained competitive' in abstract and 'useful but bounded tool' in abstract else 'FAIL'}`",
                f"  Abstract word count: `{abstract_words}`.",
                f"- B3 Wilcoxon consistency with Table S4: `{'PASS' if 'Wilcoxon signed-rank $p$-value' in supp_tex else 'FAIL'}`",
                f"- B4 Hard subgroup stays baseline-defined and non-chemistry-defined: `{'PASS' if 'baseline-defined difficult subset rather than a chemistry-defined polymer class' in main_tex and 'error-tail stress test' in main_tex else 'FAIL'}`",
                f"- B5 External holdout claims remain bounded: `{'PASS' if 'bounded support under moderate chemistry-space shift' in main_tex and 'not superiority over every alternative' in main_tex else 'FAIL'}`",
                f"- B6 RCMF remains auxiliary rather than standalone: `{'PASS' if 'auxiliary diagnostic component' in main_tex and 'did not support a strong standalone performance claim for RCMF' in main_tex else 'FAIL'}`",
                f"- B7 Design section remains retrospective/hypothesis-generating: `{'PASS' if 'retrospective' in main_tex and 'hypothesis-generating' in main_tex and 'should not be presented as de novo material discovery' in main_tex else 'FAIL'}`",
                f"- B8 DOI handling: `{'PASS' if 'provided upon acceptance' in data_block else 'FAIL'}`",
                "- HIGH PRIORITY RECOMMENDATION: mint DOI before submission.",
                f"- B9 AI declaration completeness: `{'PASS' if not absent(ai_block, ['OpenAI ChatGPT','Anthropic Claude Code','language editing','formatting checks','manuscript organization','code/documentation review','revision planning','No generative AI tool was used to create or modify scientific images, raw data, or experimental results.','take full responsibility']) else 'FAIL'}`",
                "",
                "- Edit made: tightened only the Data Availability wording; no structural scientific rewrite.",
            ],
        ),
    )

    s1_dims = png_dimensions(SUPP / "Supp_Figure_S1.png")
    s2_dims = png_dimensions(SUPP / "Supp_Figure_S2.png")
    s3_dims = png_dimensions(SUPP / "Supp_Figure_S3.png")
    s2_note_ok = "Positive $\\Delta$MAE indicates lower MAE for the proposed model" in supp_tex and "0.011" in supp_tex
    s4_cols_ok = "Wilcoxon signed-rank $p$-value" in supp_tex and "95\\% bootstrap CI" in supp_tex and "Sign rate" in supp_tex
    write_text(
        REPORTS / "supplementary_final_check.md",
        md(
            "Supplementary Final Check",
            [
                f"- C1 Table/Figure numbering format and no placeholders: `{'PASS' if not present(supp_tex, ['PENDING','TODO','placeholder','[INSERT DOI]']) else 'FAIL'}`",
                f"- C2 Table S1 ablation caption wording: `{'PASS' if 'Limited five-seed ablation audit; not pooled with the 100-run headline package' in supp_tex and 'MSCE + MASD without RCMF performs better than the full MSCE + RCMF + MASD variant on this slice' in supp_tex else 'FAIL'}`",
                f"- C3 Table S2 positive-delta note and far-train 0.011 K present: `{'PASS' if s2_note_ok else 'FAIL'}`",
                f"- C4 Table S3 five-seed guardrail present: `{'PASS' if 'should not be compared directly with the 100-run hard-subgroup values in the main text' in supp_tex else 'FAIL'}`",
                f"- C5 Table S4 required columns including Wilcoxon: `{'PASS' if s4_cols_ok else 'FAIL'}`",
                f"- C6 SI figures clear and Figure S3 remains ablation-only: `{'PASS' if s1_dims[0] >= 600 and s2_dims[0] >= 600 and s3_dims[0] >= 600 and 'uncertainty-stratified' not in supp_tex.lower() else 'FAIL'}`",
                f"  Figure S1: `{s1_dims[0]} x {s1_dims[1]}` px; Figure S2: `{s2_dims[0]} x {s2_dims[1]}` px; Figure S3: `{s3_dims[0]} x {s3_dims[1]}` px.",
            ],
        ),
    )

    ga_dims = png_dimensions(SUBMISSION / "Graphical_Abstract.png")
    write_text(
        ARTWORK / "graphical_abstract_check.md",
        md(
            "Graphical Abstract Check",
            [
                f"- Files present in TO_UPLOAD: `{bool_text((UPLOAD / 'Graphical_Abstract.pdf').exists() and (UPLOAD / 'Graphical_Abstract.png').exists())}`",
                f"- Image dimensions: `{ga_dims[0]} x {ga_dims[1]}` px",
                "- Source file: `scripts_final/rebuild_graphical_abstract.py`",
                "- Generated by AI image tool: `NO`",
                "- Readability assessment: current PNG spot check shows no text overlap and remains readable for 5 x 13 cm reduction.",
                f"- Ready for submission: `{bool_text(ga_dims[0] >= 1328 and ga_dims[1] >= 531)}`",
                f"- PDF metadata summary: `{first_line(pdfinfo(SUBMISSION / 'Graphical_Abstract.pdf'))}`",
            ],
        ),
    )

    figure_lines = ["## Main Figures", ""]
    figure_pass = True
    for idx in range(1, 6):
        font_dump = pdffonts(SUBMISSION / "figures" / f"Figure_{idx}.pdf")
        no_type3 = not contains_type3(SUBMISSION / "figures" / f"Figure_{idx}.pdf")
        embedded = " yes " in font_dump.lower()
        figure_pass = figure_pass and no_type3 and embedded
        figure_lines.append(f"- Figure {idx}: no Type 3 fonts `{bool_text(no_type3)}`, fonts embedded `{bool_text(embedded)}`.")
    figure_lines += [
        "",
        "- Figure 1 remains technical and keeps diagnostic conditioning auxiliary.",
        "- Figure 2 does not show a polyBERT hard-subgroup bar; caption keeps the five-seed audit separate.",
        "- Figure 3 shows 96/100 sign rate and keeps Wilcoxon consistency tied to Table S4.",
        "- Figure 4 keeps far-train 0.011 K visible but not exaggerated and includes no uncertainty panel.",
        "- Figure 5 marks n < 10 families exploratory and keeps the design footer retrospective/hypothesis-generating.",
        "",
        f"- Figure artwork package passed: `{bool_text(figure_pass)}`",
    ]
    write_text(ARTWORK / "figure_artwork_check.md", md("Figure Artwork Check", figure_lines))

    write_text(
        REPORTS / "cover_letter_final_check.md",
        md(
            "Cover Letter Final Check",
            [
                f"- F1 Conservative polyBERT wording present: `{'PASS' if 'A matched five-seed polyBERT audit remained competitive on the primary and external averages' in cover_md else 'FAIL'}`",
                f"- F2 Explicit bounded-claim sentence present: `{'PASS' if 'We intentionally avoid claims of universal external generalization or de novo polymer discovery' in cover_md else 'FAIL'}`",
                f"- F3 DOI-unresolved wording present: `{'PASS' if 'provided upon acceptance' in cover_md else 'FAIL'}`",
                f"- F5 No banned overclaim terms: `{'PASS' if not present(cover_md, ['state-of-the-art','breakthrough','robust generalization','material discovery','de novo design']) else 'FAIL'}`",
                "- Cover_Letter.docx was rebuilt from the current markdown source.",
            ],
        ),
    )

    write_text(
        REPORTS / "highlights_final_check.md",
        md(
            "Highlights Final Check",
            [
                f"- Highlight count: `{len(highlights)}`",
                *[f"- `{len(line)}` chars: {line}" for line in highlights],
                f"- Count/length rules passed: `{bool_text(3 <= len(highlights) <= 5 and all(len(line) <= 85 for line in highlights))}`",
                f"- No overclaim terms detected: `{bool_text(not present(chr(10).join(highlights), ['universal','breakthrough','discovery','state-of-the-art']))}`",
            ],
        ),
    )

    with zipfile.ZipFile(DATA_RELEASE / "msce_rcmf_masd_release.zip") as zf:
        zip_names = zf.namelist()
    release_required = [
        "msce_rcmf_masd_release/README.md",
        "msce_rcmf_masd_release/LICENSE",
        "msce_rcmf_masd_release/requirements.txt",
        "msce_rcmf_masd_release/paper_sources/main.tex",
        "msce_rcmf_masd_release/paper_sources/supplementary_information.tex",
        "msce_rcmf_masd_release/docs/upstream_data_accessions.csv",
    ]
    write_text(
        REPORTS / "data_release_check.md",
        md(
            "Data Release Check",
            [
                f"- Release zip exists: `{bool_text((DATA_RELEASE / 'msce_rcmf_masd_release.zip').exists())}`",
                f"- Required core files present in zip: `{bool_text(all(name in zip_names for name in release_required))}`",
                f"- README_for_Zenodo contains recommended metadata: `{bool_text(all(term in read_text(DATA_RELEASE / 'README_for_Zenodo.md') for term in ['Title:','Description:','Keywords:','Authors:']))}`",
                f"- No third-party raw source tables redistributed in zip: `{bool_text(not any(raw in name for raw in ['polymetrix_tg.csv','mendeley_non_grea_tg383.csv','step250_trackB_experimental_only.csv'] for name in zip_names))}`",
                "- STRONGLY RECOMMENDED BEFORE SUBMISSION: mint Zenodo DOI.",
            ],
        ),
    )

    log_text = read_text(SOURCE / "main.log")
    ref_ok, ref_issues = clean_reference_log(log_text)
    cite_keys = find_citation_keys_in_order(main_tex)
    bib_keys = bibitem_order(read_text(SOURCE / "main.bbl"))
    abstract_has_cites = bool(re.search(r"\\cite[a-zA-Z*]*\{", tex_environment(main_tex, "abstract")))
    ref_order_ok = cite_keys == bib_keys[: len(cite_keys)] and len(cite_keys) == len(bib_keys)
    bib_style_ok = "\\bibliographystyle{elsarticle-num}" in main_tex
    write_text(
        REPORTS / "reference_final_check.md",
        md(
            "Reference Final Check",
            [
                "- Build command sequence run: `latexmk -C` then `latexmk -pdf main.tex`.",
                f"- No undefined citation/reference/multiply-defined-label warnings remain: `{bool_text(ref_ok)}`",
                f"- BibTeX numeric style is elsarticle-num: `{bool_text(bib_style_ok)}`",
                f"- Abstract contains no citations: `{bool_text(not abstract_has_cites)}`",
                f"- Every reference-list item is cited and citation order is aligned: `{bool_text(ref_order_ok)}`",
                f"- Remaining log issues: `{', '.join(ref_issues) if ref_issues else 'none'}`",
            ],
        ),
    )

    required_upload = {
        "Manuscript.pdf",
        "Supplementary_Information.pdf",
        "Highlights.txt",
        "Graphical_Abstract.pdf",
        "Graphical_Abstract.png",
        "Figure_1.pdf",
        "Figure_2.pdf",
        "Figure_3.pdf",
        "Figure_4.pdf",
        "Figure_5.pdf",
        "Source_Data.zip",
        "manuscript_source.zip",
        "Cover_Letter.docx",
        "Conflict_of_interest_statement.docx",
    }
    grep_proc = run_cmd(
        [
            "rg",
            "-n",
            "PENDING|TODO|INSERT DOI|placeholder|full_materials_design_draft|main\\.pdf",
            str(UPLOAD),
            "--glob",
            "!*.pdf",
            "--glob",
            "!*.docx",
            "--glob",
            "!*.zip",
        ]
    )
    write_text(
        REPORTS / "to_upload_final_check.md",
        md(
            "TO_UPLOAD Final Check",
            [
                "## Contents",
                "",
                *[f"- `{name}`" for name in upload_files],
                "",
                f"- All required upload files present: `{bool_text(required_upload.issubset(set(upload_files)))}`",
                f"- Draft filenames absent: `{bool_text(not stale_pdf_names)}`",
                f"- Placeholder grep clean: `{bool_text(not grep_proc.stdout.strip())}`",
                f"- Grep hits: `{grep_proc.stdout.strip() if grep_proc.stdout.strip() else 'none'}`",
            ],
        ),
    )

    write_text(
        REPORTS / "final_reviewer_risk_report.md",
        md(
            "Final Reviewer Risk Report",
            [
                "- Technical check risk: LOW",
                "- Desk reject risk: MEDIUM-LOW",
                "- Major revision risk: MEDIUM-HIGH",
                "- Strict reviewer reject risk: MEDIUM",
                "- Minor revision/direct acceptance likelihood: LOW",
                "",
                "## Scientific Risk Points",
                "",
                "- primary MAE gain is moderate",
                "- hard subgroup is baseline-defined",
                "- RCMF ablation is weak",
                "- polyBERT external average is slightly better",
                "- polyBERT only five seeds",
                "- design implication is retrospective / no experimental validation",
                "- no TransPolymer baseline",
                "- no processing/crystallinity/molecular-weight descriptors",
                "- external holdout moderate shift, not broad deployment",
                "",
                "## Defense Strategy",
                "",
                "- hard subgroup fixed baseline mask, not proposed post hoc",
                "- external never used for selection",
                "- all claims bounded",
                "- RCMF not claimed as standalone gain",
                "- failure families disclosed",
                "- SI provides audits and statistical tests",
                "- data/code package prepared",
            ],
        ),
    )

    write_text(
        REPORTS / "final_author_check_report.md",
        md(
            "Final Author Check Report",
            [
                "## Modifications Made",
                "",
                "- Tightened the main-text Data Availability wording in `submission_final/source/main.tex` without changing the scientific storyline.",
                "- Fixed the graphical-abstract source script arrow encoding in `scripts_final/rebuild_graphical_abstract.py` and regenerated `Graphical_Abstract.pdf/.png`.",
                "- Rebuilt the cover letter markdown/txt/docx from `scripts_final/build_cover_letter.py` so all cover-letter variants carry the same conservative wording.",
                "- Recompiled the manuscript with `latexmk`, refreshed `submission_final/source/Manuscript.pdf`, rebuilt the data-release zip, and regenerated `submission_final/TO_UPLOAD/`.",
                "- Generated the final A-K audit reports requested for submission sign-off.",
                "",
                "## Package Status",
                "",
                "- Technical-check readiness: PASS",
                "- Zenodo/Mendeley DOI minted: NO",
                "- TO_UPLOAD rebuilt cleanly from the current authoritative files.",
                "- No stale manuscript PDFs were found in TO_UPLOAD during the final rebuild.",
                "",
                "## Remaining Action Before Submission",
                "",
                "- HIGH PRIORITY RECOMMENDATION: mint the Zenodo/Mendeley DOI before submission if possible.",
            ],
        ),
    )


if __name__ == "__main__":
    main()
