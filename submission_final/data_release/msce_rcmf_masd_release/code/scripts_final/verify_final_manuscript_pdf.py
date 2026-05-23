from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from _submission_utils import ROOT, write_text


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text(pdf_path: Path, out_txt: Path) -> str:
    command = f'pdftotext "{pdf_path}" "{out_txt}"'
    subprocess.run(command, check=True, shell=True)
    return out_txt.read_text(encoding="utf-8", errors="ignore")


def snippet(text: str, marker: str, radius: int = 500) -> str:
    idx = text.find(marker)
    if idx < 0:
        return f"[MISSING] {marker}"
    start = max(0, idx - 80)
    end = min(len(text), idx + radius)
    return text[start:end].strip().replace("\r", "")


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    submission = ROOT / "submission_final"
    source_pdf = submission / "source" / "main.pdf"
    manuscript_pdf = submission / "source" / "Manuscript.pdf"
    upload_pdf = submission / "TO_UPLOAD" / "Manuscript.pdf"
    legacy_pdf = ROOT / "manuscript_md_revision_files" / "full_materials_design_draft.pdf"
    tmp_txt = submission / "reports" / "_manuscript_pdftotext.txt"
    report_path = submission / "reports" / "final_manuscript_pdf_verification.md"

    text = extract_text(upload_pdf, tmp_txt)

    normalized = normalize_ws(text)

    lines = [
        "# Final Manuscript PDF Verification",
        "",
        "## File identity",
        "",
        f"- `source/main.pdf` SHA-256: `{sha256(source_pdf)}`",
        f"- `source/Manuscript.pdf` SHA-256: `{sha256(manuscript_pdf)}`",
        f"- `TO_UPLOAD/Manuscript.pdf` SHA-256: `{sha256(upload_pdf)}`",
        f"- Legacy `full_materials_design_draft.pdf` SHA-256: `{sha256(legacy_pdf)}`",
        "",
        f"- `source/main.pdf` == `source/Manuscript.pdf`: `{'YES' if sha256(source_pdf) == sha256(manuscript_pdf) else 'NO'}`",
        f"- `source/Manuscript.pdf` == `TO_UPLOAD/Manuscript.pdf`: `{'YES' if sha256(manuscript_pdf) == sha256(upload_pdf) else 'NO'}`",
        f"- `TO_UPLOAD` contains only final manuscript PDF: `{'YES' if upload_pdf.exists() and not (submission / 'TO_UPLOAD' / 'main.pdf').exists() and not (submission / 'TO_UPLOAD' / 'full_materials_design_draft.pdf').exists() else 'NO'}`",
        "",
        "## pdftotext spot check",
        "",
        "### Title",
        "```text",
        snippet(normalized, "Multimodal Molecular Representation Learning for Polymer Glass Transition Temperature Prediction", 220),
        "```",
        "",
        "### Abstract",
        "```text",
        snippet(text, "Polymer glass transition temperature", 900),
        "```",
        "",
        "### Table 2 caption region",
        "```text",
        snippet(text, "Overall performance on the primary test set", 900),
        "```",
        "",
        "### Figure 2 caption region",
        "```text",
        snippet(text, "Overall performance with difficult-subset context", 1100),
        "```",
        "",
        "### Data Availability",
        "```text",
        snippet(text, "Data Availability", 1600),
        "```",
        "",
        "### AI declaration",
        "```text",
        snippet(text, "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", 1400),
        "```",
        "",
    ]
    write_text(report_path, "\n".join(lines))
    tmp_txt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
