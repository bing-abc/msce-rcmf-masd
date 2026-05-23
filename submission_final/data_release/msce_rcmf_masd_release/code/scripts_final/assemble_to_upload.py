from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from _submission_utils import ROOT


def build_zip(zip_path: Path, base_dir: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(base_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(base_dir))


def main() -> None:
    submission = ROOT / "submission_final"
    upload = submission / "TO_UPLOAD"
    stage = submission / "_manuscript_source_stage"

    for path in [upload, stage]:
        if path.exists():
            shutil.rmtree(path)

    (upload).mkdir(parents=True, exist_ok=True)
    (stage / "source" / "figures").mkdir(parents=True, exist_ok=True)
    (stage / "supplementary" / "supplementary_figures").mkdir(parents=True, exist_ok=True)

    copy_pairs = [
        (submission / "source" / "main.tex", stage / "source" / "main.tex"),
        (submission / "source" / "main.bbl", stage / "source" / "main.bbl"),
        (submission / "source" / "references.bib", stage / "source" / "references.bib"),
        (submission / "source" / "extra.bib", stage / "source" / "extra.bib"),
        (submission / "supplementary" / "supplementary_information.tex", stage / "supplementary" / "supplementary_information.tex"),
        (submission / "supplementary" / "table_s4_statistical_tests.csv", stage / "supplementary" / "table_s4_statistical_tests.csv"),
    ]
    for src, dst in copy_pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for src_dir, dst_dir in [
        (submission / "source" / "figures", stage / "source" / "figures"),
        (submission / "supplementary" / "supplementary_figures", stage / "supplementary" / "supplementary_figures"),
    ]:
        for item in sorted(src_dir.iterdir()):
            target = dst_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    build_zip(upload / "manuscript_source.zip", stage)
    build_zip(upload / "Source_Data.zip", submission / "source_data")

    upload_pairs = [
        (submission / "source" / "Manuscript.pdf", upload / "Manuscript.pdf"),
        (submission / "supplementary" / "supplementary_information.pdf", upload / "Supplementary_Information.pdf"),
        (submission / "Highlights.txt", upload / "Highlights.txt"),
        (submission / "Graphical_Abstract.pdf", upload / "Graphical_Abstract.pdf"),
        (submission / "Graphical_Abstract.png", upload / "Graphical_Abstract.png"),
        (submission / "cover_letter" / "cover_letter_materials_design.docx", upload / "Cover_Letter.docx"),
        (submission / "cover_letter" / "conflict_of_interest_statement.docx", upload / "Conflict_of_interest_statement.docx"),
        (submission / "data_release" / "README_for_Zenodo.md", upload / "Data_release_readme.md"),
    ]
    for src, dst in upload_pairs:
        shutil.copy2(src, dst)

    for idx in range(1, 6):
        shutil.copy2(submission / "figures" / f"Figure_{idx}.pdf", upload / f"Figure_{idx}.pdf")

    shutil.rmtree(stage)


if __name__ == "__main__":
    main()
