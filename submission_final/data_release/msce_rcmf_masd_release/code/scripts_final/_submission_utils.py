from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def tex_environment(tex: str, env: str) -> str:
    pattern = rf"\\begin\{{{re.escape(env)}\}}(.*?)\\end\{{{re.escape(env)}\}}"
    match = re.search(pattern, tex, flags=re.S)
    return match.group(1).strip() if match else ""


def tex_section_block(tex: str, heading: str) -> str:
    pattern = rf"\\section\*?\{{{re.escape(heading)}\}}(.*?)(?=\\section\*?\{{|\\bibliographystyle|\\end\{{document\}})"
    match = re.search(pattern, tex, flags=re.S)
    return match.group(1).strip() if match else ""


def pdf_section_block(text: str, heading: str, stop_markers: Iterable[str]) -> str:
    norm = text.replace("\r", "")
    idx = norm.find(heading)
    if idx < 0:
        return ""
    tail = norm[idx + len(heading) :]
    stops = [tail.find(marker) for marker in stop_markers if tail.find(marker) >= 0]
    end = min(stops) if stops else len(tail)
    return tail[:end].strip()


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def find_citation_keys_in_order(tex: str) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\\cite[a-zA-Z*]*\{([^}]*)\}", tex):
        keys = [key.strip() for key in match.group(1).split(",") if key.strip()]
        for key in keys:
            if key not in seen:
                seen.add(key)
                order.append(key)
    return order


def bibitem_order(bbl: str) -> list[str]:
    return re.findall(r"\\bibitem\{([^}]+)\}", bbl)


def ref_labels_in_order(tex: str, prefix: str) -> list[str]:
    pattern = rf"\\label\{{({re.escape(prefix)}[^}}]+)\}}"
    return re.findall(pattern, tex)


def ref_mentions_in_order(tex: str, prefix: str) -> list[str]:
    pattern = rf"\\ref\{{({re.escape(prefix)}[^}}]+)\}}"
    order: list[str] = []
    seen: set[str] = set()
    for label in re.findall(pattern, tex):
        if label not in seen:
            seen.add(label)
            order.append(label)
    return order


def extract_caption_for_label(tex: str, label: str) -> str:
    env_patterns = [
        r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}",
        r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}",
    ]
    for pattern in env_patterns:
        for match in re.finditer(pattern, tex, flags=re.S):
            block = match.group(1)
            if rf"\label{{{label}}}" not in block:
                continue
            cap_match = re.search(r"\\caption\{(.*?)\}", block, flags=re.S)
            if cap_match:
                return normalize_ws(cap_match.group(1))
    return ""


def table_column_specs(tex: str) -> list[str]:
    return re.findall(r"\\begin\{tabular\}\{([^}]*)\}", tex)


@dataclass
class FontRecord:
    name: str
    type: str


def pdffonts(pdf_path: Path) -> list[FontRecord]:
    try:
        proc = subprocess.run(
            ["pdffonts", str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    rows: list[FontRecord] = []
    for line in proc.stdout.splitlines()[2:]:
        cols = line.split()
        if len(cols) >= 2:
            font_type = cols[1] if cols[1] != "Type" else " ".join(cols[1:3])
            rows.append(FontRecord(name=cols[0], type=font_type))
    return rows


def pdf_font_types(pdf_path: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["pdffonts", str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    types: list[str] = []
    for line in proc.stdout.splitlines()[2:]:
        cols = line.split()
        if len(cols) >= 2:
            types.append(cols[1] if cols[1] != "Type" else " ".join(cols[1:3]))
    return types


def contains_type3(pdf_path: Path) -> bool:
    return any(font.type.startswith("Type 3") for font in pdffonts(pdf_path))


def png_dimensions(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.width, img.height


def find_lines_with_terms(text: str, terms: Iterable[str]) -> list[tuple[int, str, str]]:
    lines = text.splitlines()
    found: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines, start=1):
        low = line.lower()
        for term in terms:
            if term.lower() in low:
                found.append((idx, term, line.strip()))
    return found


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)
