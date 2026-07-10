from __future__ import annotations

from pathlib import Path
import re

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_document_text(file_path: Path) -> str:
    path = file_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{extension}'. Use one of: {supported}")

    if extension in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif extension == ".docx":
        document = Document(path)
        text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    else:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                pages.append(extracted)
        text = "\n\n".join(pages)

    text = normalize_whitespace(text)
    if not text:
        raise ValueError("No readable text found in the input file")
    return text


def word_count(text: str) -> int:
    if not text.strip():
        return 0
    return len(text.split())
