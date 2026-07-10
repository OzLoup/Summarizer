from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

from docx import Document


def write_summary_report(
    output_path: Path,
    source_file: Path,
    summaries: Dict[str, str],
    source_word_count: int,
    model_name: str,
    document_language: str,
) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    language = document_language.lower().strip()
    is_spanish = language == "es"

    title = "Informe de Resumen del Documento" if is_spanish else "Document Summary Report"
    source_label = "Archivo de origen" if is_spanish else "Source file"
    source_count_label = "Conteo de palabras del origen" if is_spanish else "Source word count"
    model_label = "Modelo" if is_spanish else "Model"
    language_label = "Idioma del documento" if is_spanish else "Document language"
    generated_label = "Generado el" if is_spanish else "Generated at"

    section_titles = {
        "short": "Resumen Corto" if is_spanish else "Short Summary",
        "medium": "Resumen Medio" if is_spanish else "Medium Summary",
        "long": "Resumen Largo" if is_spanish else "Long Summary",
    }

    document = Document()
    document.add_heading(title, level=0)

    document.add_paragraph(f"{source_label}: {source_file}")
    document.add_paragraph(f"{source_count_label}: {source_word_count}")
    document.add_paragraph(f"{language_label}: {language}")
    document.add_paragraph(f"{model_label}: {model_name}")
    document.add_paragraph(f"{generated_label}: {datetime.now().isoformat(timespec='seconds')}")

    ordered_levels = ["short", "medium", "long"]
    for level in ordered_levels:
        if level not in summaries:
            continue
        document.add_heading(section_titles[level], level=1)
        document.add_paragraph(summaries[level])

    document.save(output_path)
    return output_path
