from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_document_text, word_count
from .report import write_summary_report
from .summarizer import LocalSummarizer, ProgressEvent, detect_document_language


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize long documents locally and export DOCX reports",
    )
    parser.add_argument("input_file", type=Path, help="Path to source document (.txt, .md, .docx, .pdf)")
    parser.add_argument(
        "--level",
        choices=["short", "medium", "long", "all"],
        default="all",
        help="Summary detail level",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output DOCX file path (default: ./output/<input_name>_summary.docx)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional custom local Hugging Face model name for all languages",
    )
    parser.add_argument(
        "--language",
        choices=["auto", "en", "es"],
        default="auto",
        help="Document language. Use auto to detect and preserve source language",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Execution device for local model inference",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=2,
        help="Beam search width for generation. Lower is faster, higher can improve quality (default: 2)",
    )
    return parser


def resolve_output_path(input_file: Path, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output
    return Path("output") / f"{input_file.stem}_summary.docx"


class SingleLineProgress:
    """Renders a single-line progress indicator for one `summarize()` call.

    Percent complete is tracked as fraction of *pipeline steps* (chunks + a
    reserved final step + any merge rounds discovered along the way), not just
    chunk index. Otherwise a single-chunk document jumps to "100%" after its
    one chunk finishes while the (often equally slow) final reduction pass is
    still running.
    """

    def __init__(self) -> None:
        self._last_len = 0
        self._total_units = 0
        self._completed_units = 0

    def _write_line(self, text: str) -> None:
        padding = " " * max(0, self._last_len - len(text))
        print(f"\r{text}{padding}", end="", flush=True)
        self._last_len = len(text)

    def _percent(self) -> int:
        if self._total_units <= 0:
            return 0
        if self._completed_units >= self._total_units:
            return 100
        return min(99, int((self._completed_units / self._total_units) * 100))

    def update(self, event: ProgressEvent) -> None:
        phase = event.get("phase", "")
        level = event.get("level", "")
        language = event.get("language", "")
        total = int(event.get("chunk_total", 0) or 0)
        index = int(event.get("chunk_index", 0) or 0)
        chunks_done = index if phase == "chunk_done" else max(index - 1, 0)

        if phase == "init":
            self._total_units = total + 1  # +1 reserved for the final reduction step
            self._completed_units = 0
            text = f"[progress] level={level} lang={language} chunks=0/{total} {self._percent()}% phase=chunking"
        elif phase in {"chunk_started", "chunk_done"}:
            if phase == "chunk_done":
                self._completed_units += 1
            text = f"[progress] level={level} lang={language} chunks={chunks_done}/{total} {self._percent()}% phase=chunking"
        elif phase == "merge_started":
            self._total_units += 1  # discovered mid-run, not counted at init
            round_number = int(event.get("merge_round", 0) or 0)
            text = f"[progress] level={level} lang={language} chunks={total}/{total} {self._percent()}% phase=merge round={round_number}"
        elif phase == "merge_done":
            self._completed_units += 1
            round_number = int(event.get("merge_round", 0) or 0)
            text = f"[progress] level={level} lang={language} chunks={total}/{total} {self._percent()}% phase=merge-done round={round_number}"
        elif phase == "final_started":
            text = f"[progress] level={level} lang={language} chunks={total}/{total} {self._percent()}% phase=finalizing"
        elif phase == "final_done":
            self._completed_units = self._total_units
            text = f"[progress] level={level} lang={language} chunks={total}/{total} 100% phase=done"
        else:
            text = f"[progress] level={level} lang={language} phase={phase}"

        self._write_line(text)

    def finish(self) -> None:
        print()


def main() -> None:
    args = build_parser().parse_args()

    text = read_document_text(args.input_file)
    source_words = word_count(text)
    detected_language = detect_document_language(text)
    document_language = detected_language if args.language == "auto" else args.language

    summarizer = LocalSummarizer(model_name=args.model, device=args.device, num_beams=args.num_beams)
    progress = SingleLineProgress()

    summaries = {}
    if args.level == "all":
        # "long" is the most detailed level, so it's generated first straight from
        # the source text; "medium" and "short" are then derived by condensing that
        # summary further instead of re-chunking and re-summarizing the whole
        # source document from scratch for each level.
        summaries["long"] = summarizer.summarize(
            text, level="long", language=document_language, progress_callback=progress.update
        )
        progress.finish()
        summaries["medium"] = summarizer.summarize(
            summaries["long"], level="medium", language=document_language, progress_callback=progress.update
        )
        progress.finish()
        summaries["short"] = summarizer.summarize(
            summaries["medium"], level="short", language=document_language, progress_callback=progress.update
        )
        progress.finish()
    else:
        summaries[args.level] = summarizer.summarize(
            text, level=args.level, language=document_language, progress_callback=progress.update
        )
        progress.finish()

    selected_model = summarizer.model_for_language(document_language)

    output_path = resolve_output_path(args.input_file, args.output)
    final_path = write_summary_report(
        output_path=output_path,
        source_file=args.input_file,
        summaries=summaries,
        source_word_count=source_words,
        model_name=selected_model,
        document_language=document_language,
    )

    print(f"Detected language: {detected_language}")
    print(f"Summary language: {document_language}")
    print(f"Summary report generated: {final_path}")


if __name__ == "__main__":
    main()
