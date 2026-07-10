from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, TypedDict

from langdetect import DetectorFactory, LangDetectException, detect
import torch
from transformers import pipeline

from .io_utils import normalize_whitespace


@dataclass(frozen=True)
class LevelConfig:
    chunk_size_words: int
    chunk_overlap_words: int
    chunk_min_length: int
    chunk_max_length: int
    final_min_length: int
    final_max_length: int
    max_intermediate_words: int


LEVEL_CONFIGS: Dict[str, LevelConfig] = {
    "short": LevelConfig(
        chunk_size_words=750,
        chunk_overlap_words=80,
        chunk_min_length=45,
        chunk_max_length=95,
        final_min_length=80,
        final_max_length=140,
        max_intermediate_words=350,
    ),
    "medium": LevelConfig(
        chunk_size_words=850,
        chunk_overlap_words=90,
        chunk_min_length=70,
        chunk_max_length=140,
        final_min_length=150,
        final_max_length=260,
        max_intermediate_words=550,
    ),
    "long": LevelConfig(
        chunk_size_words=950,
        chunk_overlap_words=110,
        chunk_min_length=100,
        chunk_max_length=190,
        final_min_length=260,
        final_max_length=500,
        max_intermediate_words=900,
    ),
}

DetectorFactory.seed = 0
SUPPORTED_LANGUAGES = {"en", "es"}


class ProgressEvent(TypedDict, total=False):
    phase: str
    level: str
    language: str
    chunk_index: int
    chunk_total: int
    merge_round: int


def detect_document_language(text: str) -> str:
    sample = normalize_whitespace(text)[:4000]
    if not sample:
        return "en"

    try:
        detected = detect(sample).lower().strip()
    except LangDetectException:
        return "en"

    if detected.startswith("es"):
        return "es"
    if detected.startswith("en"):
        return "en"
    return "en"


class LocalSummarizer:
    DEFAULT_MODEL_BY_LANGUAGE = {
        "en": "sshleifer/distilbart-cnn-12-6",
        # mrm8488/bert2bert_shared-spanish-finetuned-summarization ships with several
        # encoder layers that load as uninitialized meta tensors under current
        # transformers/torch versions, so it silently fails and falls back to
        # returning the unsummarized input text. mT5_multilingual_XLSum loads and
        # generates correctly and is fine-tuned for summarization in Spanish.
        "es": "csebuetnlp/mT5_multilingual_XLSum",
    }

    def __init__(self, model_name: str | None = None, device: str = "auto", num_beams: int = 2) -> None:
        resolved_device = self._resolve_device(device)
        self.model_name = model_name
        self._resolved_device = resolved_device
        self._num_beams = num_beams
        self._pipelines: Dict[str, object] = {}

    def model_for_language(self, language: str) -> str:
        if self.model_name:
            return self.model_name
        return self.DEFAULT_MODEL_BY_LANGUAGE.get(language, self.DEFAULT_MODEL_BY_LANGUAGE["en"])

    def _get_pipeline_for_language(self, language: str):
        model_name = self.model_for_language(language)
        cache_key = f"{language}:{model_name}"
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]

        try:
            summarization_pipeline = pipeline(
                task="summarization",
                model=model_name,
                tokenizer=model_name,
                device=self._resolved_device,
            )
        except Exception as error:
            # Some model architectures (e.g. tied-weight encoder-decoder checkpoints)
            # can't be moved to certain accelerators and raise here rather than at
            # inference time. Fall back to CPU once instead of silently returning
            # unsummarized text on every later call.
            if self._resolved_device == -1:
                raise
            print(
                f"[warning] '{model_name}' failed to load on device "
                f"'{self._resolved_device}' ({error}); falling back to CPU.",
                file=sys.stderr,
            )
            summarization_pipeline = pipeline(
                task="summarization",
                model=model_name,
                tokenizer=model_name,
                device=-1,
            )

        self._pipelines[cache_key] = summarization_pipeline
        return summarization_pipeline

    @staticmethod
    def _resolve_device(device: str) -> int | str:
        normalized = device.lower().strip()
        has_mps = torch.backends.mps.is_available()
        if normalized == "auto":
            if torch.cuda.is_available():
                return 0
            if has_mps:
                return "mps"
            return -1
        if normalized == "cpu":
            return -1
        if normalized == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but no CUDA device is available")
            return 0
        if normalized == "mps":
            if not has_mps:
                raise RuntimeError("MPS was requested but no MPS device is available")
            return "mps"
        raise ValueError("Device must be one of: auto, cpu, cuda, mps")

    @staticmethod
    def _split_into_chunks(text: str, chunk_size_words: int, overlap_words: int) -> List[str]:
        words = text.split()
        if len(words) <= chunk_size_words:
            return [text]

        chunks: List[str] = []
        step = max(chunk_size_words - overlap_words, 1)
        for start in range(0, len(words), step):
            end = start + chunk_size_words
            chunk_words = words[start:end]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            if end >= len(words):
                break
        return chunks

    def _run_model(self, text: str, min_length: int, max_length: int, language: str) -> str:
        text = normalize_whitespace(text)
        words = len(text.split())
        if words < 40:
            return text

        adjusted_min = min(min_length, max(18, words // 3))
        adjusted_max = min(max_length, max(adjusted_min + 10, int(words * 0.9)))

        try:
            result = self._get_pipeline_for_language(language)(
                text,
                min_length=adjusted_min,
                max_length=adjusted_max,
                do_sample=False,
                truncation=True,
                num_beams=self._num_beams,
            )
            return normalize_whitespace(result[0]["summary_text"])
        except Exception:
            return text

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[ProgressEvent], None] | None,
        event: ProgressEvent,
    ) -> None:
        if progress_callback is not None:
            progress_callback(event)

    def summarize(
        self,
        text: str,
        level: str,
        language: str = "en",
        progress_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> str:
        normalized_level = level.lower().strip()
        if normalized_level not in LEVEL_CONFIGS:
            valid = ", ".join(LEVEL_CONFIGS.keys())
            raise ValueError(f"Invalid level '{level}'. Expected one of: {valid}")

        normalized_language = language.lower().strip()
        if normalized_language not in SUPPORTED_LANGUAGES:
            raise ValueError("Language must be one of: en, es")

        source_text = normalize_whitespace(text)
        if not source_text:
            raise ValueError("Cannot summarize empty text")

        config = LEVEL_CONFIGS[normalized_level]
        chunks = self._split_into_chunks(
            source_text,
            chunk_size_words=config.chunk_size_words,
            overlap_words=config.chunk_overlap_words,
        )

        self._emit_progress(
            progress_callback,
            {
                "phase": "init",
                "level": normalized_level,
                "language": normalized_language,
                "chunk_total": len(chunks),
            },
        )

        # A single chunk means there is nothing to map/reduce: summarizing it at
        # chunk-level length and then again at final-level length is redundant, so
        # the raw chunk is carried straight to the merge/final stage below.
        skip_chunk_pass = len(chunks) == 1

        chunk_summaries: List[str] = []
        for index, chunk in enumerate(chunks, start=1):
            self._emit_progress(
                progress_callback,
                {
                    "phase": "chunk_started",
                    "level": normalized_level,
                    "language": normalized_language,
                    "chunk_index": index,
                    "chunk_total": len(chunks),
                },
            )
            if skip_chunk_pass:
                chunk_summary = chunk
            else:
                chunk_summary = self._run_model(
                    chunk,
                    min_length=config.chunk_min_length,
                    max_length=config.chunk_max_length,
                    language=normalized_language,
                )
            chunk_summaries.append(chunk_summary)
            self._emit_progress(
                progress_callback,
                {
                    "phase": "chunk_done",
                    "level": normalized_level,
                    "language": normalized_language,
                    "chunk_index": index,
                    "chunk_total": len(chunks),
                },
            )

        merged = normalize_whitespace(" ".join(chunk_summaries))
        merge_round = 0
        while len(merged.split()) > config.max_intermediate_words:
            merge_round += 1
            self._emit_progress(
                progress_callback,
                {
                    "phase": "merge_started",
                    "level": normalized_level,
                    "language": normalized_language,
                    "chunk_total": len(chunks),
                    "merge_round": merge_round,
                },
            )
            merged = self._run_model(
                merged,
                min_length=config.final_min_length,
                max_length=config.final_max_length,
                language=normalized_language,
            )
            self._emit_progress(
                progress_callback,
                {
                    "phase": "merge_done",
                    "level": normalized_level,
                    "language": normalized_language,
                    "chunk_total": len(chunks),
                    "merge_round": merge_round,
                },
            )

        self._emit_progress(
            progress_callback,
            {
                "phase": "final_started",
                "level": normalized_level,
                "language": normalized_language,
                "chunk_total": len(chunks),
            },
        )
        # The last merge round already reduced `merged` using these exact final
        # bounds, so re-running the model here would just re-summarize its own
        # output. Only run it when the merge loop never fired.
        if merge_round > 0:
            final_summary = merged
        else:
            final_summary = self._run_model(
                merged,
                min_length=config.final_min_length,
                max_length=config.final_max_length,
                language=normalized_language,
            )
        self._emit_progress(
            progress_callback,
            {
                "phase": "final_done",
                "level": normalized_level,
                "language": normalized_language,
                "chunk_total": len(chunks),
            },
        )
        return final_summary
