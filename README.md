# Local Document Summarizer (Python)

Local CLI application to summarize long documents using a free local model and export results to a Word document.

## Features

- Local summarization (no paid API required)
- Automatic language detection (English/Spanish)
- Summaries and report labels are generated in the source language
- Supports long documents through chunking + merge summarization
- Console progress logs per chunk, merge rounds, and level completion
- Three summary levels:
  - short: concise overview
  - medium: balanced detail
  - long: detailed summary
- Input formats: `.txt`, `.md`, `.docx`, `.pdf`
- Output format: `.docx`

## Requirements

- Python 3.10+
- Internet connection only for first model download

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Usage

### Summarize all levels

```bash
summarize-doc path/to/document.pdf --level all
```

### Summarize one level

```bash
summarize-doc path/to/document.docx --level short
```

### Set custom output file

```bash
summarize-doc path/to/document.txt --level medium --output output/my_summary.docx
```

### Force CPU execution

```bash
summarize-doc path/to/document.md --level long --device cpu
```

Device defaults to `auto`, which now picks CUDA, then Apple Silicon (`mps`), then CPU. If a model fails to load on the selected accelerator, it automatically falls back to CPU with a warning instead of failing silently.

### Control beam search width

```bash
summarize-doc path/to/document.md --level all --num-beams 4
```

Generation defaults to `--num-beams 2` (down from each model's built-in default of 4) for roughly 2x faster inference with a small quality tradeoff. Raise it for higher-quality (slower) summaries.

### Force document language manually

```bash
summarize-doc path/to/document.txt --level all --language es
```

## Notes

- Default model by language (auto-detected):
  - English: `sshleifer/distilbart-cnn-12-6`
  - Spanish: `csebuetnlp/mT5_multilingual_XLSum`
- You can override with `--model <model_name>` to force a custom model.
- Generated reports include source metadata, model name, generation timestamp, and the requested summary sections.
- `--level all` generates `long` first, then derives `medium` and `short` by further condensing that summary rather than re-processing the source document from scratch for each level.
