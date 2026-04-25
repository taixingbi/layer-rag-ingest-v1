#!/usr/bin/env python3
"""Convert plain text into Stage 1 raw-input JSON chunks (prose / resume-style sources)."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent
load_dotenv(_ROOT_DIR / ".env")
load_dotenv()  # optional override from current working directory

logger = logging.getLogger(__name__)


def configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a text file into Stage 1 chunks "
            "(split by section headings and paragraphs). "
            "synthetic_questions are always empty; use synthetic_questions.py on points after prepare_payloads."
        )
    )
    parser.add_argument(
        "input_path",
        help=(
            "Path to source text file, or dataset root directory (for example: data1). "
            "Directory mode reads <root>/raw/*.txt and writes to <root>/processed/chunks_*.json."
        ),
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default="",
        help="Path to output JSON file (required for single-file mode).",
    )
    parser.add_argument(
        "--chunk-id-width",
        type=int,
        default=4,
        help="Zero-padding width for chunk_id (default: 4 -> 0001, 0002...).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    logger.info("Reading input file: %s", path.resolve())
    text = path.read_text(encoding="utf-8")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"Input file is empty after trimming: {path}")
    logger.info(
        "Loaded text: raw_bytes=%d chars_after_strip=%d lines=%d",
        len(text.encode("utf-8")),
        len(cleaned),
        cleaned.count("\n") + 1,
    )
    return cleaned


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if s.startswith(("-", "*", "•")):
        return False
    if ":" in s:
        return False
    alpha_only = re.sub(r"[^A-Za-z]", "", s)
    if not alpha_only:
        return False
    return s == s.upper()


def _starts_new_paragraph(line: str, prev_line: str) -> bool:
    if not prev_line:
        return False
    if line.startswith(("•", "-", "*")):
        return True
    if "—" in line and not line.endswith("."):
        return True
    if "•" in line and re.search(r"\b(Present|19\d{2}|20\d{2})\b", line):
        return True
    if prev_line.endswith(".") and line[:1].isupper():
        return True
    return False


def _split_dense_block(block: str) -> list[str]:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return []
    if len(lines) == 1:
        return [lines[0]]

    paragraphs: list[str] = []
    buf: list[str] = [lines[0]]
    for line in lines[1:]:
        if _starts_new_paragraph(line, buf[-1]):
            paragraphs.append("\n".join(buf).strip())
            buf = [line]
        else:
            buf.append(line)
    if buf:
        paragraphs.append("\n".join(buf).strip())
    return [p for p in paragraphs if p]


def split_sections_and_paragraphs(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    current_section = "ROOT"
    current_lines: list[str] = []
    chunks: list[tuple[str, str]] = []

    def flush_section(section: str, section_lines: list[str]) -> None:
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            return
        blocks = re.split(r"\n\s*\n+", section_text)
        for block in blocks:
            b = block.strip()
            if not b:
                continue
            for paragraph in _split_dense_block(b):
                chunks.append((section, paragraph))

    for raw_line in lines:
        line = raw_line.rstrip()
        if is_heading(line):
            flush_section(current_section, current_lines)
            current_section = line.strip()
            current_lines = []
            continue
        current_lines.append(line)

    flush_section(current_section, current_lines)
    sections = {s for s, _ in chunks}
    logger.info(
        "Split into %d paragraph chunk(s) across %d section label(s): %s",
        len(chunks),
        len(sections),
        ", ".join(sorted(sections)[:12]) + ("…" if len(sections) > 12 else ""),
    )
    logger.debug("Chunk preview (section, text_len): %s", [(s, len(t)) for s, t in chunks[:8]])
    return chunks


def build_stage1_chunks(
    section_paragraphs: list[tuple[str, str]],
    chunk_id_width: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for idx, (section, paragraph) in enumerate(section_paragraphs, start=1):
        chunk_id = str(idx).zfill(chunk_id_width)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "section": section,
                "text": paragraph,
                "synthetic_questions": [],
            }
        )
    return chunks


def write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    logger.info("Writing %d chunk(s) to %s", len(payload), path.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _process_single_file(
    *,
    input_path: Path,
    output_path: Path,
    chunk_id_width: int,
) -> int:
    text = read_text(input_path)
    section_paragraphs = split_sections_and_paragraphs(text)
    if not section_paragraphs:
        raise ValueError("No chunkable content found in input text.")

    logger.info("Building %d Stage 1 chunk record(s)", len(section_paragraphs))

    payload = build_stage1_chunks(
        section_paragraphs=section_paragraphs,
        chunk_id_width=chunk_id_width,
    )
    write_json(output_path, payload)
    logger.info("Done: wrote %d Stage 1 chunk(s) to %s", len(payload), output_path.resolve())
    return len(payload)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    configure_logging(verbose=args.verbose)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else None

    logger.info(
        "Starting plain_text_chunks: input=%s output=%s chunk_id_width=%d",
        input_path,
        output_path,
        args.chunk_id_width,
    )

    if input_path.is_dir():
        if output_path is not None:
            raise ValueError("output_path must be omitted in directory mode.")
        raw_dir = input_path / "raw"
        processed_dir = input_path / "processed"
        if not raw_dir.exists():
            raise FileNotFoundError(f"Directory mode expects raw folder: {raw_dir}")
        processed_dir.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(raw_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found under {raw_dir}")

        total_chunks = 0
        logger.info("Directory mode: processing %d text file(s) from %s", len(txt_files), raw_dir)
        for txt_file in txt_files:
            out_file = processed_dir / f"chunks_{txt_file.stem}.json"
            logger.info("Processing source: %s -> %s", txt_file, out_file)
            total_chunks += _process_single_file(
                input_path=txt_file,
                output_path=out_file,
                chunk_id_width=args.chunk_id_width,
            )
        logger.info(
            "Directory mode complete: files=%d total_chunks=%d output_dir=%s",
            len(txt_files),
            total_chunks,
            processed_dir.resolve(),
        )
    else:
        if output_path is None:
            raise ValueError("output_path is required in single-file mode.")
        _process_single_file(
            input_path=input_path,
            output_path=output_path,
            chunk_id_width=args.chunk_id_width,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("plain_text_chunks total_latency_ms=%.1f", elapsed_ms)


if __name__ == "__main__":
    main()
