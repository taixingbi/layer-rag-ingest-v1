#!/usr/bin/env python3
"""Chunk Markdown for RAG: ATX headings, optional GitHub-export preamble strip, size packing."""

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
load_dotenv()

logger = logging.getLogger(__name__)

_GITHUB_TXT_PREFIX = re.compile(
    r"^source: github:[^\n]+\npath_in_archive:[^\n]+\n---\s*\n",
    re.MULTILINE,
)
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def strip_github_export_preamble(text: str) -> str:
    """Remove header written by github_tree_to_txt.py before the real Markdown."""
    t = text.lstrip("\ufeff")
    if not t.startswith("source: github:"):
        return t
    stripped, n = _GITHUB_TXT_PREFIX.subn("", t, count=1)
    if n:
        logger.debug("Stripped GitHub export preamble (%d bytes removed)", len(t) - len(stripped))
        return stripped.lstrip("\n")
    return t


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    logger.info("Reading input file: %s", path.resolve())
    raw = path.read_text(encoding="utf-8")
    text = strip_github_export_preamble(raw).strip()
    if not text:
        raise ValueError(f"Input file is empty after preamble strip and trim: {path}")
    logger.info(
        "Loaded text: chars=%d lines=%d",
        len(text),
        text.count("\n") + 1,
    )
    return text


def split_atx_sections(text: str) -> list[tuple[str, str]]:
    """Split into (section_title, body) using ATX headings (# .. ######)."""
    text = text.strip()
    matches = list(_ATX_HEADING.finditer(text))
    out: list[tuple[str, str]] = []

    if not matches:
        if text:
            out.append(("ROOT", text))
        return out

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            out.append(("ROOT", preamble))

    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            # Keep heading-only stubs (e.g. H1 before first H2) so the title is not dropped.
            body = title
            logger.debug("Heading %r had no body; using title as chunk text", title)
        out.append((title, body))

    return out


def _paragraphs_from_body(body: str) -> list[str]:
    parts = re.split(r"\n{2,}", body.strip())
    return [p.strip() for p in parts if p.strip()]


def pack_paragraphs(
    paragraphs: list[str],
    *,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """Merge consecutive small paragraphs until min_chars, cap at max_chars."""
    if not paragraphs:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        sep = 2 if buf else 0
        if buf_len + sep + len(p) <= max_chars:
            buf.append(p)
            buf_len += sep + len(p)
            if buf_len >= min_chars:
                flush()
            continue

        if buf:
            flush()
        if len(p) <= max_chars:
            buf = [p]
            buf_len = len(p)
            if buf_len >= min_chars:
                flush()
            continue

        # Oversized single block: split on newlines into sub-parts then repack
        sub = [s.strip() for s in p.splitlines() if s.strip()]
        if len(sub) <= 1:
            chunks.append(p[:max_chars])
            remainder = p[max_chars:].strip()
            if remainder:
                chunks.extend(pack_paragraphs([remainder], min_chars=min_chars, max_chars=max_chars))
            continue
        chunks.extend(pack_paragraphs(sub, min_chars=min_chars, max_chars=max_chars))

    flush()
    return chunks


def markdown_to_section_chunks(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
) -> list[tuple[str, str]]:
    """Return (section_title, chunk_text) rows in document order."""
    rows: list[tuple[str, str]] = []
    for section, body in split_atx_sections(text):
        paras = _paragraphs_from_body(body)
        packed = pack_paragraphs(paras, min_chars=min_chars, max_chars=max_chars)
        for chunk in packed:
            rows.append((section, chunk))
    return rows


def build_stage1_chunks(
    section_chunks: list[tuple[str, str]],
    chunk_id_width: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for idx, (section, paragraph) in enumerate(section_chunks, start=1):
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Chunk Markdown by ATX headings (#..######), strip github_tree_to_txt "
            "preamble, pack small paragraphs — same JSON shape as text_to_chunks.py."
        )
    )
    p.add_argument(
        "input_path",
        help=(
            "Path to a .md/.txt file, or dataset root (reads <root>/raw/*.{txt,md} "
            "and writes <root>/processed/chunks_*.json)."
        ),
    )
    p.add_argument(
        "output_path",
        nargs="?",
        default="",
        help="Output JSON path (required for single-file mode).",
    )
    p.add_argument(
        "--min-chunk-chars",
        type=int,
        default=400,
        help="Target minimum characters per chunk before starting a new one (default: 400).",
    )
    p.add_argument(
        "--max-chunk-chars",
        type=int,
        default=2800,
        help="Maximum characters per chunk (default: 2800).",
    )
    p.add_argument(
        "--chunk-id-width",
        type=int,
        default=4,
        help="Zero-padding width for chunk_id (default: 4).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    return p.parse_args()


def _process_single_file(
    *,
    input_path: Path,
    output_path: Path,
    chunk_id_width: int,
    min_chars: int,
    max_chars: int,
) -> int:
    text = read_text(input_path)
    section_chunks = markdown_to_section_chunks(
        text, min_chars=min_chars, max_chars=max_chars
    )
    if not section_chunks:
        raise ValueError("No chunkable content found after Markdown split.")

    logger.info(
        "Markdown chunking: sections=%d chunks=%d min_chars=%d max_chars=%d",
        len({s for s, _ in section_chunks}),
        len(section_chunks),
        min_chars,
        max_chars,
    )

    payload = build_stage1_chunks(
        section_chunks=section_chunks,
        chunk_id_width=chunk_id_width,
    )
    write_json(output_path, payload)
    logger.info("Done: wrote %d chunk(s) to %s", len(payload), output_path.resolve())
    return len(payload)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    configure_logging(verbose=args.verbose)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else None

    min_c, max_c = args.min_chunk_chars, args.max_chunk_chars
    if max_c < min_c:
        raise SystemExit("--max-chunk-chars must be >= --min-chunk-chars")

    logger.info(
        "Starting markdown_to_chunks: input=%s output=%s min=%d max=%d",
        input_path,
        output_path,
        min_c,
        max_c,
    )

    if input_path.is_dir():
        if output_path is not None:
            raise ValueError("output_path must be omitted in directory mode.")
        raw_dir = input_path / "raw"
        processed_dir = input_path / "processed"
        if not raw_dir.exists():
            raise FileNotFoundError(f"Directory mode expects raw folder: {raw_dir}")
        processed_dir.mkdir(parents=True, exist_ok=True)

        paths = sorted(raw_dir.glob("*.txt")) + sorted(raw_dir.glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"No .txt or .md files under {raw_dir}")

        total = 0
        logger.info("Directory mode: %d file(s) from %s", len(paths), raw_dir)
        for src in paths:
            out_file = processed_dir / f"chunks_{src.stem}.json"
            logger.info("Processing: %s -> %s", src, out_file)
            total += _process_single_file(
                input_path=src,
                output_path=out_file,
                chunk_id_width=args.chunk_id_width,
                min_chars=min_c,
                max_chars=max_c,
            )
        logger.info("Directory mode complete: total_chunks=%d", total)
    else:
        if output_path is None:
            raise ValueError("output_path is required in single-file mode.")
        _process_single_file(
            input_path=input_path,
            output_path=output_path,
            chunk_id_width=args.chunk_id_width,
            min_chars=min_c,
            max_chars=max_c,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("markdown_to_chunks total_latency_ms=%.1f", elapsed_ms)


if __name__ == "__main__":
    main()
