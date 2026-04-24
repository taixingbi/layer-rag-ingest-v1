#!/usr/bin/env python3
"""Download GitHub /tree/ paths from a URL list and export text files as .txt."""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

import httpx
from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent
load_dotenv(_ROOT_DIR / ".env")
load_dotenv()

logger = logging.getLogger(__name__)

# https://github.com/owner/repo/tree/<ref>/<path>
_TREE_URL_RE = re.compile(
    r"^https?://github\.com/"
    r"(?P<owner>[^/]+)/(?P<repo>[^/]+)/tree/"
    r"(?P<ref>[^/]+)/(?P<path>.+?)/?$",
    re.IGNORECASE,
)

_TEXT_SUFFIXES = frozenset(
    {
        ".md",
        ".markdown",
        ".mdx",
        ".txt",
        ".rst",
        ".adoc",
        ".asciidoc",
        ".textile",
        ".org",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
    }
)


def configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def parse_github_tree_url(url: str) -> tuple[str, str, str, str]:
    """Return (owner, repo, ref, path_under_repo) for a /tree/ URL."""
    raw = url.strip()
    if not raw:
        raise ValueError("empty URL")
    m = _TREE_URL_RE.match(raw)
    if not m:
        raise ValueError(
            "Expected URL like "
            "https://github.com/<owner>/<repo>/tree/<branch>/<folder> — got: "
            f"{raw!r}"
        )
    owner = m.group("owner")
    repo = m.group("repo")
    ref = unquote(m.group("ref"))
    path_under = unquote(m.group("path")).strip("/")
    return owner, repo, ref, path_under


def _archive_urls(owner: str, repo: str, ref: str) -> list[str]:
    ref_enc = ref.replace("/", "%2F")
    return [
        f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{ref_enc}",
        f"https://codeload.github.com/{owner}/{repo}/zip/refs/tags/{ref_enc}",
        f"https://codeload.github.com/{owner}/{repo}/zip/{ref_enc}",
    ]


def _zip_root_dir(namelist: Iterable[str]) -> str:
    first = next((n for n in namelist if n.strip()), "")
    if not first or "/" not in first:
        raise RuntimeError("Unexpected archive layout: no root directory in zip")
    return first.split("/", 1)[0] + "/"


def _headers() -> dict[str, str]:
    token = (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def download_repo_zip(client: httpx.Client, owner: str, repo: str, ref: str) -> bytes:
    last_err: Exception | None = None
    for url in _archive_urls(owner, repo, ref):
        logger.debug("GET %s", url)
        try:
            r = client.get(url, follow_redirects=True, timeout=120.0)
            r.raise_for_status()
            return r.content
        except httpx.HTTPStatusError as e:
            last_err = e
            if e.response.status_code == 404:
                continue
            raise
    if last_err:
        raise RuntimeError(
            f"Could not download archive for {owner}/{repo} @ {ref!r} "
            f"(tried branch/tags/ref). Last error: {last_err}"
        ) from last_err
    raise RuntimeError(f"Could not download archive for {owner}/{repo} @ {ref!r}")


def _is_text_candidate(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(sfx) for sfx in _TEXT_SUFFIXES)


def _safe_output_stem(repo: str, rel_under_docs: str) -> str:
    base = f"{repo}__{rel_under_docs}"
    base = base.replace("/", "_").replace("\\", "_")
    for ch in '<>:"|?*':
        base = base.replace(ch, "_")
    return base[:200] if len(base) > 200 else base


def extract_tree_to_txt(
    *,
    zip_bytes: bytes,
    path_under_repo: str,
    out_dir: Path,
    repo: str,
) -> int:
    """Write one .txt per file under path_under_repo inside the GitHub zip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(zip_bytes)
    written = 0
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
        root = _zip_root_dir(names)
        prefix = f"{root}{path_under_repo}/" if path_under_repo else root
        logger.debug("Zip root=%r target_prefix=%r", root, prefix)

        for member in names:
            if member.endswith("/") or not member.startswith(prefix):
                continue
            rel = member[len(prefix) :]
            if not rel or rel.endswith("/"):
                continue
            if not _is_text_candidate(rel):
                logger.debug("skip (extension): %s", member)
                continue
            data = zf.read(member)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="replace")
                logger.warning("UTF-8 decode issues (replaced): %s", member)

            header = (
                f"source: github:{repo}\n"
                f"path_in_archive: {member}\n"
                f"---\n\n"
            )
            stem = _safe_output_stem(repo, rel)
            out_path = out_dir / f"{stem}.txt"
            out_path.write_text(header + text, encoding="utf-8", newline="\n")
            logger.info("Wrote %s", out_path)
            written += 1
    return written


def read_repo_list(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read GitHub /tree/ URLs from a repo list file and export each "
            "folder's text files as .txt under --out-dir."
        )
    )
    p.add_argument(
        "--repo-list",
        type=Path,
        default=_ROOT_DIR / "data2" / "raw" / "repo.txt",
        help="Path to newline-separated GitHub tree URLs (default: data2/raw/repo.txt).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT_DIR / "data2" / "raw" / "github_docs_txt",
        help="Directory for extracted .txt files (default: data2/raw/github_docs_txt).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    configure_logging(verbose=args.verbose)
    repo_list = args.repo_list
    if not repo_list.is_file():
        raise SystemExit(f"repo list not found: {repo_list}")

    urls = read_repo_list(repo_list)
    if not urls:
        raise SystemExit(f"No URLs in {repo_list}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = _headers()
    if headers:
        logger.info("Using GITHUB_TOKEN/GH_TOKEN for authenticated downloads.")
    else:
        logger.info("No GITHUB_TOKEN/GH_TOKEN set (public repos only, lower rate limits).")

    total_files = 0
    with httpx.Client(headers=headers) as client:
        for url in urls:
            owner, repo, ref, path_under = parse_github_tree_url(url)
            logger.info("Fetching %s/%s @ %s path=%r", owner, repo, ref, path_under)
            zbytes = download_repo_zip(client, owner, repo, ref)
            n = extract_tree_to_txt(
                zip_bytes=zbytes,
                path_under_repo=path_under,
                out_dir=out_dir,
                repo=repo,
            )
            if n == 0:
                logger.warning("No text files written for %s (check path/ref).", url)
            total_files += n

    logger.info("Done. URLs=%d files_written=%d out_dir=%s", len(urls), total_files, out_dir)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("github_tree_to_txt total_latency_ms=%.1f", elapsed_ms)


if __name__ == "__main__":
    main()
