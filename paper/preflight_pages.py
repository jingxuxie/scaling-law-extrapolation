#!/usr/bin/env python3
"""Validate the AAAI main-paper content/reference page boundary.

The manuscript may use at most seven technical-content pages followed by at
most two reference-only pages.  The check deliberately inspects rendered PDF
text rather than trusting source line placement, since floats can otherwise
spill technical material into the reference allowance.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> str:
    try:
        completed = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"required preflight executable not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip()
        raise SystemExit(f"{' '.join(args)} failed: {message}") from exc
    return completed.stdout


def _page_text(pdf: Path, first: int, last: int) -> str:
    return _run(
        "pdftotext",
        "-f",
        str(first),
        "-l",
        str(last),
        "-layout",
        str(pdf),
        "-",
    )


def main() -> None:
    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "main.pdf")
    if not pdf.is_file():
        raise SystemExit(f"missing PDF: {pdf}")

    info = _run("pdfinfo", str(pdf))
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, flags=re.MULTILINE)
    if match is None:
        raise SystemExit("could not read page count from pdfinfo")
    pages = int(match.group(1))
    if not 8 <= pages <= 9:
        raise SystemExit(
            f"{pdf} has {pages} pages; expected seven technical pages plus one or two reference pages"
        )

    technical = _page_text(pdf, 1, 7)
    if any(line.strip() == "References" for line in technical.splitlines()):
        raise SystemExit("References begin before page 8, leaving technical-page budget unused")

    reference_page = _page_text(pdf, 8, 8)
    first_nonempty = next((line.strip() for line in reference_page.splitlines() if line.strip()), "")
    if not first_nonempty.startswith("References"):
        raise SystemExit(
            "page 8 does not begin with the References heading; a technical float may have spilled into the reference allowance"
        )

    print(f"page preflight passed: 7 technical pages + {pages - 7} reference page(s)")


if __name__ == "__main__":
    main()
