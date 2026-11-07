"""Portable publication-output paths without requiring a sibling TeX project."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def paper_directory():
    configured = os.environ.get("MOTIF_PAPER_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local_paper = PROJECT_ROOT.parent / "足球分析研究" / "els-cas-templates"
    return local_paper if local_paper.is_dir() else PROJECT_ROOT / "outputs" / "paper"


def figure_directory():
    return paper_directory() / "figures"
