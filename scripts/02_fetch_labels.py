#!/usr/bin/env python
"""Obtain outcome labels. Labels (ToxCast/Tox21 binary hit-calls, endpoint annotations, SI tables with
the paper's LDH / MT / cell-count hits and points of departure) ship with the authors' analysis repo
(BSD-3-Clause): https://github.com/jessica-ewald/2024_09_09_Axiom_OASIS . This script shallow-clones it into
data/external/ and writes tidy tables to data/processed/{labels,endpoints}.parquet.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import EXTERNAL  # noqa: E402
from cpsa.data.build_dataset import save_labels  # noqa: E402

REPO_URL = "https://github.com/jessica-ewald/2024_09_09_Axiom_OASIS.git"
REPO_DIR = EXTERNAL / "ewald_repo"


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
    save_labels()
    print("wrote data/processed/labels.parquet and endpoints.parquet")


if __name__ == "__main__":
    main()
