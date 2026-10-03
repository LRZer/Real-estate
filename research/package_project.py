"""Create a small, shareable research package without temporary environments."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / "70城房价预测-申请展示.zip"
EXCLUDED_PARTS = {".venv", "__pycache__", "official_pages", ".git", "cache", "tmp"}
INCLUDED_SUFFIXES = {".py", ".md", ".json", ".csv", ".txt", ".png", ".pdf", ".pt", ".npz", ".joblib"}


def main() -> None:
    selected = [
        path for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)
        and path.suffix.lower() in INCLUDED_SUFFIXES
    ]
    if not (ROOT / "data" / "official_nbs_70city.csv").exists():
        raise FileNotFoundError("The cleaned data must be included")
    if len(selected) < 20:
        raise RuntimeError("Too few deliverable files; inspect the project before packaging")
    required = ("最终实验报告.md", "output/pdf/70城新房指数预测_成果展示.pdf",
                "results/research/checkpoint_verification.json", "results/research/models/fair_tcn/seed_79.pt")
    for item in required:
        if not (ROOT / item).exists():
            raise FileNotFoundError(item)
    with ZipFile(ARCHIVE, "w", ZIP_DEFLATED, compresslevel=8) as bundle:
        for path in sorted(selected):
            bundle.write(path, arcname=Path(ROOT.name) / path.relative_to(ROOT))
    print(f"Created {ARCHIVE} with {len(selected)} files ({ARCHIVE.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
