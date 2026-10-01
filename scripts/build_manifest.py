from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"
INCLUDED = ("data", "vectors", "results", "configs", "docs", "prompts", "evaluation", "scripts", "tests")
ROOT_FILES = (".gitignore", "README.md", "AI_USAGE.md", "requirements.txt", "requirements-inference.txt")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    paths = sorted([
        path
        for folder in INCLUDED
        for path in (ROOT / folder).rglob("*")
        if path.is_file() and '__pycache__' not in path.parts
    ] + [ROOT / name for name in ROOT_FILES])
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in paths]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.name} with {len(paths)} files")


if __name__ == "__main__":
    main()
