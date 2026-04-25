from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def add_path(zf: zipfile.ZipFile, root: Path, path: Path) -> None:
    if path.is_file():
        zf.write(path, arcname=str(path.relative_to(root)))
        return
    if path.is_dir():
        for file in path.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=str(file.relative_to(root)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final submission zip bundle.")
    parser.add_argument("--output", default="artifacts/submission_bundle.zip")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    include = [
        root / "server",
        root / "client",
        root / "scripts",
        root / "notebooks",
        root / "README.md",
        root / "requirements.txt",
        root / "Dockerfile",
        root / "run_local.ps1",
        root / "SUBMISSION_CHECKLIST.md",
        root / "artifacts" / "training_run",
    ]

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in include:
            if item.exists():
                add_path(zf, root, item)

    print(f"Created submission bundle: {output}")


if __name__ == "__main__":
    main()
