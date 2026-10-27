"""Write generated evidence only when absent or byte-for-byte unchanged."""
from __future__ import annotations

from pathlib import Path


def write_unchanged_or_new(path: Path, content: str) -> None:
    path = Path(path)
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"Refusing to overwrite changed evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
