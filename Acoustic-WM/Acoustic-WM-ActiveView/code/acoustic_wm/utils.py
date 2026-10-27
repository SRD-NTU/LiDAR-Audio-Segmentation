from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def input_path(path):
    p=Path(path)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
