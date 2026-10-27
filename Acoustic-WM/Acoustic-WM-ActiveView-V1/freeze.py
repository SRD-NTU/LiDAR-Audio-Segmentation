"""Hash V1 source/dependencies and verify the untouched V0 source snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_manifest(path=HERE / "SOURCE_MANIFEST.json"):
    manifest = json.loads(Path(path).read_text())
    files = manifest["files"]
    for relative, expected in files.items():
        file = HERE / relative
        if not file.is_file() or sha(file) != expected:
            raise RuntimeError(f"V1 source or dependency differs from its frozen manifest: {file}")
    actual = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if actual != manifest["codebase_sha256"]:
        raise RuntimeError("V1 source manifest aggregate identity is invalid")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "SOURCE_MANIFEST.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_source_manifest(args.output)
        print(json.dumps({"source_manifest": str(args.output),
                          "codebase_sha256": result["codebase_sha256"],
                          "verified_files": len(result["files"])}))
        return
    files = sorted([p for p in HERE.rglob("*.py") if "outputs" not in p.parts and
                    "prepared" not in p.parts and "__pycache__" not in p.parts] +
                   [HERE / "config.json", HERE / "requirements.txt", HERE / "README.md"] +
                   list((HERE / "checkpoints").glob("*.pt")) +
                   [HERE / "assets/sensor/metadata/microphone_array.json",
                    HERE / "assets/pointbind/point-audio.pt"])
    identity = {str(p.relative_to(HERE)): sha(p) for p in files}
    codebase_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    parent_manifest = json.loads((PARENT / "MANIFEST.json").read_text())
    checked = 0
    for relative, record in parent_manifest["files"].items():
        if not relative.startswith("Acoustic-WM-ActiveView/"):
            continue
        path = PARENT / relative
        if sha(path) != record["sha256"]:
            raise RuntimeError(f"V0 differs from the archived source identity: {path}")
        checked += 1
    if checked == 0:
        raise RuntimeError("Parent source manifest contains no V0 files")
    report = {"schema": "activeview-v1-source-manifest-v1",
              "codebase_sha256": codebase_sha, "files": identity,
              "v0_source_manifest": str(PARENT / "MANIFEST.json"),
              "v0_source_manifest_sha256": sha(PARENT / "MANIFEST.json"),
              "v0_files_verified_unchanged": checked,
              "frontend_runtime_identity": str(PARENT / "MANIFEST.json"),
              "note": "No Git metadata is present; listed SHA-256 values are the source identity."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"source_manifest": str(args.output),
                      "codebase_sha256": codebase_sha,
                      "v0_files_verified_unchanged": checked}))


if __name__ == "__main__":
    main()
