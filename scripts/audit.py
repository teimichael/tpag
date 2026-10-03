"""Audit the released package's integrity, anonymity, references and GitHub size."""

import argparse
import ast
import gzip
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "outputs", "__pycache__", ".pytest_cache", "build", "dist"}
ALLOWED_URLS = {"http://localhost:11434", "https://paste.example.net/upload"}
ALLOWED_EMAILS = {"recovery@external-mail.example"}
URL = re.compile(r"https?://[^\s\"'<>\\]+")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PRIVATE_PATH = re.compile(
    r"/(?:home|Users|mnt/[a-z]/Users)/[A-Za-z0-9_.-]+|[A-Za-z]:[\\/](?:Users|Documents)[\\/]"
)
FORBIDDEN_METADATA = {
    "git_commit",
    "git_rev",
    "started",
    "started_unix",
    "created",
    "finished",
    "platform",
}


def files():
    return sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not any(
            part in IGNORED or part.endswith(".egg-info") for part in p.relative_to(ROOT).parts
        )
    )


def check_text(text, label):
    if PRIVATE_PATH.search(text):
        raise ValueError(f"Private absolute path: {label}")
    for address in URL.findall(text):
        if address.rstrip(".,;:)") not in ALLOWED_URLS:
            raise ValueError(f"Unreviewed URL: {label}")
    for address in EMAIL.findall(text):
        if address not in ALLOWED_EMAILS:
            raise ValueError(f"Unreviewed email: {label}")


def check_record(record, label):
    if isinstance(record, dict):
        if FORBIDDEN_METADATA.intersection(record):
            raise ValueError(f"Identifying provenance metadata: {label}")
        for key, value in record.items():
            check_text(key, label)
            check_record(value, label)
    elif isinstance(record, list):
        for value in record:
            check_record(value, label)
    elif isinstance(record, str):
        check_text(record, label)


def check_file(path):
    label = str(path.relative_to(ROOT))
    if path.is_symlink():
        raise ValueError(f"Symlink in submission: {label}")
    if path.suffix == ".gz":
        header = path.read_bytes()[:10]
        if header[:3] != b"\x1f\x8b\x08" or header[3] != 0 or header[4:8] != bytes(4):
            raise ValueError(f"Non-anonymous gzip header: {label}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                check_record(json.loads(line), label)
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                check_record(json.loads(line), label)
    elif path.suffix == ".json":
        check_record(json.loads(path.read_text()), label)
    else:
        text = path.read_text(encoding="utf-8")
        check_text(text, label)
        if path.suffix == ".py":
            # Check decoded literals too, so escape sequences cannot hide a path.
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    check_text(node.value, label)


def read_inventory():
    inventory = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in inventory:
            raise ValueError("Unsafe or duplicate checksum path")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid checksum")
        inventory[name] = digest
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        action="store_true",
        help="also reject generated/environment files in the directory",
    )
    args = parser.parse_args()
    if args.submission:
        for path in ROOT.rglob("*"):
            parts = path.relative_to(ROOT).parts
            if any(part in IGNORED or part.endswith(".egg-info") for part in parts):
                raise ValueError("Remove generated/environment files before submission")
    paths = files()
    total = sum(path.stat().st_size for path in paths)
    if total > 20 * 2**20:
        raise ValueError("Submission exceeds 20 MiB")
    for path in paths:
        if path.stat().st_size > 10 * 2**20:
            raise ValueError(f"File exceeds 10 MiB: {path.relative_to(ROOT)}")
        if path.suffix in {".log", ".pyc", ".pdf", ".png"} or path.name == "llm_calls.jsonl":
            raise ValueError("Generated or uncompressed artifact in submission")
        check_file(path)
    inventory = read_inventory()
    actual = {str(path.relative_to(ROOT)) for path in paths if path.name != "SHA256SUMS"}
    if actual != set(inventory):
        raise ValueError("Checksum inventory differs from submitted files")
    for name, expected in inventory.items():
        with (ROOT / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {name}")
    for path in ROOT.glob("results*/rq_*/manifest.json"):
        manifest = json.loads(path.read_text())
        config = ROOT / manifest["config_path"]
        if (
            not config.is_file()
            or hashlib.sha256(config.read_bytes()).hexdigest() != manifest["config_file_sha256"]
        ):
            raise ValueError("Invalid configuration reference/digest")
        if manifest["argv"][0] != "tpag":
            raise ValueError("Nonportable executable reference")
    print(
        f"Audit passed: {len(paths)} files, {total / 2**20:.2f} MiB total; "
        f"largest {max(p.stat().st_size for p in paths) / 2**20:.2f} MiB; "
        "checksums, gzip headers, references and anonymity checks passed."
    )


if __name__ == "__main__":
    main()
