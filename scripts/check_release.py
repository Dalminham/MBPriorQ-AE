#!/usr/bin/env python3
"""Audit an MBPriorQ AE candidate for common public-release blockers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    "",
    ".c",
    ".cc",
    ".cff",
    ".cpp",
    ".csv",
    ".f",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".py",
    ".properties",
    ".scala",
    ".sbt",
    ".sha256",
    ".sh",
    ".sv",
    ".tcl",
    ".tex",
    ".toml",
    ".template",
    ".txt",
    ".v",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "checkpoints",
    "datasets",
    "local_runs",
    "models",
    "outputs",
    "simWorkspace",
    "target",
    "tmp",
}
FORBIDDEN_SUFFIXES = {".alib", ".db", ".ddc", ".safetensors", ".svf"}
FORBIDDEN_HARDWARE_SUFFIXES = {".v", ".vh", ".vhd", ".vhdl", ".sv", ".tcl"}
ALLOWED_BINARY_SUFFIXES = {".imatrix"}
FORBIDDEN_PATH_PREFIXES = {
    "evidence/dc/",
    "hardware/dc/",
    "third_party/microscopiq/",
}
FORBIDDEN_PATH_NAMES = {
    "scripts/run_dc_source_checks.sh",
    "scripts/validate_dc_evidence.py",
    "metadata/dc_evidence_manifest.json",
}
SENSITIVE_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "generic API secret": re.compile(
        r"(?i)(api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]{12,}['\"]"
    ),
    "workstation path": re.compile(r"/home/weiyijian/|/ssd/"),
}
FORBIDDEN_CONTENT_PATTERNS = {
    "comparison-method identifier": re.compile(r"\b(?:Tender|MicroScopiQ)\b", re.IGNORECASE),
    "commercial-EDA identifier": re.compile(
        r"\b(?:Design Compiler|Synopsys|SMIC)\b", re.IGNORECASE
    ),
}
CONTENT_PATTERN_ALLOWLIST = {"scripts/check_release.py"}
PLACEHOLDER_ALLOWLIST = {
    "docs/PROVENANCE.md",
    "docs/RELEASE_CHECKLIST.md",
    "scripts/check_release.py",
}
FINAL_PLACEHOLDER_PATTERN = re.compile(
    r"TO[-_]BE[-_](?:ASSIGNED|CONFIRMED)|zenodo\.TO[-_]BE[-_]ASSIGNED",
    re.IGNORECASE,
)
CHECKSUM_SKIP_NAMES = {
    "ae.aux",
    "ae.fdb_latexmk",
    "ae.fls",
    "ae.log",
    "ae.out",
    "ae.pdf",
    "release_files.sha256",
}


def iter_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in CHECKSUM_SKIP_NAMES - {"release_files.sha256"}
        and not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
    ]


def distributable_files() -> list[Path]:
    return [path for path in iter_files() if path.name not in CHECKSUM_SKIP_NAMES]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_checksums() -> list[str]:
    manifest = ROOT / "metadata" / "release_files.sha256"
    if not manifest.is_file():
        return ["missing release checksum manifest"]

    declared: dict[str, str] = {}
    findings: list[str] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            findings.append(f"invalid checksum row {line_number}")
            continue
        digest, rel = match.groups()
        if rel in declared:
            findings.append(f"duplicate checksum path: {rel}")
        declared[rel] = digest

    expected = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in distributable_files()
    }
    for rel in sorted(expected.keys() - declared.keys()):
        findings.append(f"file missing from checksum manifest: {rel}")
    for rel in sorted(declared.keys() - expected.keys()):
        findings.append(f"stale checksum path: {rel}")
    for rel in sorted(expected.keys() & declared.keys()):
        if expected[rel] != declared[rel]:
            findings.append(f"checksum mismatch: {rel}")
    return findings


def audit_zenodo_metadata() -> list[str]:
    path = ROOT / ".zenodo.json"
    if not path.is_file():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [f"invalid .zenodo.json: {exc}"]
    if not isinstance(payload, dict):
        return ["invalid .zenodo.json: root must be an object"]

    findings: list[str] = []
    if "metadata" in payload:
        findings.append("invalid .zenodo.json: metadata fields must be at the root")
    for key in ("title", "upload_type", "description", "creators", "license", "version"):
        if key not in payload:
            findings.append(f"invalid .zenodo.json: missing {key}")
    if payload.get("upload_type") != "software":
        findings.append("invalid .zenodo.json: upload_type must be software")
    if payload.get("access_right") != "open":
        findings.append("invalid .zenodo.json: access_right must be open")
    if payload.get("license") != "Apache-2.0":
        findings.append("invalid .zenodo.json: license must be Apache-2.0")
    creators = payload.get("creators")
    if not isinstance(creators, list) or len(creators) != 8:
        findings.append("invalid .zenodo.json: expected the eight paper authors")
    return findings


def audit(strict: bool) -> list[str]:
    findings: list[str] = []
    files = iter_files()

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel in FORBIDDEN_PATH_NAMES or any(
            rel.startswith(prefix) for prefix in FORBIDDEN_PATH_PREFIXES
        ):
            findings.append(f"out-of-scope comparison/commercial artifact: {rel}")
        if "tender" in rel.lower() or "microscopiq" in rel.lower():
            findings.append(f"comparison-method reproduction path: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden distributable file: {rel}")
        if rel.startswith("hardware/") and path.suffix.lower() in FORBIDDEN_HARDWARE_SUFFIXES:
            findings.append(f"generated/EDA hardware file: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES | ALLOWED_BINARY_SUFFIXES:
            findings.append(f"unreviewed binary/file type: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            if label == "workstation path" and rel in PLACEHOLDER_ALLOWLIST:
                continue
            if pattern.search(content):
                findings.append(f"{label}: {rel}")
        if rel not in CONTENT_PATTERN_ALLOWLIST:
            for label, pattern in FORBIDDEN_CONTENT_PATTERNS.items():
                if pattern.search(content):
                    findings.append(f"{label}: {rel}")

    required = [
        "README.md",
        "artifact_appendix/ae.tex",
        "artifact_appendix/ae_body.tex",
        "artifact_appendix/build.sh",
        "docs/AE_SCOPE.md",
        "docs/PROVENANCE.md",
        "docs/RELEASE_CHECKLIST.md",
        "environment/software.yml",
        "environment/hardware.yml",
        "experiments/core_accuracy/run.sh",
        "experiments/core_accuracy/expected.csv",
        "experiments/activation_attribution/run.sh",
        "experiments/activation_attribution/expected.csv",
        "experiments/granularity_ablation/run.sh",
        "experiments/granularity_ablation/expected.csv",
        "experiments/model_family_ppl/run_paper16.sh",
        "experiments/model_family_ppl/models.json",
        "experiments/offload_equivalence/run.sh",
        "experiments/downstream_benchmarks/run.sh",
        "experiments/downstream_benchmarks/expected.csv",
        "evidence/ppl/local_validation.json",
        "scripts/run_ppl_suite.py",
        "scripts/check_offload_structure.py",
        "scripts/run_downstream_benchmark.py",
        "scripts/validate_downstream_results.py",
        "software/mbpriorq_ae/offload.py",
        "scripts/run_smoke.sh",
        "scripts/run_hardware_modules.sh",
        "scripts/run_hardware_system.sh",
        "scripts/run_hardware_all.sh",
    ]
    for rel in required:
        if not (ROOT / rel).is_file():
            findings.append(f"missing required file: {rel}")

    if strict:
        strict_required = [
            "LICENSE",
            "CITATION.cff",
            ".zenodo.json",
            "metadata/source_manifest.json",
            "metadata/release_files.sha256",
            "scripts/run_smoke.sh",
        ]
        for rel in strict_required:
            if not (ROOT / rel).is_file():
                findings.append(f"missing release file: {rel}")

        for rel in ("README.md", "artifact_appendix/ae.tex", "CITATION.cff", ".zenodo.json"):
            path = ROOT / rel
            if path.is_file() and FINAL_PLACEHOLDER_PATTERN.search(
                path.read_text(encoding="utf-8")
            ):
                findings.append(f"unresolved release placeholder: {rel}")

        findings.extend(audit_checksums())
        findings.extend(audit_zenodo_metadata())

    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="require all release files")
    args = parser.parse_args()

    findings = audit(args.strict)
    if findings:
        print("Release audit failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("Release audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
