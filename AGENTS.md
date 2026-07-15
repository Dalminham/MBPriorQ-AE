# MBPriorQ AE Rules

This repository is the public MICRO 2026 artifact, not a development mirror.

- Include only files needed to evaluate claims listed in `docs/AE_SCOPE.md`.
- Keep experiment names paper-oriented. Do not use reviewer identifiers.
- Preserve source revision, file origin, modifications, and checksums.
- Do not include model weights, credentials, user-specific absolute paths,
  proprietary EDA libraries, or unlicensed third-party code.
- Do not include reproductions, source code, or result-validation workflows for
  other authors' methods.
- Do not include commercial EDA inputs, outputs, screenshots, or area/power
  data. The AE hardware scope is open MBPriorQ functional simulation only.
- A smoke test is not paper evidence. Keep smoke, core, and extended workflows
  visibly separate.
- Never mark a workflow complete until it runs cleanly from this repository and
  its expected result is recorded.
- Use `MultiMSA` consistently.
- Commit coherent progress and do not add generated model/checkpoint data.
