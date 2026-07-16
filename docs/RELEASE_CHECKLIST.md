# Zenodo Release Checklist

Follow [`ZENODO_RELEASE.md`](ZENODO_RELEASE.md) for the required DOI reservation,
tagging, archive, and publication order.

## Blocking Metadata

- [x] Authors approve the Apache-2.0 public source license.
- [x] Author names, affiliations, and corresponding-author contact are recorded.
- [x] Final paper title and abstract are transcribed into metadata drafts.
- [x] Confirmed author metadata is promoted to final `.zenodo.json` and
      `CITATION.cff` files.
- [ ] The HotCRP abstract is replaced with the accepted-paper abstract; the removed
      unsupported end-to-end speedup claim must not remain.

## Source And Licensing

- [x] Curated software source closure and lightweight PPL integration are
      imported, provenance-recorded, and source-equivalence tested.
- [x] Curated hardware source closure is imported and provenance-recorded.
- [x] The only retained third-party file is the Qwen model license needed for
      the external model dependency.
- [x] No unrelated baseline reproduction is bundled.
- [x] No model weights, datasets restricted from redistribution, PDKs, `.db`
      libraries, credentials, or workstation-specific paths are present.
- [x] No commercial EDA inputs, outputs, screenshots, or area/power data are
      bundled.

## Executable Workflows

- [x] `scripts/run_smoke.sh` passes in the isolated validation environment.
- [x] Lightweight Qwen3-0.6B smoke workflow passes all 146 windows.
- [x] Existing Qwen3-0.6B activation-attribution and refined-granularity rows pass.
- [ ] Final clean-archive quick pass covers the expanded Llama2-7B drivers.
- [x] Full-GPU/streamed equivalence and real Qwen3-VL language-side BF16 streaming pass.
- [x] Qwen3-0.6B BF16/MBPriorQ downstream generation smokes pass on all three datasets.
- [x] Metadata EBW accounting and deterministic tensor fixtures pass.
- [x] Hardware module and public packet-top functional simulations pass.
- [x] Every result claimed in the final AE scope has an executable MBPriorQ
      reproduction or bounded functional-validation path.

## Documentation

- [x] Root README gives setup, runtime, disk, hardware, and expected outputs.
- [x] Every public experiment has one command, expected runtime, expected result, and
      a troubleshooting section.
- [x] Installation smoke, lightweight model reproduction, complete long
      workflows, and hardware functional validation are not conflated.
- [x] The artifact appendix compiles in the paper's two-column format within
      the two-page limit and without overfull lines.
- [ ] Final HotCRP key results and dependency fields match `docs/AE_SCOPE.md`.

## Release

- [ ] `python scripts/check_release.py --strict` passes.
- [x] Candidate source closure is validated from a `git archive` extraction
      without `.git` metadata or access to untracked repository files.
- [ ] The same clean-archive validation is rerun from the final release tag
      after the DOI is inserted.
- [ ] Release archive checksum is recorded.
- [ ] Git tag and Zenodo version agree.
- [ ] DOI resolves publicly before the final AE submission update.
