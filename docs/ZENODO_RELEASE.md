# Zenodo Release Runbook

This is the authoritative release order for artifact version `1.0.0`. Do not
publish a Zenodo record or create the final Git tag while a DOI placeholder
remains.

## 1. Create The Public Source Repository

Create an empty public repository for this curated artifact, push `main`, and
record its URL in `CITATION.cff` as `repository-code`. Do not expose the source
EasyLLM or accelerator working trees as the AE artifact, and do not add files
outside this curated repository.

## 2. Create A Zenodo Draft And Reserve Its DOI

Create a new Zenodo upload with resource type **Software**, public file
visibility, version `1.0.0`, and license `Apache-2.0`. Use `.zenodo.json` as the
metadata source. In the DOI field, select that the upload does not already have
a DOI and use **Get a DOI now!**. Save the draft, copy the reserved version DOI,
and do not delete the draft.

Do not publish yet. Zenodo documents that a reserved DOI may be inserted into
the files before upload and that publishing preserves the reserved identifier.

## 3. Backfill The Reserved DOI

Replace the two DOI placeholders in:

- `artifact_appendix/ae.tex`;
- `metadata/hotcrp_fields.md`.

Also add the bare DOI to the root of `CITATION.cff`, for example:

```yaml
doi: "10.5281/zenodo.12345678"
```

Rebuild the appendix and regenerate the combined paper-plus-appendix submission
PDF. The DOI URL entered in HotCRP must be the same version DOI.

## 4. Validate And Tag The Immutable Source

From a clean working tree, run:

```bash
./scripts/run_smoke.sh
./scripts/run_hardware_all.sh
./artifact_appendix/build.sh
python scripts/generate_release_checksums.py
python scripts/check_release.py --strict
sha256sum -c metadata/release_files.sha256
```

Commit the DOI and regenerated checksum manifest. Create the annotated tag
`v1.0.0` only after every command passes. Generate the upload from that tag:

```bash
git archive --format=tar.gz --prefix=MBPriorQ-AE-v1.0.0/ \
  --output=MBPriorQ-AE-v1.0.0.tar.gz v1.0.0
sha256sum MBPriorQ-AE-v1.0.0.tar.gz
```

Extract this archive in a fresh directory and rerun the strict audit, checksum
verification, software smoke, hardware workflows, and appendix build. Record
the archive SHA-256 in `docs/RELEASE_CHECKLIST.md`.

## 5. Upload, Preview, And Publish

Upload exactly `MBPriorQ-AE-v1.0.0.tar.gz` to the existing Zenodo draft. Check
the creator order, title, version, Apache-2.0 license, public visibility, and
reserved DOI in Zenodo Preview. Publish only after the uploaded archive hash
matches the recorded release hash.

After publication, verify that the DOI URL resolves publicly, then upload the
combined paper-plus-appendix PDF and paste the prepared fields from
`metadata/hotcrp_fields.md` into MICRO HotCRP. Mark the submission ready for
review before the deadline.
