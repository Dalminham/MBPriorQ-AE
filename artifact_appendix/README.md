# Artifact Appendix

`ae_body.tex` contains the appendix body and can be included after the paper's
main text. `ae.tex` is a standalone ACM two-column wrapper used to validate the
two-page limit locally.

Before release, replace `ArtifactVersion`, `ArtifactDOI`, and `ArtifactLicense`
in `ae.tex`, and use the same values in the paper-integrated appendix and
release metadata.

Build and check it with:

```bash
./artifact_appendix/build.sh
```

The script fails if compilation fails or the standalone appendix exceeds two
pages. The original official-template checksum is retained in
`metadata/source_manifest.json`; the body follows its required abstract,
meta-information, description, installation, workflow, expected-results,
customization, notes, and methodology structure.
