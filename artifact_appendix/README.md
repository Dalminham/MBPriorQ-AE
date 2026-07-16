# Artifact Appendix

`ae_body.tex` contains the appendix body. `ae.tex` is a standalone ACM
two-column wrapper used to validate the two-page limit locally. The HotCRP
submission is a single PDF containing the paper followed by this compiled
appendix; reviewers are not expected to compile or read a loose TeX file.

The version and license are fixed to `v1.0.0` and Apache-2.0. Before release,
replace only `ArtifactDOI` in `ae.tex`, and use the same DOI in the
paper-integrated appendix and HotCRP metadata.

Build and check it with:

```bash
./artifact_appendix/build.sh
```

The script fails if compilation fails or the standalone appendix exceeds two
pages. The original official-template checksum is retained in
`metadata/source_manifest.json`; the body follows its required abstract,
meta-information, description, installation, workflow, expected-results,
customization, notes, and methodology structure.

Build the complete submission PDF without modifying the paper TeX project:

```bash
./artifact_appendix/build_submission.sh /path/to/paper/main.pdf
```

The default output is `MBPriorQ_AE_Submission.pdf` at the repository root. The
script compiles the appendix, appends it to the supplied paper PDF, checks the
combined page count, and verifies that the appendix is present.
