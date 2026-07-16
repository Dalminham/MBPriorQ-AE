#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

ARTIFACT_FIRST_PAGE="${ARTIFACT_FIRST_PAGE:-1}"
for _pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error -jobname=ae \
    "\\def\\ArtifactFirstPage{${ARTIFACT_FIRST_PAGE}}\\input{ae.tex}" >/dev/null
done
if grep -q 'Overfull \\hbox' ae.log; then
  echo "Artifact appendix contains an overfull line; inspect ae.log." >&2
  exit 1
fi
PAGES="$(pdfinfo ae.pdf | awk '/^Pages:/ {print $2}')"
if [[ -z "${PAGES}" || "${PAGES}" -gt 2 ]]; then
  echo "Artifact appendix must be at most two pages; observed: ${PAGES:-unknown}" >&2
  exit 1
fi
PAGE_LABEL="pages"
if [[ "${PAGES}" == "1" ]]; then
  PAGE_LABEL="page"
fi
echo "Artifact appendix compiled successfully (${PAGES} ${PAGE_LABEL})."
