#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

latexmk -pdf -interaction=nonstopmode -halt-on-error ae.tex
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
