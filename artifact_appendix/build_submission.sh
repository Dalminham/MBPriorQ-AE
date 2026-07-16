#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 PAPER.pdf [OUTPUT.pdf]" >&2
  exit 2
fi

PAPER_PDF="$(realpath "$1")"
OUTPUT_PDF="${2:-${REPO_ROOT}/MBPriorQ_AE_Submission.pdf}"

if [[ ! -f "${PAPER_PDF}" ]]; then
  echo "Paper PDF not found: ${PAPER_PDF}" >&2
  exit 1
fi
if ! command -v pdfunite >/dev/null 2>&1; then
  echo "pdfunite is required to assemble the submission PDF." >&2
  exit 1
fi

PAPER_PAGES="$(pdfinfo "${PAPER_PDF}" | awk '/^Pages:/ {print $2}')"
ARTIFACT_FIRST_PAGE="$((PAPER_PAGES + 1))" "${ROOT}/build.sh"
APPENDIX_PAGES="$(pdfinfo "${ROOT}/ae.pdf" | awk '/^Pages:/ {print $2}')"
mkdir -p "$(dirname "${OUTPUT_PDF}")"
pdfunite "${PAPER_PDF}" "${ROOT}/ae.pdf" "${OUTPUT_PDF}"
OUTPUT_PAGES="$(pdfinfo "${OUTPUT_PDF}" | awk '/^Pages:/ {print $2}')"
EXPECTED_PAGES="$((PAPER_PAGES + APPENDIX_PAGES))"

if [[ "${OUTPUT_PAGES}" -ne "${EXPECTED_PAGES}" ]]; then
  echo "Submission PDF page mismatch: expected ${EXPECTED_PAGES}, observed ${OUTPUT_PAGES}." >&2
  exit 1
fi
if ! pdftotext "${OUTPUT_PDF}" - | grep -i "Artifact Appendix" >/dev/null; then
  echo "Submission PDF does not contain the compiled Artifact Appendix." >&2
  exit 1
fi

echo "Submission PDF: ${OUTPUT_PDF}"
echo "Paper pages: ${PAPER_PAGES}; appendix pages: ${APPENDIX_PAGES}; total: ${OUTPUT_PAGES}."
