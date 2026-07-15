#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/hardware_env.sh"
SBT="$(find_sbt)"
configure_java_headers

if ! command -v verilator >/dev/null 2>&1; then
  echo "ERROR: verilator was not found." >&2
  exit 1
fi

OUT="${ROOT}/local_runs/hardware_modules"
mkdir -p "${OUT}/modules" "${OUT}/simWorkspace"

export MBPRIORQ_SIM_WORKSPACE="${OUT}/simWorkspace"
export MBPRIORQ_SCALE_RECONSTRUCTOR_CSV="${OUT}/modules/scale_reconstructor.csv"
export MBPRIORQ_PACKET_SCHEDULER_CSV="${OUT}/modules/packet_scheduler.csv"
export MBPRIORQ_FPU_POOL_CSV="${OUT}/modules/shared_fpu_pool.csv"
export MBPRIORQ_MULTIMSA_CSV="${OUT}/modules/multimsa_paths.csv"
export MBPRIORQ_OUTPUT_PAIR_JOIN_CSV="${OUT}/modules/output_pair_join.csv"

cd "${ROOT}/hardware/spinal"
"${SBT}" \
  "runMain Simulation.MBPriorQScaleReconstructorSim" \
  "runMain Simulation.MBPriorQUpgradedPacketSchedulerSim" \
  "runMain Simulation.MBPriorQSharedFpuPoolSim" \
  "runMain Simulation.MBPriorQRefinedScheduledMultiMSASim" \
  "runMain Simulation.MBPriorQOutputPairJoinBufferSim"

cd "${ROOT}"
python scripts/validate_hardware_results.py --actual "${OUT}" --scope modules
