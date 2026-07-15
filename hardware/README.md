# Hardware Artifact

This directory contains the curated SpinalHDL source closure required by the
public MBPriorQ accelerator top and its functional simulations. Generated
Verilog is created locally by SpinalHDL and is not distributed. Commercial EDA
material and prior-work accelerator implementations are intentionally excluded.

## Functional Test Matrix

| Scope | Simulation | Contract checked |
|---|---|---|
| Module | `Simulation.MBPriorQScaleReconstructorSim` | regular, weight-only refined, activation-only refined, and both-refined scale selection |
| Module | `Simulation.MBPriorQUpgradedPacketSchedulerSim` | metadata makes VPU work issuable before matrix data makes MultiMSA work issuable; output commits by block index |
| Module | `Simulation.MBPriorQSharedFpuPoolSim` | mixed regular/refined work shares a bounded FPU pool and every block completes once |
| Module | `Simulation.MBPriorQRefinedScheduledMultiMSASim` | one regular matrix versus four refined partial matrices with sub-block indices 0--3 |
| Module | `Simulation.MBPriorQOutputPairJoinBufferSim` | matrix/dequant-factor matching and ordered release under backpressure |
| System | `Simulation.MBPriorQExternal1024PacketTopSim` | mixed-mask 1024-bit input packets through parsing, compute, synchronization, and output packetization |

The system fixture uses the public 16-lane top, 16 logical blocks, independent
weight/activation masks, reordered scale packets, and output backpressure. It
checks 49 matrix-scale pairs serialized into 245 output packets against an
independently evaluated MSA reference.

## Requirements

- Java 17;
- sbt 1.10.2;
- Verilator 4.034 or newer;
- a C++ compiler and make.

Run module-level functional validation:

```bash
scripts/run_hardware_modules.sh
```

Run the public top from 1024-bit packet input to packet output:

```bash
scripts/run_hardware_system.sh
```

Both workflows elaborate SpinalHDL with Verilator and validate generated CSVs
against `evidence/hardware/`. `scripts/run_hardware_all.sh` runs both. The
observed module cycles are regression fixtures, not synthesis or paper-level
performance measurements.

## Troubleshooting

- If `sbt` is missing, install the pinned 1.10.2 launcher described in
  `../environment/README.md`; do not substitute an untested system Scala build.
- If Verilator is not found, activate `mbpriorq-ae-hw` and confirm
  `verilator --version` reports 4.034 or newer.
- A failed CSV comparison is a test failure, not harmless formatting drift.
  Remove the ignored `local_runs/hardware_*`, `hardware/spinal/tmp/`, and
  `hardware/spinal/target/` build products, then rerun before investigating
  source or tool-version differences.
- The system build is compilation-heavy. Ensure at least 16 GB RAM and 10 GB
  temporary disk space are available.
