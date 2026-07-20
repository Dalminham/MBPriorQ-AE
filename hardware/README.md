# MBPriorQ Hardware Artifact

This directory contains the SpinalHDL implementation and deterministic
functional simulations of the MBPriorQ accelerator dataflow. The complete
BlackBox accepts 1024-bit FP4 data/metadata packets, executes the regular or
refined path selected by the weight and activation masks, joins each MultiMSA
partial matrix with its FP32 dequant factor, and emits 1024-bit output packets.

## Setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate mbpriorq-ae
```

The validated toolchain uses Java 17, sbt 1.10.2, SpinalHDL 1.8, Verilator
4.228, GNU Make, and a C++ compiler.

## Reproduce and Validate

```bash
./hardware/run_modules.sh   # five module-level simulations
./hardware/run_system.sh    # complete 16-lane 1024-bit packet path
./hardware/run_all.sh       # both workflows
```

Each command performs three checks:

1. The Scala testbench drives a deterministic fixture and asserts the RTL
   output against an expected functional contract.
2. `hardware/validate_results.py` checks the generated CSV semantically and
   prints a human-readable PASS/FAIL summary.
3. The generated CSV is compared with the checked-in result under
   [`expected/`](expected/) to confirm deterministic reproduction.

Generated files are written under `local_runs/hardware_modules/` and
`local_runs/hardware_system/`.

## Complete BlackBox

The public top is `MBPriorQ.MBPriorQ` in
`spinal/src/main/scala/MBPriorQ/MBPriorQ.scala`.

**Inputs**

- `MSA_EN`: starts the packet-processing transaction.
- `data_valid` and `data_packet[1023:0]`: provide the 1024-bit packet stream.
  A packet contains an 8-bit type, an 8-bit block/scale-entry index, and a
  1008-bit payload.
- `output_ready`: applies downstream flow control.

**Outputs**

- `output_pulse`: indicates that one output packet is valid.
- `output_packet[1023:0]`: carries the block index, sub-block index, segment
  index, 16x16 BF16 partial-matrix payload, and FP32 dequant factor.

The top-level simulation sends one HEAD packet, reordered base/extended scale
packets, 16 weight-data packets, and one activation-data packet. Weight and
activation masks are independent. Five blocks take the regular path and eleven
take the refined path. Periodic deassertion of `output_ready` injects output
backpressure.

For every expected `(block_index, sub_block_idx)` pair, the testbench runs a
reference MSA calculation, reconstructs the scale in the testbench, reassembles
the five emitted packet segments, and checks the complete matrix and scale.
This produces 49 matrix-scale pairs and 245 physical output packets.

## Module-Level Contracts

`./hardware/run_modules.sh` runs all five rows below. The module simulations
separate control-path failures from complete-pipeline failures; the system
simulation then verifies their integrated numerical and packet behavior.

| Module | BlackBox input | BlackBox output | Simulation stimulus and oracle | Result CSV |
|---|---|---|---|---|
| Scale reconstructor | Weight/activation mask bits; FP8 base and three extended scales per tensor; FP32 global scales | One regular or four refined FP32 dequant factors and a valid sub-block mask | Exercises all four weight/activation mask combinations. Exact FP32 output bits and valid sub-blocks are compared with testbench-computed factors. | `modules/scale_reconstructor.csv` |
| Packet scheduler | Metadata/data packet-arrival events, refined mask, MultiMSA/VPU readiness, completion events, output readiness | VPU and MultiMSA issue events, block-index-ordered output events, completion status | Sends scale metadata before FP4 data, requiring VPU issue before MultiMSA issue. Injects block completions in order `3,1,2,0` and checks ordered commit plus the regular/refined output-beat counts. | `modules/packet_scheduler.csv` |
| Shared FPU pool | A mask of eight regular/refined block jobs and a bounded 16-FPU pool | Per-block completion events and FPU occupancy counters | Submits all eight mixed-path jobs together. Checks that every block is accepted and completes exactly once and that occupancy never exceeds the pool size. | `modules/shared_fpu_pool.csv` |
| MultiMSA path | One 16-value FP4 weight micro-block, one 16-value FP4 activation micro-block, block index, regular/refined mode | Indexed 16x16 BF16 partial matrices and block completion | Runs one regular and one refined case. Checks block identity, nonzero partials, one regular sub-block `[0]`, four refined sub-blocks `[0,1,2,3]`, and the expected latency relation. Exact matrix values are checked again at the integrated top. | `modules/multimsa_paths.csv` |
| Output pair join | Independently arriving `(block, sub-block, matrix)` events and `(block, factor mask, factors)` events | Ordered `(block, sub-block, matrix, dequant factor)` pairs | Deliberately reorders matrix and factor arrivals and applies output backpressure. Checks that output waits for both values, preserves their exact payloads, and commits in block/sub-block order. | `modules/output_pair_join.csv` |

## Reading the Results

Module CSVs report the applied case, expected behavior, observed behavior, and
`status`. They are intended to be readable without inspecting waveform files.
Scale arrays are ordered as `[base, extended_0, extended_1, extended_2]`.

The complete-system run produces two complementary files:

- `system/system_summary.csv` is the primary reviewer-facing result. Each row
  reports one logical block, its weight/activation masks, selected path,
  sub-blocks, matrix/scale matches, packet count, output-cycle range, and status.
- `system/external_1024_top.csv` is the low-level packet trace. Each
  matrix-scale pair occupies five rows: segments 0-3 carry the first 4000 matrix
  bits, and segment 4 carries the remaining 96 matrix bits plus the FP32 scale.
  `last=true` marks the fifth segment. Gaps in `cycle` show the injected
  backpressure rather than missing packets.

## What `expected/` Means

[`expected/`](expected/) contains the approved golden CSVs for these fixed
fixtures. It is not used to calculate the expected RTL output. Functional
expectations are declared or calculated inside each testbench and asserted
before any CSV is written. The golden files provide a second, reviewer-visible
check that the complete observation is reproducible.

A successful run ends with descriptive PASS lines followed by:

```text
Hardware functional and golden-result validation passed.
```

The reviewer may also rerun the validator directly:

```bash
python hardware/validate_results.py \
  --actual local_runs/hardware_modules --scope modules

python hardware/validate_results.py \
  --actual local_runs/hardware_system --scope system
```
