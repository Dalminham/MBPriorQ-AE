# MBPriorQ Hardware

This directory contains the SpinalHDL source and functional simulations for the
MBPriorQ accelerator dataflow.

## Setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate mbpriorq-ae
```

The validated toolchain uses Java 17, sbt 1.10.2, SpinalHDL 1.8, Verilator
4.034 or newer, GNU Make, and a C++ compiler.

## Run

```bash
./hardware/run_modules.sh   # module-level mechanisms
./hardware/run_system.sh    # complete 1024-bit packet path
./hardware/run_all.sh       # both workflows
```

Generated traces are compared with [`expected/`](expected/).

## Functional Coverage

| Scope | Contract |
|---|---|
| Scale reconstruction | regular, weight-refined, activation-refined, and both-refined scale selection |
| Packet scheduling | metadata-first VPU issue, MultiMSA issue, and block-index completion |
| Shared FPU pool | allocation and completion of mixed regular/refined work |
| MultiMSA | one regular matrix or four refined partial matrices |
| Output joining | matrix/dequant-factor matching and ordered release under backpressure |
| Packet top | 1024-bit parsing, compute, synchronization, and output packetization |

The system fixture uses the public 16-lane top, 16 logical blocks, independent
weight/activation masks, reordered scale packets, and output backpressure. It
checks 49 matrix-scale pairs serialized into 245 output packets.
