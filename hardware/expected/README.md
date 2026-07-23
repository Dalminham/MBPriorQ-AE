# Expected Hardware Results

This directory contains the checked-in golden CSVs for the deterministic
hardware fixtures. They are reproducibility references, not the functional
oracle: each Scala testbench first computes or declares its expected behavior
and asserts the RTL outputs before writing a CSV. `hardware/validate_results.py`
then performs semantic checks on the generated CSVs and compares them with
these files.

- `modules/` records the readable inputs, expected behavior, observed behavior,
  and status for the five module-level simulations.
- `system/system_summary.csv` is the per-block summary for the
  complete 16-lane packet path.
- `system/external_1024_top.csv` is the low-level 1024-bit output-packet trace
  retained for packet ordering and backpressure inspection.

`hardware/run_modules.sh` and `hardware/run_system.sh` regenerate and validate
the corresponding results.
