# Hardware Expected Results

The CSVs in this directory were regenerated after curating the SpinalHDL source
closure from MBPriorQ-Accelerator commit
`476c7bc9846e377309e93a69d66626cab63b3ddd`.

Key expected observations are:

- all four independent weight/activation mask combinations select the expected
  FP32 dequant factors;
- metadata makes the regular/refined VPU work visible at fixture cycles 6/8,
  before the first MultiMSA issue at cycle 14;
- the reduced shared-pool fixture accepts and completes eight mixed-path blocks
  while never exceeding its 16-FPU capacity;
- the reduced MultiMSA fixture emits one regular partial or four refined partials
  with sub-block indices 0--3;
- the public 16-lane top emits 49 matrix-scale pairs as 245 ordered 1024-bit
  packets under output backpressure.

Cycle values include test-interface overhead and serve only as deterministic
functional regression observables. They are not area, power, or paper-level
performance evidence.
