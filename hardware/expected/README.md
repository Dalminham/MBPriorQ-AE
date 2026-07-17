# Expected Hardware Traces

The checked CSVs provide deterministic references for the functional hardware
workflows:

- all four weight/activation mask combinations select the expected FP32
  dequant factors;
- VPU work becomes issuable when scale metadata is ready, independently of
  MultiMSA data readiness;
- mixed regular/refined blocks share the bounded FPU pool;
- MultiMSA emits one regular partial or four indexed refined partials;
- the 16-lane packet top emits 49 matrix-scale pairs as 245 ordered packets
  under output backpressure.

`hardware/run_modules.sh` and `hardware/run_system.sh` regenerate and compare
these traces.
