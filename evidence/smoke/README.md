# Smoke Expected Result

`expected.json` records deterministic CPU metrics for the paper MBPriorQ
calibration/prior path on synthetic tensors. The smoke runner
uses numerical tolerance and is intended to detect broken imports, altered FP4
semantics, VMB selection regressions, and EBW-accounting regressions.
