# Determinism and Divergence Thresholds

The CPU/GPU comparison harness (`compare_cpu_gpu`) is the source of truth for
cross-device determinism. The current goal is exact agreement for DMG mode
(simulated on CPU and GPU).

## Running the check

```bash
./build/compare_cpu_gpu <rom> <frames> [worlds] [gpu_id] --report <path>
```

Example:

```bash
./build/compare_cpu_gpu cpu_instrs.gb 120 --report build/diff_report.json
```

The tool prints hash and mismatch summaries and (optionally) emits a JSON report
suitable for CI gating.

## Divergence thresholds (current)

All thresholds are strict:

- WRAM/VRAM/MBC RAM: **0** mismatched bytes allowed.
- Packed observations: **0** mismatched bytes allowed.
- Registers (PC/SP/LY/STAT): must match exactly.

Any mismatch is considered a failure and should be investigated.

## Interpreting `diff_report.json`

The report includes per-world buffer hashes, mismatch counts, and the first
mismatch location/value. When `mismatches == 0`, the `first_index`, `cpu_value`,
`gpu_value` fields are set to 0.

Use this report to:

- Identify which buffer diverged first.
- Track regressions between revisions.
- Pinpoint acceptable exceptions (if/when they are agreed on).

## Known tolerable drift

None documented yet (as of 2026-01-20). If we decide to allow a specific
exception, list it here with scope, rationale, and tracking issue.
