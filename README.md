# madrona_sameboy

This repository is archived as of 2026-01-20.

## Why we abandoned Madrona

These reasons are carried over from GBxCuLE Learning Lab (/home/bkase/Documents/a4-core/projects/gbxcule-learning-lab/index.md):

- GPU megakernel dominates frame time in Madrona; Nsight shows ~390 ms per frame at 512 worlds.
- GPU throughput remains far below CPU even at large env counts; scaling alone doesn't close the gap.
- PPU is tightly coupled and sequential; simple parallelization in Madrona is infeasible without a rewrite.
- Multi-block megakernel configs hang or fail to compile (reg pressure), limiting occupancy tuning.
- compute-sanitizer reports invalid writes in Madrona GPU ECS init at scale, suggesting platform risk.
