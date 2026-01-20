# Experiments

- 2026-01-20: CPU benchmark (headless --benchmark, 120 frames). 64 worlds: 7680 frames in 0.256s = 29,993 fps total; 256 worlds: 30,720 frames in 0.819s = 37,530 fps total.
- 2026-01-20: GPU benchmark (headless_gpu --benchmark, 120 frames). 64 worlds: 7680 frames in 18.055s = 425 fps total (6.65 per-world); 256 worlds: 30,720 frames in 39.25s = 783 fps total (3.06 per-world).
- 2026-01-20: CPU benchmark sweep (headless --benchmark, 120 frames). Worlds=1: 2,878 fps total; 8: 10,407; 32: 28,710; 64: 30,946; 128: 36,885; 256: 39,095; 512: 39,584; 1024: 39,827.
- 2026-01-20: GPU benchmark sweep (headless_gpu --benchmark, 120 frames). Worlds=1: 17.56 fps total; 8: 120.22; 32: 314.56; 64: 426.99; 128: 587.32; 256: 778.99; 512: 1332.88; 1024: 1965.51. (Per-world FPS declines with scale.)
