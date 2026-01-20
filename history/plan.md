Madrona SameBoy CUDA Plan (DMG-only, scanline-accurate)
=======================================================

Goals
-----
- DMG-only emulator running on Madrona CPU and GPU.
- Scanline-accurate PPU (no dot/T-cycle accuracy).
- GPU-safe SameBoy fork (no file I/O, no host time, no device malloc/free).
- CPU/GPU compatibility or determinism where feasible in scanline mode.
- Observations: 2-bit grayscale, downsampled 2x (160x144 -> 80x72).
- Single ROM shared across all worlds (Pokemon Red later; cpu_instrs.gb now).

Non-goals (for now)
-------------------
- CGB/SGB, RTC, audio, printer, camera, workboy, cheats, rewind, debugger.
- CuLE-style API surface or Python bindings (PyBoy wrapper comes later).
- Cycle/dot-accurate rendering.

Architecture Overview
---------------------
1) Core emulator (SameBoy GPU fork)
   - DMG-only build with GPU-safe compile flags.
   - New compile-time config header (e.g., sameboy_gpu_config.h) controlling:
     - GB_GPU_MODE
     - GB_DISABLE_TIMEKEEPING (RTC off)
     - GB_DISABLE_DEBUGGER, GB_DISABLE_CHEATS, GB_DISABLE_REWIND
     - GB_DISABLE_AUDIO (new flag to stub APU)
     - GB_DISABLE_FILE_IO (new flag to stub save/load)
   - Remove or compile-out modules not needed.

2) Fast scanline PPU
   - Add scanline PPU path in display.c:
     - GB_display_run_scanline(gb, cycles)
     - Advances line on LINE_LENGTH and renders line in one shot.
     - Minimal STAT/LY/VBlank update logic preserved.
   - Gate by gb->fast_ppu or gb->ppu_mode.
   - In GB_advance_cycles, call scanline PPU when enabled.

3) Madrona integration
   - One GBMachine per world, one simTick per world.
   - GBState contains GB_gameboy_t.
   - WRAM/VRAM/MBC RAM are SoA components bound to gb->ram/gb->vram/gb->mbc_ram.
   - ROM pointer in config points to device memory for GPU builds.

4) Observation path
   - Generate 2-bit grayscale from framebuffer (0..3).
   - Downsample 2x to 80x72 (2x2 box filter or similar).
   - Pack 4 pixels per byte into GBObsPacked (row-major).

Key Data Layout
---------------
- GBFrameBuffer: 160x144 uint32_t or uint8_t grayscale.
- GBObsPacked: (80x72)/4 = 1440 bytes per world.
- Layout: row-major, 4 pixels/byte, lowest 2 bits = leftmost pixel.

Build & GPU Compilation Plan
----------------------------
Problem today:
- madrona_build_compile_defns only collects .cpp sources; SameBoy core is .c.

Fix options:
- Preferred: add a GPU aggregation TU (sameboy_core_gpu.cpp) that includes the
  required SameBoy .c files under extern "C".
- Alternative: patch madrona_build_compile_defns to include .c sources.
- Keep CPU build using SameBoy .c directly.

SameBoy Fork Changes
--------------------
1) No dynamic allocation on device
   - Add GB_init_no_alloc API:
     - Takes pointers for ram, vram, mbc_ram, rom, rom_size.
     - Does not malloc/free internally.
   - Guard all malloc/free in gb.c/mbc.c with GB_GPU_MODE.

2) Remove time & RTC
   - Compile timing.c with GB_DISABLE_TIMEKEEPING on GPU.
   - Stub rtc_run or early return (RTC ignored).
   - Ensure no time() or gettimeofday() in GPU build.

3) Remove audio
   - Stub GB_apu_run and audio output logic under GB_DISABLE_AUDIO.

4) Scanline PPU
   - Implement GB_display_run_scanline in display.c.
   - Hook into GB_advance_cycles when fast mode is enabled.

Madrona Sim Changes
-------------------
- src/sim.cpp:
  - simTick calls GB_run_frame (now scanline-driven).
  - Convert frame buffer -> downsampled packed obs.
  - Use deterministic grayscale mapping (no float).
- src/headless_gpu.cpp:
  - cudaMalloc ROM, cudaMemcpy to device.
  - Pass device pointer in Sim::Config.
  - Enable scanline mode via config.

Determinism Plan
----------------
Deterministic mode:
- Fixed seed for any random logic.
- RTC disabled.
- Scanline PPU only.
- Avoid floating-point paths on GPU.

Validation harness:
- Run CPU and GPU for N frames and compare:
  - Hash of RAM/VRAM/MBC RAM
  - Hash of packed obs
  - Key registers (PC/SP/LY/STAT)

Performance Plan
----------------
- Default to scanline mode.
- Use packed obs (2-bit) + 2x downsample.
- Batch frames per step to reduce launch overhead.
- Keep action/obs buffers on GPU.
- Tune block size and world count for Blackwell occupancy.

Milestones & Tasks
------------------
Milestone 0: Build system fixes (1-2 days)
- Add GPU aggregation TU for SameBoy C sources.
- Ensure NVRTC compilation sees SameBoy code.
- Confirm headless_gpu builds and runs.

Milestone 1: GPU-safe SameBoy fork (3-5 days)
- Add GPU config header + compile flags.
- Add GB_init_no_alloc.
- Remove/guard file I/O, malloc/free, time/RTC, debugger, audio.

Milestone 2: Scanline PPU (5-8 days)
- Implement GB_display_run_scanline.
- Hook into GB_advance_cycles.
- Validate cpu_instrs.gb in CPU and GPU builds.

Milestone 3: Observation pipeline (2-4 days)
- Downsample 2x and pack 2-bit obs.
- Export packed obs from Madrona.
- Add consistency tests.

Milestone 4: Determinism checks (2-4 days)
- CPU/GPU hash comparison tool.
- Track divergences and decide tolerable drift.

Milestone 5: Performance tuning (ongoing)
- Increase env counts.
- Adjust frames per step.
- Optimize memory layout and packing.

Open Implementation Choices (defer)
-----------------------------------
- Exact downsample method: average vs max vs center pixel.
- GPU-only grayscale generation vs GB_set_rgb_encode_callback.
- MBC RAM size default (Pokemon Red likely MBC3; ignore RTC).
