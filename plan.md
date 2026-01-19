Madrona SameBoy Implementation Plan (CPU-first, GPU-ready)
=========================================================

Goal
----
Implement full SPEC.md compliance for madrona_sameboy with a single sim_tick
node per world, CPU backend working now, and layout + taskgraph ready for GPU.
Ensure cpu_instrs.gb runs correctly in the Madrona environment and we can see
its output (viewer + headless).

Decisions
---------
- Use one Game Boy entity per world (GBMachine archetype) so sim_tick runs via
  ParallelForNode and is GPU-friendly.
- Keep GB_gameboy_t as the authoritative core state, but stop copying OAM/HRAM/
  IO each tick (use the internal arrays directly). Keep WRAM/VRAM/MBC RAM as
  external SoA buffers via gb->ram / gb->vram / gb->mbc_ram pointers.
- Remove singleton-only execution (StepNode) and make sim_tick a non-static
  function so it can be compiled for GPU kernels.
- Keep serial fully disabled in emulation (no transfer callbacks), but capture
  SB/SC writes on CPU for test output only.
- Add the MWGPU entry macro in sim.cpp for GPU builds, even though we’ll run
  CPU-only now.

Plan Steps
----------
1) ECS + Taskgraph refactor
   - Define GBMachine archetype with GBState/GBRam/GBVram/GBMbcRam/
     GBFrameBuffer/GBObs/GBInput components.
   - Register those components + archetype; export GBInput and GBObs columns.
   - Replace StepNode with ParallelForNode<Engine, simTick, ...>.

2) sim_tick cleanup + GPU readiness
   - Make simTick non-static and remove singleton-only access.
   - Drop per-tick memcpy for OAM/HRAM/IO (use gb->oam/hram/io directly).
   - Keep WRAM/VRAM/MBC RAM pointers wired once during init.
   - Add MADRONA_BUILD_MWGPU_ENTRY in sim.cpp.

3) Initialization updates
   - Create the GBMachine entity in Sim constructor, store it in Sim data.
   - Initialize SameBoy with ROM padding, configure MBC, then apply post-boot
     state (boot ROM disabled).
   - Wire serial capture (write hook only) under #ifndef MADRONA_GPU_MODE.

4) Headless + viewer fixes
   - Update headless access to use GBMachine entity and gb->io_registers.
   - Improve tilemap decode (optionally skip duplicated tiles) and keep serial
     output detection.
   - Keep viewer export usage intact (now backed by archetype column).

Reference Sources Used
----------------------
- SPEC.md (repo root): required subsystems, taskgraph + sim_tick semantics,
  data layout guidance.
- SameBoy core:
  - SameBoy/Core/sm83_cpu.c (CPU loop / interrupts / timing)
  - SameBoy/Core/memory.c (memory map + IO side effects + DMA)
  - SameBoy/Core/display.c (PPU timing + render_line path)
  - SameBoy/Core/timing.c (GB_advance_cycles + timers + serial)
  - SameBoy/Core/gb.c (GB_reset_internal, GB_run_frame, post-boot defaults)
  - SameBoy/Core/gb.h (GB_gameboy_t layout + GB_IO_* constants)
  - SameBoy/Core/mbc.c (cartridge/MBC config)
  - SameBoy/Core/joypad.c (JOYP / GB_set_key_state)

- Madrona integration examples:
  - madrona_escape_room/src/types.hpp (archetype + component patterns)
  - madrona_escape_room/src/sim.cpp (ParallelForNode usage + MWGPU entry)
  - madrona_escape_room/src/sim.hpp (Sim/Engine world data pattern)

- Madrona engine headers:
  - madrona/include/madrona/taskgraph_builder.hpp (ParallelForNode constraints)
  - madrona/include/madrona/context.hpp (Context::get / singleton access)
  - madrona/include/madrona/ecs.hpp (Entity definition)
