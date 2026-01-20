# Learnings
- 2026-01-20: sameboy_core_gpu.cpp includes must stay in sync with SAMEBOY_CORE_SRCS in src/CMakeLists.txt for GPU aggregation.
- 2026-01-20: NVRTC compile defs must include sameboy_core_gpu.cpp in madrona_build_compile_defns SRCS so GPU builds see SameBoy C code.
- 2026-01-20: headless_gpu run hits NVRTC errors (time_t/time.h, FILE in apu.h, GB_SECTION zero-length arrays, typeof in gb.h, libcudacxx lacking std::memset/free), so bd-aww now depends on GPU-safe flags/time/audio/file-IO work.
- 2026-01-20: GPU builds now include sameboy_gpu_config.h via GB_GPU_BUILD in NVRTC flags, which defines GB_GPU_MODE and disables timekeeping/debugger/cheats/rewind/audio/file I/O for SameBoy on GPU.
- 2026-01-20: SameBoy build flags are now centralized (GB_DISABLE_TIMEKEEPING/AUDIO/FILE_IO + GB_DMG_ONLY) for CPU/GPU; GB_is_cgb/GB_is_sgb helpers return false when GB_DMG_ONLY is set.
- 2026-01-20: Added GB_init_no_alloc for external RAM/VRAM/MBC/ROM pointers; ownership flags prevent internal free/alloc and MBC RAM realloc.
- 2026-01-20: Guarded SameBoy core allocations with GB_MALLOC/GB_FREE/GB_REALLOC macros so GB_GPU_MODE builds avoid host malloc/free symbols.
- 2026-01-20: RTC/timekeeping now uses GB_HOST_TIME() (0 on GB_GPU_MODE/GB_DISABLE_TIMEKEEPING) so GPU builds avoid time()/time.h while CPU remains unchanged.
