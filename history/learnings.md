# Learnings
- 2026-01-20: sameboy_core_gpu.cpp includes must stay in sync with SAMEBOY_CORE_SRCS in src/CMakeLists.txt for GPU aggregation.
- 2026-01-20: NVRTC compile defs must include sameboy_core_gpu.cpp in madrona_build_compile_defns SRCS so GPU builds see SameBoy C code.
