// GPU aggregation TU for SameBoy core C sources.
// Keep includes in sync with SAMEBOY_CORE_SRCS in src/CMakeLists.txt.
// Optional modules are stubbed out for GPU builds to cut compile size.

#include "sameboy_gpu_config.h"
#include "sameboy_gpu_stubs.c"

#include "apu.c"
#include "display.c"
#include "gb.c"
#include "joypad.c"
#include "mbc.c"
#include "memory.c"
#include "random.c"
#if !defined(GB_GPU_MODE)
#include "save_state.c"
#endif
#include "sm83_cpu.c"
#include "timing.c"
