// GPU aggregation TU for SameBoy core C sources.
// Keep includes in sync with SAMEBOY_CORE_SRCS in src/CMakeLists.txt.

#include "sameboy_gpu_config.h"

#include "apu.c"
#include "camera.c"
#include "display.c"
#include "gb.c"
#include "joypad.c"
#include "mbc.c"
#include "memory.c"
#include "printer.c"
#include "random.c"
#include "rumble.c"
#if !defined(GB_GPU_MODE)
#include "save_state.c"
#endif
#include "sgb.c"
#include "sm83_cpu.c"
#include "timing.c"
#include "workboy.c"
