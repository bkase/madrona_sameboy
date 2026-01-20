// GPU aggregation TU for SameBoy core C sources.
// Keep includes in sync with SAMEBOY_CORE_SRCS in src/CMakeLists.txt.

extern "C" {
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
#include "save_state.c"
#include "sgb.c"
#include "sm83_cpu.c"
#include "timing.c"
#include "workboy.c"
}
