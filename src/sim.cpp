#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"

#include <cstdlib>
#include <cstring>

using namespace madrona;

namespace madSameBoy {

namespace {

constexpr uint8_t btnRight = 1 << 0;
constexpr uint8_t btnLeft = 1 << 1;
constexpr uint8_t btnUp = 1 << 2;
constexpr uint8_t btnDown = 1 << 3;
constexpr uint8_t btnA = 1 << 4;
constexpr uint8_t btnB = 1 << 5;
constexpr uint8_t btnSelect = 1 << 6;
constexpr uint8_t btnStart = 1 << 7;

static uint32_t rgbEncode(GB_gameboy_t *, uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t gray = (uint8_t)((uint16_t(r) * 30 + uint16_t(g) * 59 + uint16_t(b) * 11) / 100);
    return 0xFF000000u | (uint32_t(gray) << 16) | (uint32_t(gray) << 8) | uint32_t(gray);
}

static void initPostBootState(GB_gameboy_t *gb)
{
    GB_reset(gb);

    gb->af = 0x01B0;
    gb->bc = 0x0013;
    gb->de = 0x00D8;
    gb->hl = 0x014D;
    gb->sp = 0xFFFE;
    gb->pc = 0x0100;

    GB_write_memory(gb, 0xFF05, 0x00);
    GB_write_memory(gb, 0xFF06, 0x00);
    GB_write_memory(gb, 0xFF07, 0x00);
    GB_write_memory(gb, 0xFF10, 0x80);
    GB_write_memory(gb, 0xFF11, 0xBF);
    GB_write_memory(gb, 0xFF12, 0xF3);
    GB_write_memory(gb, 0xFF14, 0xBF);
    GB_write_memory(gb, 0xFF16, 0x3F);
    GB_write_memory(gb, 0xFF17, 0x00);
    GB_write_memory(gb, 0xFF19, 0xBF);
    GB_write_memory(gb, 0xFF1A, 0x7F);
    GB_write_memory(gb, 0xFF1B, 0xFF);
    GB_write_memory(gb, 0xFF1C, 0x9F);
    GB_write_memory(gb, 0xFF1E, 0xBF);
    GB_write_memory(gb, 0xFF20, 0xFF);
    GB_write_memory(gb, 0xFF21, 0x00);
    GB_write_memory(gb, 0xFF22, 0x00);
    GB_write_memory(gb, 0xFF23, 0xBF);
    GB_write_memory(gb, 0xFF24, 0x77);
    GB_write_memory(gb, 0xFF25, 0xF3);
    GB_write_memory(gb, 0xFF26, 0xF1);
    GB_write_memory(gb, 0xFF40, 0x91);
    GB_write_memory(gb, 0xFF42, 0x00);
    GB_write_memory(gb, 0xFF43, 0x00);
    GB_write_memory(gb, 0xFF45, 0x00);
    GB_write_memory(gb, 0xFF47, 0xFC);
    GB_write_memory(gb, 0xFF48, 0xFF);
    GB_write_memory(gb, 0xFF49, 0xFF);
    GB_write_memory(gb, 0xFF4A, 0x00);
    GB_write_memory(gb, 0xFF4B, 0x00);
    GB_write_memory(gb, 0xFFFF, 0x00);

    GB_write_memory(gb, 0xFF50, 0x01);
}

static void applyInput(GB_gameboy_t *gb, uint8_t buttons)
{
    GB_set_key_state(gb, GB_KEY_RIGHT, (buttons & btnRight) != 0);
    GB_set_key_state(gb, GB_KEY_LEFT, (buttons & btnLeft) != 0);
    GB_set_key_state(gb, GB_KEY_UP, (buttons & btnUp) != 0);
    GB_set_key_state(gb, GB_KEY_DOWN, (buttons & btnDown) != 0);
    GB_set_key_state(gb, GB_KEY_A, (buttons & btnA) != 0);
    GB_set_key_state(gb, GB_KEY_B, (buttons & btnB) != 0);
    GB_set_key_state(gb, GB_KEY_SELECT, (buttons & btnSelect) != 0);
    GB_set_key_state(gb, GB_KEY_START, (buttons & btnStart) != 0);
}

#ifndef MADRONA_GPU_MODE
static bool serialWriteHook(GB_gameboy_t *gb, uint16_t addr, uint8_t value)
{
    auto *serial = static_cast<GBSerial *>(GB_get_user_data(gb));
    if (!serial) {
        return true;
    }

    if (addr == 0xFF01) {
        serial->sbWrites++;
        serial->lastSB = value;
    } else if (addr == 0xFF02) {
        serial->scWrites++;
        if ((value & 0x80) && !serial->overflowed) {
            uint8_t ch = serial->lastSB;
            if (serial->length + 1 < sizeof(serial->text)) {
                serial->text[serial->length++] = static_cast<char>(ch);
                serial->text[serial->length] = '\0';
            } else {
                serial->overflowed = 1;
            }
        }
    }
    return true;
}
#endif

} // namespace

void simTick(Engine &, GBState &state, GBRam &wram, GBVram &vram,
             GBMbcRam &mbc, GBFrameBuffer &frame, GBObs &obs, GBInput &input)
{
    GB_gameboy_t *gb = &state.gb;

    gb->ram = wram.data;
    gb->vram = vram.data;
    gb->mbc_ram = mbc.data;
    gb->screen = frame.pixels;

    applyInput(gb, input.buttons);
    GB_run_frame(gb);

    for (uint32_t i = 0; i < consts::screenPixels; i++) {
        uint8_t gray = (uint8_t)(frame.pixels[i] & 0xFFu);
        obs.pixels[i] = (uint8_t)(3 - (gray >> 6));
    }
}

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    registry.registerComponent<GBState>();
    registry.registerComponent<GBRam>();
    registry.registerComponent<GBVram>();
    registry.registerComponent<GBMbcRam>();
    registry.registerComponent<GBFrameBuffer>();
    registry.registerComponent<GBObs>();
    registry.registerComponent<GBInput>();
#ifndef MADRONA_GPU_MODE
    registry.registerComponent<GBSerial>();
#endif

    registry.registerArchetype<GBMachine>();

    registry.exportColumn<GBMachine, GBInput>((uint32_t)ExportID::Input);
    registry.exportColumn<GBMachine, GBObs>((uint32_t)ExportID::Observation);
}

void Sim::setupTasks(TaskGraphManager &mgr, const Config &)
{
    TaskGraphBuilder &builder = mgr.init(TaskGraphID::Step);
    builder.addToGraph<ParallelForNode<Engine,
        simTick,
            GBState,
            GBRam,
            GBVram,
            GBMbcRam,
            GBFrameBuffer,
            GBObs,
            GBInput
        >>({});
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &)
    : WorldBase(ctx)
{
    machine = ctx.makeEntity<GBMachine>();

    auto &state = ctx.get<GBState>(machine);
    auto &wram = ctx.get<GBRam>(machine);
    auto &vram = ctx.get<GBVram>(machine);
    auto &mbc = ctx.get<GBMbcRam>(machine);
    auto &frame = ctx.get<GBFrameBuffer>(machine);
    auto &obs = ctx.get<GBObs>(machine);
    auto &input = ctx.get<GBInput>(machine);
#ifndef MADRONA_GPU_MODE
    auto &serial = ctx.get<GBSerial>(machine);
#endif

    ::memset(&state, 0, sizeof(state));
    ::memset(&wram, 0, sizeof(wram));
    ::memset(&vram, 0, sizeof(vram));
    ::memset(&mbc, 0, sizeof(mbc));
    ::memset(&frame, 0, sizeof(frame));
    ::memset(&obs, 0, sizeof(obs));
#ifndef MADRONA_GPU_MODE
    ::memset(&serial, 0, sizeof(serial));
#endif
    input.buttons = 0;

    GB_gameboy_t *gb = &state.gb;
#ifdef MADRONA_GPU_MODE
    gb->model = GB_MODEL_DMG_B;
    gb->ram = wram.data;
    gb->ram_size = sizeof(wram.data);
    gb->vram = vram.data;
    gb->vram_size = sizeof(vram.data);
    gb->mbc_ram = mbc.data;
    gb->mbc_ram_size = sizeof(mbc.data);
    gb->rom = cfg.romData;
    gb->rom_size = (uint32_t)cfg.romSize;
    gb->ram_is_external = true;
    gb->vram_is_external = true;
    gb->mbc_ram_is_external = true;
    gb->rom_is_external = true;
    gb->cartridge_type = &GB_cart_defs[0];
    gb->clock_multiplier = 1.0;
    gb->apu_output.max_cycles_per_sample = 0x400;
    gb->data_bus_decay = 12;
    GB_reset(gb);
#else
    GB_init(gb, GB_MODEL_DMG_B);

    uint8_t *init_ram = gb->ram;
    uint8_t *init_vram = gb->vram;
    if (init_ram) {
        ::free(init_ram);
    }
    if (init_vram) {
        ::free(init_vram);
    }

    gb->ram = wram.data;
    gb->ram_size = sizeof(wram.data);
    gb->vram = vram.data;
    gb->vram_size = sizeof(vram.data);
    gb->mbc_ram = nullptr;
    gb->mbc_ram_size = 0;

    gb->rom = cfg.romData;
    gb->rom_size = (uint32_t)cfg.romSize;
    gb->ram_is_external = true;
    gb->vram_is_external = true;
    gb->rom_is_external = true;
#endif

    GB_configure_cart(gb);

#ifdef MADRONA_GPU_MODE
    if (gb->mbc_ram_size > sizeof(mbc.data)) {
        gb->mbc_ram_size = sizeof(mbc.data);
    }
    gb->mbc_ram = mbc.data;
#else
    if (gb->mbc_ram) {
        size_t mbc_size = gb->mbc_ram_size;
        if (mbc_size > sizeof(mbc.data)) {
            mbc_size = sizeof(mbc.data);
            gb->mbc_ram_size = (uint32_t)mbc_size;
        }
        ::memcpy(mbc.data, gb->mbc_ram, mbc_size);
        ::free(gb->mbc_ram);
        gb->mbc_ram = mbc.data;
    } else {
        gb->mbc_ram = mbc.data;
    }
    gb->mbc_ram_is_external = true;
#endif

    GB_set_border_mode(gb, GB_BORDER_NEVER);
    GB_set_rgb_encode_callback(gb, rgbEncode);
    GB_set_pixels_output(gb, frame.pixels);

#ifndef MADRONA_GPU_MODE
    GB_set_user_data(gb, &serial);
    GB_set_write_memory_callback(gb, serialWriteHook);
#endif

    initPostBootState(gb);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

} // namespace madSameBoy
