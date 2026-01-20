#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include <madrona/mw_cpu.hpp>

#include "sim.hpp"

using namespace madrona;
using namespace madSameBoy;

static bool readFile(const char *path, std::vector<uint8_t> &out)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return false;
    }

    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);

    if (size <= 0) {
        return false;
    }

    out.resize(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char *>(out.data()), size)) {
        return false;
    }

    return true;
}

static size_t roundedRomSize(size_t size)
{
    size = (size + 0x3FFF) & ~size_t(0x3FFF);
    while (size & (size - 1)) {
        size |= size >> 1;
        size++;
    }
    if (size < 0x8000) {
        size = 0x8000;
    }
    return size;
}

static bool containsToken(const GBSerial &serial, const char *token)
{
    if (serial.length == 0) {
        return false;
    }
    return std::strstr(serial.text, token) != nullptr;
}

static uint32_t decodeTileMapMode(const GBVram &vram, uint16_t map_base,
                                  uint8_t offset, char *out, size_t out_size)
{
    if (out_size < 18 * 21 + 1) {
        return 0;
    }

    size_t pos = 0;
    uint32_t printable = 0;
    for (uint32_t row = 0; row < 18; row++) {
        for (uint32_t col = 0; col < 20; col++) {
            uint8_t tile = vram.data[map_base + row * 32 + col];
            uint8_t ascii = (uint8_t)(tile + offset);
            char ch = ' ';
            if (ascii >= 0x20 && ascii <= 0x7E) {
                ch = static_cast<char>(ascii);
                if (ascii != 0x20) {
                    printable++;
                }
            }
            out[pos++] = ch;
        }
        out[pos++] = '\n';
    }
    out[pos] = '\0';
    return printable;
}

static bool decodeTileMapBest(const GBVram &vram, const GBState &state,
                              char *out, size_t out_size)
{
    constexpr size_t kTileBufSize = 18 * 21 + 1;
    if (out_size < kTileBufSize) {
        return false;
    }

    uint8_t lcdc = state.gb.io_registers[GB_IO_LCDC];
    uint16_t bg_base = (lcdc & 0x08) ? 0x1C00 : 0x1800;
    uint16_t win_base = (lcdc & 0x40) ? 0x1C00 : 0x1800;

    const uint8_t offsets[] = {0x00, 0x20, 0x10, 0x30};
    char best_buf[kTileBufSize];
    best_buf[0] = '\0';
    uint32_t best_score = 0;

    for (uint8_t offset : offsets) {
        uint32_t score = decodeTileMapMode(vram, bg_base, offset,
                                           best_buf, sizeof(best_buf));
        if (score > best_score) {
            best_score = score;
            std::memcpy(out, best_buf, kTileBufSize);
        }
        uint32_t win_score = decodeTileMapMode(vram, win_base, offset,
                                               best_buf, sizeof(best_buf));
        if (win_score > best_score) {
            best_score = win_score;
            std::memcpy(out, best_buf, kTileBufSize);
        }
    }

    return best_score > 0;
}

int main(int argc, char **argv)
{
    const char *rom_path = DEFAULT_ROM_PATH;
    uint32_t max_frames = 12000;
    uint32_t num_worlds = 1;
    uint32_t num_workers = 0;

    if (argc >= 2) {
        rom_path = argv[1];
    }
    if (argc >= 3) {
        max_frames = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
    }
    if (argc >= 4) {
        num_worlds = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
        if (num_worlds == 0) {
            num_worlds = 1;
        }
    }
    if (argc >= 5) {
        num_workers = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
    }

    std::vector<uint8_t> rom_data;
    if (!readFile(rom_path, rom_data)) {
        fprintf(stderr, "Failed to read ROM: %s\n", rom_path);
        return 2;
    }

    size_t padded_size = roundedRomSize(rom_data.size());
    std::vector<uint8_t> rom_padded(padded_size, 0xFF);
    std::memcpy(rom_padded.data(), rom_data.data(), rom_data.size());

    Sim::Config sim_cfg {};
    sim_cfg.romData = rom_padded.data();
    sim_cfg.romSize = rom_padded.size();

    std::vector<Sim::WorldInit> world_inits(num_worlds);

    TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit> exec({
        .numWorlds = num_worlds,
        .numExportedBuffers = (uint32_t)ExportID::NumExports,
        .numWorkers = num_workers,
    }, sim_cfg, world_inits.data(), (CountT)TaskGraphID::NumTaskGraphs);

    struct WorldRefs {
        GBSerial *serial;
        GBVram *vram;
        GBObs *obs;
        GBState *state;
    };
    std::vector<WorldRefs> worlds;
    worlds.reserve(num_worlds);
    for (uint32_t i = 0; i < num_worlds; i++) {
        auto &ctx = exec.getWorldContext(i);
        auto machine = ctx.data().machine;
        worlds.push_back({
            &ctx.get<GBSerial>(machine),
            &ctx.get<GBVram>(machine),
            &ctx.get<GBObs>(machine),
            &ctx.get<GBState>(machine),
        });
    }

    std::vector<bool> saw_pass(num_worlds, false);
    std::vector<bool> saw_fail(num_worlds, false);
    std::vector<std::array<char, 18 * 21 + 1>> tile_text(num_worlds);
    for (auto &buf : tile_text) {
        buf[0] = '\0';
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t frame = 0; frame < max_frames; frame++) {
        exec.run();
        bool all_done = true;
        for (uint32_t i = 0; i < num_worlds; i++) {
            if (saw_pass[i] || saw_fail[i]) {
                continue;
            }
            auto &serial = *worlds[i].serial;
            if (serial.length > 0) {
                saw_pass[i] = containsToken(serial, "Passed");
                saw_fail[i] = containsToken(serial, "Failed");
            }
            if (!(saw_pass[i] || saw_fail[i])) {
                auto &buf = tile_text[i];
                if (decodeTileMapBest(*worlds[i].vram, *worlds[i].state,
                                      buf.data(), buf.size())) {
                    saw_pass[i] = std::strstr(buf.data(), "Passed") != nullptr;
                    saw_fail[i] = std::strstr(buf.data(), "Failed") != nullptr;
                }
            }
            if (!(saw_pass[i] || saw_fail[i])) {
                all_done = false;
            }
        }
        if (all_done) {
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    auto &serial0 = *worlds[0].serial;
    if (serial0.length > 0) {
        fputs(serial0.text, stdout);
        if (serial0.text[serial0.length - 1] != '\n') {
            fputc('\n', stdout);
        }
    } else if (tile_text[0][0] != '\0') {
        fputs(tile_text[0].data(), stdout);
    } else {
        fprintf(stderr, "No serial output after %u frames\n", max_frames);
    }

    uint8_t min_pix = 255;
    uint8_t max_pix = 0;
    for (uint32_t i = 0; i < consts::screenPixels; i++) {
        uint8_t v = worlds[0].obs->pixels[i];
        if (v < min_pix) min_pix = v;
        if (v > max_pix) max_pix = v;
    }

    fprintf(stderr, "Serial writes: SB=%u SC=%u, obs range=[%u,%u]\n",
            serial0.sbWrites, serial0.scWrites, min_pix, max_pix);
    fprintf(stderr, "PC=0x%04X SP=0x%04X LCDC=0x%02X\n",
            worlds[0].state->gb.pc, worlds[0].state->gb.sp,
            worlds[0].state->gb.io_registers[GB_IO_LCDC]);

    // Performance output
    double total_seconds = duration.count() / 1000000.0;
    double frames_per_sec = (double)(max_frames * num_worlds) / total_seconds;
    double fps_per_world = (double)max_frames / total_seconds;

    fprintf(stderr, "\nCPU Performance Results:\n");
    fprintf(stderr, "  Worlds: %u\n", num_worlds);
    fprintf(stderr, "  Frames per world: %u\n", max_frames);
    fprintf(stderr, "  Total frames: %u\n", max_frames * num_worlds);
    fprintf(stderr, "  Time: %.3f seconds\n", total_seconds);
    fprintf(stderr, "  Total throughput: %.2f frames/sec\n", frames_per_sec);
    fprintf(stderr, "  Per-world rate: %.2f FPS\n", fps_per_world);

    bool any_fail = false;
    bool all_pass = true;
    for (uint32_t i = 0; i < num_worlds; i++) {
        if (saw_fail[i]) {
            any_fail = true;
        }
        if (!saw_pass[i]) {
            all_pass = false;
        }
        if (num_worlds > 1) {
            fprintf(stderr, "World %u: %s\n", i,
                    saw_fail[i] ? "Failed" : (saw_pass[i] ? "Passed" : "Unknown"));
        }
    }
    if (any_fail) {
        return 1;
    }
    if (all_pass) {
        return 0;
    }
    return 3;
}
