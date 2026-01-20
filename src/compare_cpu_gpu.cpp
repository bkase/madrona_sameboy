#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>

#include <madrona/mw_cpu.hpp>
#include <madrona/mw_gpu.hpp>

#include "sim.hpp"

using namespace madrona;
using namespace madSameBoy;

#ifndef MADRONA_SAMEBOY_SIM_SRCS
#define MADRONA_SAMEBOY_SIM_SRCS
#endif

#ifndef MADRONA_SAMEBOY_COMPILE_FLAGS
#define MADRONA_SAMEBOY_COMPILE_FLAGS
#endif

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

static uint64_t hashBytes(const uint8_t *data, size_t size)
{
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; i++) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

struct DiffSummary {
    size_t mismatches;
    size_t first_index;
    uint8_t cpu_value;
    uint8_t gpu_value;
};

static DiffSummary diffBytes(const uint8_t *cpu, const uint8_t *gpu, size_t size)
{
    DiffSummary diff {0, 0, 0, 0};
    for (size_t i = 0; i < size; i++) {
        if (cpu[i] == gpu[i]) {
            continue;
        }
        if (diff.mismatches == 0) {
            diff.first_index = i;
            diff.cpu_value = cpu[i];
            diff.gpu_value = gpu[i];
        }
        diff.mismatches++;
    }
    return diff;
}

static bool reportDiff(const char *label, uint64_t cpu_hash, uint64_t gpu_hash,
                       const uint8_t *cpu, const uint8_t *gpu, size_t size)
{
    if (cpu_hash == gpu_hash) {
        printf("%s hash: 0x%016" PRIx64 " (match)\n", label, cpu_hash);
        return true;
    }

    printf("%s hash mismatch: cpu=0x%016" PRIx64 " gpu=0x%016" PRIx64 "\n",
           label, cpu_hash, gpu_hash);
    DiffSummary diff = diffBytes(cpu, gpu, size);
    printf("  mismatches: %zu", diff.mismatches);
    if (diff.mismatches > 0) {
        printf(", first@0x%zx cpu=0x%02x gpu=0x%02x",
               diff.first_index, diff.cpu_value, diff.gpu_value);
    }
    printf("\n");
    return false;
}

int main(int argc, char **argv)
{
    const char *rom_path = DEFAULT_ROM_PATH;
    uint32_t num_frames = 120;
    uint32_t num_worlds = 1;
    int gpu_id = 0;

    if (argc >= 2) {
        rom_path = argv[1];
    }
    if (argc >= 3) {
        num_frames = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
    }
    if (argc >= 4) {
        num_worlds = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
        if (num_worlds == 0) {
            num_worlds = 1;
        }
    }
    if (argc >= 5) {
        gpu_id = static_cast<int>(std::strtol(argv[4], nullptr, 10));
    }

    std::vector<uint8_t> rom_data;
    if (!readFile(rom_path, rom_data)) {
        fprintf(stderr, "Failed to read ROM: %s\n", rom_path);
        return 2;
    }

    size_t padded_size = roundedRomSize(rom_data.size());
    std::vector<uint8_t> rom_padded(padded_size, 0xFF);
    std::memcpy(rom_padded.data(), rom_data.data(), rom_data.size());

    Sim::Config cpu_cfg {};
    cpu_cfg.romData = rom_padded.data();
    cpu_cfg.romSize = rom_padded.size();
    std::vector<Sim::WorldInit> world_inits(num_worlds);

    TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit> cpu_exec({
        .numWorlds = num_worlds,
        .numExportedBuffers = (uint32_t)ExportID::NumExports,
        .numWorkers = 0,
    }, cpu_cfg, world_inits.data(), (CountT)TaskGraphID::NumTaskGraphs);

    auto *cpu_input = static_cast<GBInput *>(
        cpu_exec.getExported((uint32_t)ExportID::Input));
    auto *cpu_wram = static_cast<GBRam *>(
        cpu_exec.getExported((uint32_t)ExportID::Ram));
    auto *cpu_vram = static_cast<GBVram *>(
        cpu_exec.getExported((uint32_t)ExportID::Vram));
    auto *cpu_mbc = static_cast<GBMbcRam *>(
        cpu_exec.getExported((uint32_t)ExportID::MbcRam));

    for (uint32_t i = 0; i < num_worlds; i++) {
        cpu_input[i].buttons = 0;
    }

    for (uint32_t frame = 0; frame < num_frames; frame++) {
        cpu_exec.run();
    }

    CUcontext cu_ctx = MWCudaExecutor::initCUDA(gpu_id);

    uint8_t *d_rom = nullptr;
    cudaMalloc(&d_rom, rom_padded.size());
    cudaMemcpy(d_rom, rom_padded.data(), rom_padded.size(), cudaMemcpyHostToDevice);

    Sim::Config gpu_cfg {};
    gpu_cfg.romData = d_rom;
    gpu_cfg.romSize = rom_padded.size();

    static const char *user_sources[] = {
        MADRONA_SAMEBOY_SIM_SRCS
    };

    static const char *user_compile_flags[] = {
        MADRONA_SAMEBOY_COMPILE_FLAGS
    };

    StateConfig state_cfg {
        .worldInitPtr = world_inits.data(),
        .numWorldInitBytes = sizeof(Sim::WorldInit),
        .userConfigPtr = &gpu_cfg,
        .numUserConfigBytes = sizeof(Sim::Config),
        .numWorldDataBytes = sizeof(Sim),
        .worldDataAlignment = alignof(Sim),
        .numWorlds = num_worlds,
        .numTaskGraphs = (uint32_t)TaskGraphID::NumTaskGraphs,
        .numExportedBuffers = (uint32_t)ExportID::NumExports,
    };

    CompileConfig compile_cfg {
        .userSources = {user_sources, std::size(user_sources)},
        .userCompileFlags = {user_compile_flags, std::size(user_compile_flags)},
        .optMode = CompileConfig::OptMode::Debug,
    };

    MWCudaExecutor gpu_exec(state_cfg, compile_cfg, cu_ctx);
    MWCudaLaunchGraph step_graph = gpu_exec.buildLaunchGraph(TaskGraphID::Step);

    auto *d_input = static_cast<GBInput *>(
        gpu_exec.getExported((uint32_t)ExportID::Input));
    auto *d_wram = static_cast<GBRam *>(
        gpu_exec.getExported((uint32_t)ExportID::Ram));
    auto *d_vram = static_cast<GBVram *>(
        gpu_exec.getExported((uint32_t)ExportID::Vram));
    auto *d_mbc = static_cast<GBMbcRam *>(
        gpu_exec.getExported((uint32_t)ExportID::MbcRam));

    std::vector<GBInput> h_input(num_worlds);
    for (auto &in : h_input) {
        in.buttons = 0;
    }
    cudaMemcpy(d_input, h_input.data(), sizeof(GBInput) * num_worlds,
               cudaMemcpyHostToDevice);

    for (uint32_t frame = 0; frame < num_frames; frame++) {
        gpu_exec.run(step_graph);
    }

    cudaDeviceSynchronize();

    std::vector<GBRam> h_wram(num_worlds);
    std::vector<GBVram> h_vram(num_worlds);
    std::vector<GBMbcRam> h_mbc(num_worlds);

    cudaMemcpy(h_wram.data(), d_wram, sizeof(GBRam) * num_worlds,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vram.data(), d_vram, sizeof(GBVram) * num_worlds,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mbc.data(), d_mbc, sizeof(GBMbcRam) * num_worlds,
               cudaMemcpyDeviceToHost);

    cudaFree(d_rom);

    bool all_match = true;
    for (uint32_t i = 0; i < num_worlds; i++) {
        uint64_t cpu_wram_hash = hashBytes(cpu_wram[i].data, sizeof(cpu_wram[i].data));
        uint64_t cpu_vram_hash = hashBytes(cpu_vram[i].data, sizeof(cpu_vram[i].data));
        uint64_t cpu_mbc_hash = hashBytes(cpu_mbc[i].data, sizeof(cpu_mbc[i].data));

        uint64_t gpu_wram_hash = hashBytes(h_wram[i].data, sizeof(h_wram[i].data));
        uint64_t gpu_vram_hash = hashBytes(h_vram[i].data, sizeof(h_vram[i].data));
        uint64_t gpu_mbc_hash = hashBytes(h_mbc[i].data, sizeof(h_mbc[i].data));

        if (num_worlds > 1) {
            printf("World %u:\n", i);
        }

        bool wram_match = reportDiff("WRAM", cpu_wram_hash, gpu_wram_hash,
                                     cpu_wram[i].data, h_wram[i].data,
                                     sizeof(cpu_wram[i].data));
        bool vram_match = reportDiff("VRAM", cpu_vram_hash, gpu_vram_hash,
                                     cpu_vram[i].data, h_vram[i].data,
                                     sizeof(cpu_vram[i].data));
        bool mbc_match = reportDiff("MBC RAM", cpu_mbc_hash, gpu_mbc_hash,
                                    cpu_mbc[i].data, h_mbc[i].data,
                                    sizeof(cpu_mbc[i].data));

        all_match = all_match && wram_match && vram_match && mbc_match;
    }

    return all_match ? 0 : 1;
}
