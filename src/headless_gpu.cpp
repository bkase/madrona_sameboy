#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>

#include <madrona/mw_gpu.hpp>

#include "sim.hpp"
#include "consts.hpp"

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

int main(int argc, char **argv)
{
    const char *rom_path = DEFAULT_ROM_PATH;
    uint32_t max_frames = 12000;
    uint32_t num_worlds = 1;
    int gpu_id = 0;
    uint32_t frames_per_step = 1;
    bool benchmark_only = false;
    bool null_step = false;
    bool fast_ppu = false;
    bool skip_ppu = false;
    uint32_t render_every = 1;

    std::vector<const char *> positional;
    positional.reserve(static_cast<size_t>(argc));
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (std::strcmp(arg, "--benchmark") == 0 ||
            std::strcmp(arg, "--bench") == 0) {
            benchmark_only = true;
            continue;
        }
        if (std::strcmp(arg, "--null-step") == 0 ||
            std::strcmp(arg, "--null") == 0) {
            null_step = true;
            continue;
        }
        if (std::strcmp(arg, "--fast-ppu") == 0) {
            fast_ppu = true;
            continue;
        }
        if (std::strcmp(arg, "--skip-ppu") == 0) {
            skip_ppu = true;
            continue;
        }
        if (std::strcmp(arg, "--render-every") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--render-every requires a value\n");
                return 2;
            }
            render_every = static_cast<uint32_t>(
                std::strtoul(argv[++i], nullptr, 10));
            if (render_every == 0) {
                render_every = 1;
            }
            continue;
        }
        if (arg[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", arg);
            return 2;
        }
        positional.push_back(arg);
    }

    if (positional.size() >= 1) {
        rom_path = positional[0];
    }
    if (positional.size() >= 2) {
        max_frames = static_cast<uint32_t>(
            std::strtoul(positional[1], nullptr, 10));
    }
    if (positional.size() >= 3) {
        num_worlds = static_cast<uint32_t>(
            std::strtoul(positional[2], nullptr, 10));
        if (num_worlds == 0) {
            num_worlds = 1;
        }
    }
    if (positional.size() >= 4) {
        gpu_id = static_cast<int>(
            std::strtol(positional[3], nullptr, 10));
    }
    if (positional.size() >= 5) {
        frames_per_step = static_cast<uint32_t>(
            std::strtoul(positional[4], nullptr, 10));
        if (frames_per_step == 0) {
            frames_per_step = 1;
        }
    }

    std::vector<uint8_t> rom_data;
    if (!readFile(rom_path, rom_data)) {
        fprintf(stderr, "Failed to read ROM: %s\n", rom_path);
        return 2;
    }

    size_t padded_size = roundedRomSize(rom_data.size());
    std::vector<uint8_t> rom_padded(padded_size, 0xFF);
    std::memcpy(rom_padded.data(), rom_data.data(), rom_data.size());

    // Initialize CUDA
    CUcontext cu_ctx = MWCudaExecutor::initCUDA(gpu_id);

    uint8_t *d_rom = nullptr;
    cudaMalloc(&d_rom, rom_padded.size());
    cudaMemcpy(d_rom, rom_padded.data(), rom_padded.size(), cudaMemcpyHostToDevice);

    Sim::Config sim_cfg {};
    sim_cfg.romData = d_rom;
    sim_cfg.romSize = rom_padded.size();
    sim_cfg.disableRendering = 1;
    sim_cfg.framesPerStep = frames_per_step;
    sim_cfg.useNullStep = null_step ? 1u : 0u;
    sim_cfg.fastPpu = fast_ppu ? 1u : 0u;
    sim_cfg.skipPpu = skip_ppu ? 1u : 0u;
    sim_cfg.renderEvery = render_every;

    std::vector<Sim::WorldInit> world_inits(num_worlds);

    // Source files for GPU compilation
    static const char *user_sources[] = {
        MADRONA_SAMEBOY_SIM_SRCS
    };

    static const char *user_compile_flags[] = {
        MADRONA_SAMEBOY_COMPILE_FLAGS
    };

    StateConfig state_cfg {
        .worldInitPtr = world_inits.data(),
        .numWorldInitBytes = sizeof(Sim::WorldInit),
        .userConfigPtr = &sim_cfg,
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
        .optMode = CompileConfig::OptMode::Optimize,
    };

    fprintf(stderr, "Compiling GPU kernels for %u worlds...\n", num_worlds);

    MWCudaExecutor exec(state_cfg, compile_cfg, cu_ctx);

    // Build launch graph for step taskgraph
    MWCudaLaunchGraph step_graph = exec.buildLaunchGraph(TaskGraphID::Step);

    // GPU pointers to exported columns
    GBInput *d_input = static_cast<GBInput *>(
        exec.getExported((uint32_t)ExportID::Input));
    GBObs *d_obs = static_cast<GBObs *>(
        exec.getExported((uint32_t)ExportID::Observation));

    // Host buffers for reading back results
    std::vector<GBInput> h_input(num_worlds);
    std::vector<GBObs> h_obs(num_worlds);

    // Initialize inputs to zero (no buttons pressed)
    for (auto &in : h_input) {
        in.buttons = 0;
    }
    cudaMemcpy(d_input, h_input.data(), sizeof(GBInput) * num_worlds,
               cudaMemcpyHostToDevice);

    uint64_t frames_per_world = (uint64_t)max_frames * frames_per_step;
    fprintf(stderr, "Running %llu frames on GPU...\n",
            static_cast<unsigned long long>(frames_per_world));

    auto start = std::chrono::high_resolution_clock::now();

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream\n");
        return 3;
    }

    for (uint32_t frame = 0; frame < max_frames; frame++) {
        exec.runAsync(step_graph, stream);
    }

    // Sync to ensure all work is done
    cudaStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (!benchmark_only) {
        // Copy observation back to check results
        cudaMemcpy(h_obs.data(), d_obs, sizeof(GBObs) * num_worlds,
                   cudaMemcpyDeviceToHost);
    }

    // Calculate performance metrics
    uint64_t total_frames = frames_per_world * num_worlds;
    double total_seconds = duration.count() / 1000000.0;
    double frames_per_sec = (double)total_frames / total_seconds;
    double fps_per_world = (double)frames_per_world / total_seconds;

    // Check observation pixel range
    uint8_t min_pix = 255;
    uint8_t max_pix = 0;
    if (!benchmark_only) {
        for (uint32_t i = 0; i < consts::screenPixels; i++) {
            uint8_t v = h_obs[0].pixels[i];
            if (v < min_pix) min_pix = v;
            if (v > max_pix) max_pix = v;
        }
    }

    printf("GPU Performance Results:\n");
    printf("  Worlds: %u\n", num_worlds);
    printf("  Frames per world: %llu\n",
           static_cast<unsigned long long>(frames_per_world));
    printf("  Total frames: %llu\n",
           static_cast<unsigned long long>(total_frames));
    printf("  Time: %.3f seconds\n", total_seconds);
    printf("  Total throughput: %.2f frames/sec\n", frames_per_sec);
    printf("  Per-world rate: %.2f FPS\n", fps_per_world);
    if (!benchmark_only) {
        printf("  Observation range: [%u, %u]\n", min_pix, max_pix);
    }

    cudaStreamDestroy(stream);
    cudaFree(d_rom);

    return 0;
}
