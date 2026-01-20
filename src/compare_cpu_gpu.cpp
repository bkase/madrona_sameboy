#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
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

static void packObs(const GBObs &obs, GBObsPacked &out)
{
    constexpr size_t packed_size = consts::screenPixels / 4;
    for (size_t i = 0; i < packed_size; i++) {
        size_t base = i * 4;
        uint8_t b0 = obs.pixels[base] & 0x3;
        uint8_t b1 = obs.pixels[base + 1] & 0x3;
        uint8_t b2 = obs.pixels[base + 2] & 0x3;
        uint8_t b3 = obs.pixels[base + 3] & 0x3;
        out.bytes[i] = (uint8_t)(b0 | (b1 << 2) | (b2 << 4) | (b3 << 6));
    }
}

struct DiffSummary {
    size_t mismatches;
    size_t first_index;
    uint8_t cpu_value;
    uint8_t gpu_value;
};

struct BufferReport {
    uint64_t cpu_hash;
    uint64_t gpu_hash;
    DiffSummary diff;
    bool match;
};

struct RegsReport {
    bool match;
    GBRegs cpu;
    GBRegs gpu;
};

struct WorldReport {
    uint32_t world;
    BufferReport wram;
    BufferReport vram;
    BufferReport mbc;
    BufferReport obs;
    RegsReport regs;
};

static std::string hex64(uint64_t value)
{
    char buf[19];
    std::snprintf(buf, sizeof(buf), "0x%016" PRIx64, value);
    return std::string(buf);
}

static std::string jsonEscape(const char *value)
{
    std::string out;
    for (const char *ptr = value; *ptr != '\0'; ++ptr) {
        switch (*ptr) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(*ptr) < 0x20) {
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x",
                              static_cast<unsigned char>(*ptr));
                out += buf;
            } else {
                out += *ptr;
            }
            break;
        }
    }
    return out;
}

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

static BufferReport reportDiff(const char *label, const uint8_t *cpu,
                               const uint8_t *gpu, size_t size)
{
    BufferReport report {};
    report.cpu_hash = hashBytes(cpu, size);
    report.gpu_hash = hashBytes(gpu, size);
    if (report.cpu_hash == report.gpu_hash) {
        printf("%s hash: 0x%016" PRIx64 " (match)\n", label, report.cpu_hash);
        report.match = true;
        report.diff = {0, 0, 0, 0};
        return report;
    }

    printf("%s hash mismatch: cpu=0x%016" PRIx64 " gpu=0x%016" PRIx64 "\n",
           label, report.cpu_hash, report.gpu_hash);
    report.match = false;
    report.diff = diffBytes(cpu, gpu, size);
    printf("  mismatches: %zu", report.diff.mismatches);
    if (report.diff.mismatches > 0) {
        printf(", first@0x%zx cpu=0x%02x gpu=0x%02x",
               report.diff.first_index, report.diff.cpu_value,
               report.diff.gpu_value);
    }
    printf("\n");
    return report;
}

static void printUsage(const char *exe)
{
    printf("Usage: %s [rom] [frames] [worlds] [gpu_id] [--report path]\n", exe);
}

static bool writeReport(const char *path, const char *rom_path, uint32_t frames,
                        uint32_t worlds, const std::vector<WorldReport> &reports,
                        bool all_match)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }

    out << "{\n";
    out << "  \"rom\": \"" << jsonEscape(rom_path) << "\",\n";
    out << "  \"frames\": " << frames << ",\n";
    out << "  \"worlds\": " << worlds << ",\n";
    out << "  \"thresholds\": {\n";
    out << "    \"allowed_mismatches\": 0,\n";
    out << "    \"regs_match_required\": true\n";
    out << "  },\n";
    out << "  \"world_results\": [\n";

    for (size_t i = 0; i < reports.size(); i++) {
        const WorldReport &world = reports[i];
        out << "    {\n";
        out << "      \"world\": " << world.world << ",\n";
        out << "      \"buffers\": {\n";

        auto writeBuffer = [&out](const char *name, const BufferReport &buffer, bool last) {
            out << "        \"" << name << "\": {\n";
            out << "          \"match\": " << (buffer.match ? "true" : "false") << ",\n";
            out << "          \"cpu_hash\": \"" << hex64(buffer.cpu_hash) << "\",\n";
            out << "          \"gpu_hash\": \"" << hex64(buffer.gpu_hash) << "\",\n";
            out << "          \"mismatches\": " << buffer.diff.mismatches << ",\n";
            out << "          \"first_index\": " << buffer.diff.first_index << ",\n";
            out << "          \"cpu_value\": "
                << static_cast<unsigned>(buffer.diff.cpu_value) << ",\n";
            out << "          \"gpu_value\": "
                << static_cast<unsigned>(buffer.diff.gpu_value) << "\n";
            out << "        }" << (last ? "" : ",") << "\n";
        };

        writeBuffer("wram", world.wram, false);
        writeBuffer("vram", world.vram, false);
        writeBuffer("mbc_ram", world.mbc, false);
        writeBuffer("obs_packed", world.obs, true);

        out << "      },\n";
        out << "      \"regs\": {\n";
        out << "        \"match\": " << (world.regs.match ? "true" : "false") << ",\n";
        out << "        \"cpu\": {\n";
        out << "          \"pc\": " << world.regs.cpu.pc << ",\n";
        out << "          \"sp\": " << world.regs.cpu.sp << ",\n";
        out << "          \"ly\": " << static_cast<unsigned>(world.regs.cpu.ly) << ",\n";
        out << "          \"stat\": " << static_cast<unsigned>(world.regs.cpu.stat) << "\n";
        out << "        },\n";
        out << "        \"gpu\": {\n";
        out << "          \"pc\": " << world.regs.gpu.pc << ",\n";
        out << "          \"sp\": " << world.regs.gpu.sp << ",\n";
        out << "          \"ly\": " << static_cast<unsigned>(world.regs.gpu.ly) << ",\n";
        out << "          \"stat\": " << static_cast<unsigned>(world.regs.gpu.stat) << "\n";
        out << "        }\n";
        out << "      }\n";
        out << "    }" << (i + 1 < reports.size() ? "," : "") << "\n";
    }

    out << "  ],\n";
    out << "  \"all_match\": " << (all_match ? "true" : "false") << "\n";
    out << "}\n";
    return true;
}

int main(int argc, char **argv)
{
    const char *rom_path = DEFAULT_ROM_PATH;
    uint32_t num_frames = 120;
    uint32_t num_worlds = 1;
    int gpu_id = 0;
    const char *report_path = nullptr;

    std::vector<const char *> positional;
    positional.reserve(argc);
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[i], "--report") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--report requires a path\n");
                return 2;
            }
            report_path = argv[++i];
            continue;
        }
        if (std::strncmp(argv[i], "--report=", 9) == 0) {
            report_path = argv[i] + 9;
            continue;
        }
        positional.push_back(argv[i]);
    }

    if (positional.size() >= 1) {
        rom_path = positional[0];
    }
    if (positional.size() >= 2) {
        num_frames = static_cast<uint32_t>(std::strtoul(positional[1], nullptr, 10));
    }
    if (positional.size() >= 3) {
        num_worlds = static_cast<uint32_t>(std::strtoul(positional[2], nullptr, 10));
        if (num_worlds == 0) {
            num_worlds = 1;
        }
    }
    if (positional.size() >= 4) {
        gpu_id = static_cast<int>(std::strtol(positional[3], nullptr, 10));
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
    cpu_cfg.disableRendering = 0;
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
    auto *cpu_obs = static_cast<GBObs *>(
        cpu_exec.getExported((uint32_t)ExportID::Observation));
    auto *cpu_regs = static_cast<GBRegs *>(
        cpu_exec.getExported((uint32_t)ExportID::Regs));

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
    gpu_cfg.disableRendering = 0;

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
    auto *d_obs = static_cast<GBObs *>(
        gpu_exec.getExported((uint32_t)ExportID::Observation));
    auto *d_regs = static_cast<GBRegs *>(
        gpu_exec.getExported((uint32_t)ExportID::Regs));

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
    std::vector<GBObs> h_obs(num_worlds);
    std::vector<GBRegs> h_regs(num_worlds);

    cudaMemcpy(h_wram.data(), d_wram, sizeof(GBRam) * num_worlds,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vram.data(), d_vram, sizeof(GBVram) * num_worlds,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mbc.data(), d_mbc, sizeof(GBMbcRam) * num_worlds,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs.data(), d_obs, sizeof(GBObs) * num_worlds,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_regs.data(), d_regs, sizeof(GBRegs) * num_worlds,
               cudaMemcpyDeviceToHost);

    cudaFree(d_rom);

    bool all_match = true;
    std::vector<WorldReport> world_reports;
    world_reports.reserve(num_worlds);
    for (uint32_t i = 0; i < num_worlds; i++) {
        GBObsPacked cpu_packed {};
        GBObsPacked gpu_packed {};
        packObs(cpu_obs[i], cpu_packed);
        packObs(h_obs[i], gpu_packed);

        if (num_worlds > 1) {
            printf("World %u:\n", i);
        }

        BufferReport wram_report = reportDiff("WRAM", cpu_wram[i].data,
                                              h_wram[i].data,
                                              sizeof(cpu_wram[i].data));
        BufferReport vram_report = reportDiff("VRAM", cpu_vram[i].data,
                                              h_vram[i].data,
                                              sizeof(cpu_vram[i].data));
        BufferReport mbc_report = reportDiff("MBC RAM", cpu_mbc[i].data,
                                             h_mbc[i].data,
                                             sizeof(cpu_mbc[i].data));
        BufferReport obs_report = reportDiff("ObsPacked", cpu_packed.bytes,
                                             gpu_packed.bytes,
                                             sizeof(cpu_packed.bytes));

        RegsReport regs_report {};
        const GBRegs &cpu_reg = cpu_regs[i];
        const GBRegs &gpu_reg = h_regs[i];
        regs_report.cpu = cpu_reg;
        regs_report.gpu = gpu_reg;
        if (cpu_reg.pc != gpu_reg.pc || cpu_reg.sp != gpu_reg.sp ||
            cpu_reg.ly != gpu_reg.ly || cpu_reg.stat != gpu_reg.stat) {
            regs_report.match = false;
            printf("Regs mismatch: cpu PC=0x%04x SP=0x%04x LY=%u STAT=0x%02x "
                   "gpu PC=0x%04x SP=0x%04x LY=%u STAT=0x%02x\n",
                   cpu_reg.pc, cpu_reg.sp, cpu_reg.ly, cpu_reg.stat,
                   gpu_reg.pc, gpu_reg.sp, gpu_reg.ly, gpu_reg.stat);
        } else {
            regs_report.match = true;
        }

        world_reports.push_back(WorldReport {
            i, wram_report, vram_report, mbc_report, obs_report, regs_report
        });

        all_match = all_match && wram_report.match && vram_report.match &&
            mbc_report.match && obs_report.match && regs_report.match;
    }

    if (report_path != nullptr) {
        if (!writeReport(report_path, rom_path, num_frames, num_worlds,
                         world_reports, all_match)) {
            fprintf(stderr, "Failed to write report: %s\n", report_path);
            return 2;
        }
    }

    return all_match ? 0 : 1;
}
