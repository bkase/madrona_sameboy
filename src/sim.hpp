#pragma once

#include <cstddef>

#include <madrona/taskgraph_builder.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/ecs.hpp>

#include "types.hpp"

namespace madSameBoy {

class Engine;

enum class TaskGraphID : uint32_t {
    Step,
    NumTaskGraphs,
};

enum class ExportID : uint32_t {
    Input,
    Observation,
    Ram,
    Vram,
    MbcRam,
    Regs,
    NumExports,
};

struct Sim : public madrona::WorldBase {
    struct Config {
        uint8_t *romData;
        size_t romSize;
        uint32_t disableRendering;
        uint32_t framesPerStep;
        uint32_t useNullStep;
        uint32_t fastPpu;
    };

    struct WorldInit {};

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);
    static void setupTasks(madrona::TaskGraphManager &mgr,
                           const Config &cfg);

    Sim(Engine &ctx, const Config &cfg, const WorldInit &);

    madrona::Entity machine;
    uint32_t disableRendering;
    uint32_t framesPerStep;
    uint32_t useNullStep;
    uint32_t fastPpu;
};

class Engine : public madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;
};

}
