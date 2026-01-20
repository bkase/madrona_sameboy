#pragma once

#include <cstdint>
#include <cstddef>

#include <madrona/components.hpp>
#include <madrona/ecs.hpp>

#include "consts.hpp"

extern "C" {
#include "gb.h"
}

namespace madSameBoy {

struct GBState {
    GB_gameboy_t gb;
};

struct GBRam {
    uint8_t data[0x2000];
};

struct GBVram {
    uint8_t data[0x2000];
};

struct GBMbcRam {
    uint8_t data[0x8000];
};

struct GBFrameBuffer {
    uint32_t pixels[consts::screenPixels];
};

struct GBObs {
    uint8_t pixels[consts::screenPixels];
};

struct GBObsPacked {
    uint8_t bytes[consts::screenPixels / 4];
};

struct GBInput {
    uint8_t buttons;
};

struct GBSerial {
    uint8_t curByte;
    uint8_t bitCount;
    uint8_t lastSB;
    uint8_t overflowed;
    uint8_t _pad;
    uint32_t length;
    uint32_t sbWrites;
    uint32_t scWrites;
    char text[4096];
};

#ifdef MADRONA_GPU_MODE
struct GBMachine : public madrona::Archetype<
    GBState,
    GBRam,
    GBVram,
    GBMbcRam,
    GBFrameBuffer,
    GBObs,
    GBInput
> {};
#else
struct GBMachine : public madrona::Archetype<
    GBState,
    GBRam,
    GBVram,
    GBMbcRam,
    GBFrameBuffer,
    GBObs,
    GBInput,
    GBSerial
> {};
#endif

}
