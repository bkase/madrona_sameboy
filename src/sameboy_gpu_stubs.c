// GPU stubs for SameBoy optional modules not used in Madrona GPU builds.

typedef struct GB_gameboy_s GB_gameboy_t;

#ifndef GB_ENUM
#define GB_ENUM(type, ...) enum : type __VA_ARGS__
#endif

#include "defs.h"
#include "camera.h"
#include "printer.h"
#include "rumble.h"
#include "sgb.h"
#include "workboy.h"

#if defined(GB_GPU_MODE)

void GB_set_camera_get_pixel_callback(GB_gameboy_t *gb,
                                      GB_camera_get_pixel_callback_t callback)
{
    (void)gb;
    (void)callback;
}

void GB_set_camera_update_request_callback(
    GB_gameboy_t *gb,
    GB_camera_update_request_callback_t callback)
{
    (void)gb;
    (void)callback;
}

void GB_camera_updated(GB_gameboy_t *gb)
{
    (void)gb;
}

uint8_t GB_camera_read_image(GB_gameboy_t *gb, uint16_t addr)
{
    (void)gb;
    (void)addr;
    return 0;
}

void GB_camera_write_register(GB_gameboy_t *gb, uint16_t addr, uint8_t value)
{
    (void)gb;
    (void)addr;
    (void)value;
}

uint8_t GB_camera_read_register(GB_gameboy_t *gb, uint16_t addr)
{
    (void)gb;
    (void)addr;
    return 0;
}

void GB_connect_printer(GB_gameboy_t *gb,
                        GB_print_image_callback_t callback,
                        GB_printer_done_callback_t done_callback)
{
    (void)callback;
    (void)done_callback;
    (void)gb;
}

void GB_set_rumble_mode(GB_gameboy_t *gb, GB_rumble_mode_t mode)
{
    (void)gb;
    (void)mode;
}

void GB_handle_rumble(GB_gameboy_t *gb)
{
    (void)gb;
}

void GB_sgb_write(GB_gameboy_t *gb, uint8_t value)
{
    (void)gb;
    (void)value;
}

void GB_sgb_render(GB_gameboy_t *gb, bool incomplete)
{
    (void)gb;
    (void)incomplete;
}

void GB_sgb_load_default_data(GB_gameboy_t *gb)
{
    (void)gb;
}

void GB_connect_workboy(GB_gameboy_t *gb,
                        GB_workboy_set_time_callback_t set_time_callback,
                        GB_workboy_get_time_callback_t get_time_callback)
{
    (void)set_time_callback;
    (void)get_time_callback;
    (void)gb;
    (void)set_time_callback;
    (void)get_time_callback;
}

bool GB_workboy_is_enabled(GB_gameboy_t *gb)
{
    (void)gb;
    return false;
}

void GB_workboy_set_key(GB_gameboy_t *gb, uint8_t key)
{
    (void)gb;
    (void)key;
}

#endif
