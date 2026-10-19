// jllm_jetson.h — Jetson hardware abstraction (power, thermal, clocks)
//
// Reads directly from sysfs/procfs — no NVIDIA SDK dependency.
// Paths must be checked against the installed BSP; universal release support is not claimed.

#pragma once

#include <cstdint>

namespace jllm {

// ── Power mode ───────────────────────────────────────────────────────────

enum PowerMode {
    POWER_MAXN = 0,   // semantic label; MAXN is not a universal 25W limit
    POWER_15W  = 1,
    POWER_10W  = 2,
    POWER_7W   = 3,   // minimum — battery / fanless
    POWER_UNKNOWN = -1
};

struct PowerState {
    PowerMode mode;
    int       watts;
    int       gpu_freq_mhz;     // current GPU frequency
    int       gpu_freq_max_mhz; // max for this power mode
    int       emc_freq_mhz;     // memory controller frequency
    int       cpu_freq_mhz;     // max CPU frequency
    int       cpu_online;       // number of online CPU cores
};

PowerState read_power_state();
void       set_power_mode(PowerMode mode);  // legacy unsupported semantic-to-ID mapping
void       lock_clocks();                   // calls jetson_clocks

// ── Thermal ──────────────────────────────────────────────────────────────

struct ThermalState {
    float cpu_temp_c;
    float gpu_temp_c;
    float board_temp_c;
    bool  throttling;     // true if any zone above trip point
};

ThermalState read_thermal();

// Should we slow down to prevent throttling?
// Returns recommended sleep_us between tokens (0 = full speed)
int thermal_backoff_us(const ThermalState& ts);

// ── System info (one-time probe at startup) ──────────────────────────────

struct JetsonInfo {
    char   l4t_version[32];      // e.g., "36.4.0"
    char   jetpack_version[16];  // e.g., "6.1"
    int    cuda_major;
    int    cuda_minor;
    int    gpu_sm_count;
    int    gpu_cuda_cores;
    int64_t total_ram_mb;
    int64_t cma_total_mb;
    int64_t nvme_free_mb;
};

JetsonInfo probe_jetson();
void       print_jetson_info(const JetsonInfo& info);

// ── tegrastats-style live monitor ────────────────────────────────────────

struct LiveStats {
    int64_t ram_used_mb;
    int64_t ram_total_mb;
    int     gpu_util_pct;     // 0-100
    int     gpu_freq_mhz;
    float   power_watts;      // total board power
    float   gpu_temp_c;
    float   tokens_per_sec;   // set by engine
};

LiveStats read_live_stats();
void      print_live_stats(const LiveStats& s);

}  // namespace jllm
