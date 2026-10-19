// power.cpp — Jetson power mode and clock control via sysfs

#include "jllm_jetson.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace jllm {

static int read_sysfs_int(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int val = 0;
    fscanf(f, "%d", &val);
    fclose(f);
    return val;
}

PowerState read_power_state() {
    PowerState ps = {};
    ps.mode = POWER_UNKNOWN; ps.watts = -1;

    // GPU frequency
    ps.gpu_freq_mhz = read_sysfs_int(
        "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq") / 1000000;
    ps.gpu_freq_max_mhz = read_sysfs_int(
        "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/max_freq") / 1000000;

    // EMC (memory controller) frequency
    ps.emc_freq_mhz = read_sysfs_int(
        "/sys/kernel/debug/bpmp/debug/clk/emc/rate") / 1000000;

    // CPU frequency (first online core)
    ps.cpu_freq_mhz = read_sysfs_int(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") / 1000;

    // Count online CPUs
    ps.cpu_online = 0;
    for (int i = 0; i < 8; i++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online", i);
        if (read_sysfs_int(path) == 1) ps.cpu_online++;
    }

    // Power mode from nvpmodel
    FILE* f = popen("nvpmodel -q 2>/dev/null", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "MAXN"))  { ps.mode = POWER_MAXN; ps.watts = -1; break; }
            if (strstr(line, "15W"))   { ps.mode = POWER_15W;  ps.watts = 15; break; }
            if (strstr(line, "10W"))   { ps.mode = POWER_10W;  ps.watts = 10; break; }
            if (strstr(line, "7W"))    { ps.mode = POWER_7W;   ps.watts = 7;  break; }
        }
        pclose(f);
    } else {
        ps.mode = POWER_UNKNOWN;
        ps.watts = -1;
    }

    return ps;
}

void set_power_mode(PowerMode mode) {
    (void)mode;
    fprintf(stderr, "[power] semantic mode has no portable nvpmodel ID; inspect the installed configuration\n");
}

void lock_clocks() {
    system("jetson_clocks 2>/dev/null");
}

}  // namespace jllm
