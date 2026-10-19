// budget.cpp — Read real Jetson memory state from /proc

#include "jllm_memory.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace jllm {

static int64_t parse_meminfo_field(const char* field) {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return -1;

    char line[256];
    int64_t value = -1;
    size_t flen = strlen(field);

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, field, flen) == 0) {
            // Format: "FieldName:     12345 kB"
            char* p = line + flen;
            while (*p == ' ' || *p == ':') p++;
            value = strtoll(p, nullptr, 10) / 1024;  // kB → MB
            break;
        }
    }
    fclose(f);
    return value;
}

MemoryBudget probe_system_memory() {
    MemoryBudget b = {};

    b.total_mb = parse_meminfo_field("MemTotal");
    int64_t available = parse_meminfo_field("MemAvailable");
    b.cma_mb = parse_meminfo_field("CmaTotal");
    if (b.cma_mb < 0) b.cma_mb = 0;
    if (b.total_mb <= 0 || available < 0 || available > b.total_mb) return {};

    // Estimate OS usage = total - available (before we load anything)
    b.os_mb = b.total_mb - available; // all current unavailability; CMA is not added again

    // CUDA context: probe by actually initializing
    // (deferred — set estimate, update after cudaSetDevice)
    b.cuda_ctx_mb = 300;

    b.model_mb = 0;
    b.kv_cache_mb = 0;
    b.scratch_mb = 0;
    b.safety_mb = 256;

    return b;
}

// ── OOM Guard ────────────────────────────────────────────────────────────

bool OOMGuard::can_extend(int64_t additional_bytes) const {
    int64_t free = real_free_mb();
    if (additional_bytes < 0 || safety_mb_ < 0 || free < safety_mb_) return false;
    const int64_t unit = 1024 * 1024;
    const int64_t needed_mb = additional_bytes / unit + (additional_bytes % unit != 0);
    return needed_mb <= free - safety_mb_;
}

int64_t OOMGuard::real_free_mb() const {
    return parse_meminfo_field("MemAvailable");
}

void OOMGuard::emergency_free() {
    // Reclaim only allocations owned by this runtime in a future implementation.
    // Global drop_caches/compaction does not reserve memory for this process.
    fprintf(stderr, "[memory] no owned reclaimable cache is registered\n");
}

}  // namespace jllm
