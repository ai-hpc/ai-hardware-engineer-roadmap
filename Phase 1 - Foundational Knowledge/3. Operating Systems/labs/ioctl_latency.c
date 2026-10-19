#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "ioctl_latency.h"
#include "ioctl_latency.skel.h"

_Static_assert(sizeof(struct ioctl_event) == 48, "event ABI changed");
static volatile sig_atomic_t stop;
static void interrupted(int signal_number) { (void)signal_number; stop = 1; }
static int consume(void *context, void *data, size_t size)
{
    (void)context;
    if (size != sizeof(struct ioctl_event)) return -EMSGSIZE;
    struct ioctl_event event;
    memcpy(&event, data, sizeof(event));
    printf("tgid=%" PRIu32 " tid=%" PRIu32 " comm=%.16s fd=%" PRId32
           " cmd=0x%" PRIx32 " result=%" PRId64 " elapsed_ns=%" PRIu64 "\n",
           event.tgid, event.tid, event.comm, event.fd, event.command,
           (int64_t)event.result, (uint64_t)event.elapsed_ns);
    return 0;
}
static int report_losses(struct ioctl_latency_bpf *skel)
{
    int cpus = libbpf_num_possible_cpus();
    if (cpus <= 0) return -1;
    __u64 *counts = calloc((size_t)cpus, sizeof(*counts));
    if (!counts) return -1;
    int failed = 0;
    for (__u32 kind = 0; kind < 2; ++kind) {
        if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.lost), &kind, counts)) {
            perror("read loss counter"); failed = 1; break;
        }
        uint64_t total = 0;
        for (int cpu = 0; cpu < cpus; ++cpu) total += counts[cpu];
        fprintf(stderr, "%s=%" PRIu64 "\n",
                kind == 0 ? "start_insert_failures" : "ring_reserve_failures", total);
    }
    free(counts);
    return failed ? -1 : 0;
}
int main(int argc, char **argv)
{
    /* Require an explicit host TGID: never silently trace every process. */
    if (argc != 2 || argv[1][0] < '1' || argv[1][0] > '9') {
        fprintf(stderr, "usage: %s HOST_TGID\n", argv[0]); return 2;
    }
    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(argv[1], &end, 10);
    if (errno || !end || *end || value > UINT_MAX) {
        fprintf(stderr, "invalid host TGID\n"); return 2;
    }
    struct sigaction action = { .sa_handler = interrupted };
    if (sigemptyset(&action.sa_mask) || sigaction(SIGINT, &action, NULL) ||
        sigaction(SIGTERM, &action, NULL)) { perror("sigaction"); return 1; }
    struct ioctl_latency_bpf *skel = ioctl_latency_bpf__open();
    if (!skel) { fprintf(stderr, "open BPF object failed\n"); return 1; }
    struct ring_buffer *ring = NULL;
    int status = 1;
    skel->rodata->target_tgid = (__u32)value;
    if (ioctl_latency_bpf__load(skel)) goto cleanup;
    ring = ring_buffer__new(bpf_map__fd(skel->maps.events), consume, NULL, NULL);
    if (!ring) { perror("create ring reader"); goto cleanup; }
    if (ioctl_latency_bpf__attach(skel)) goto cleanup;
    status = 0;
    while (!stop) {
        int rc = ring_buffer__poll(ring, 100);
        if (rc == -EINTR) continue;
        if (rc < 0) { fprintf(stderr, "poll: %s\n", strerror(-rc)); status = 1; break; }
    }
    /* Stop new events before draining the userspace-visible ring and counters. */
    ioctl_latency_bpf__detach(skel);
    if (ring_buffer__consume(ring) < 0) status = 1;
    if (report_losses(skel)) status = 1;
cleanup:
    ring_buffer__free(ring);
    ioctl_latency_bpf__destroy(skel);
    return status;
}
