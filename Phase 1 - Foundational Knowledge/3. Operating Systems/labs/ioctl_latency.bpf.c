#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include "ioctl_latency.h"

/* Configure BEFORE load: 0 traces all thread groups. A nonzero TGID is safer
 * for a short, controlled experiment. This uses the host PID namespace. */
const volatile __u32 target_tgid = 0;
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key, __u64);
    __type(value, struct ioctl_start);
} starts SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 20);
} events SEC(".maps");
/* Per-CPU counters: 0 = failed start insert; 1 = failed event reservation. */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 2);
    __type(key, __u32);
    __type(value, __u64);
} lost SEC(".maps");
static __always_inline void count_loss(__u32 kind)
{
    __u64 *n = bpf_map_lookup_elem(&lost, &kind);
    if (n) (*n)++;
}
SEC("tracepoint/syscalls/sys_enter_ioctl")
int enter_ioctl(struct trace_event_raw_sys_enter *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    if (target_tgid && (__u32)(id >> 32) != target_tgid) return 0;
    struct ioctl_start value = {
        .started_ns = bpf_ktime_get_ns(),
        .command = (__u32)ctx->args[1],
        .fd = (__s32)ctx->args[0],
    };
    if (bpf_map_update_elem(&starts, &id, &value, BPF_ANY)) count_loss(0);
    return 0;
}
SEC("tracepoint/syscalls/sys_exit_ioctl")
int exit_ioctl(struct trace_event_raw_sys_exit *ctx)
{
    __u64 id = bpf_get_current_pid_tgid();
    struct ioctl_start *saved = bpf_map_lookup_elem(&starts, &id);
    if (!saved) return 0;
    /* Copy before delete; every matched exit releases its hash entry. */
    struct ioctl_start start = *saved;
    bpf_map_delete_elem(&starts, &id);
    struct ioctl_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) { count_loss(1); return 0; }
    __builtin_memset(event, 0, sizeof(*event));
    event->elapsed_ns = bpf_ktime_get_ns() - start.started_ns;
    event->result = ctx->ret;
    event->tgid = id >> 32;
    event->tid = (__u32)id;
    event->command = start.command;
    event->fd = start.fd;
    bpf_get_current_comm(event->comm, sizeof(event->comm));
    bpf_ringbuf_submit(event, 0);
    return 0;
}
SEC("tracepoint/sched/sched_process_exit")
int cleanup_exit(void *ctx)
{
    (void)ctx;
    __u64 id = bpf_get_current_pid_tgid();
    bpf_map_delete_elem(&starts, &id);
    return 0;
}
char LICENSE[] SEC("license") = "Dual BSD/GPL";
