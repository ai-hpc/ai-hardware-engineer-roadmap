#ifndef COURSE_IOCTL_LATENCY_H
#define COURSE_IOCTL_LATENCY_H
#ifndef __VMLINUX_H__
#include <linux/types.h>
#endif
struct ioctl_start { __u64 started_ns; __u32 command; __s32 fd; };
struct ioctl_event {
    __u64 elapsed_ns;
    __s64 result;
    __u32 tgid;
    __u32 tid;
    __u32 command;
    __s32 fd;
    char comm[16];
};
#endif
