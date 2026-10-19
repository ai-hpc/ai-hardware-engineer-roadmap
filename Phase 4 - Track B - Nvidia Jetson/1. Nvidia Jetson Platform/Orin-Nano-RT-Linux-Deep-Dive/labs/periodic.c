#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Linux userspace experiment. No scheduler, affinity, or system settings change. */
struct sample { int64_t release, wake, finish; };

static int64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) { perror("clock_gettime"); exit(1); }
    return (int64_t)ts.tv_sec * INT64_C(1000000000) + ts.tv_nsec;
}

static struct timespec as_time(int64_t ns) {
    return (struct timespec){.tv_sec = (time_t)(ns / INT64_C(1000000000)),
                            .tv_nsec = (long)(ns % INT64_C(1000000000))};
}

static int64_t parse(const char *s, int64_t low, int64_t high) {
    char *end;
    errno = 0;
    long long value = strtoll(s, &end, 10);
    if (errno || !*s || *end || value < low || value > high) {
        fprintf(stderr, "invalid integer: %s (allowed %" PRId64 "..%" PRId64 ")\n", s, low, high);
        exit(2);
    }
    return (int64_t)value;
}

static void sleep_ns(int64_t ns, int absolute) {
    struct timespec request = as_time(ns), remaining;
    int error;
    do {
        error = clock_nanosleep(CLOCK_MONOTONIC, absolute ? TIMER_ABSTIME : 0,
                               &request, absolute ? NULL : &remaining);
        if (error == EINTR && !absolute) request = remaining;
    } while (error == EINTR);
    if (error) { fprintf(stderr, "clock_nanosleep: %s\n", strerror(error)); exit(1); }
}

int main(int argc, char **argv) {
    if (argc != 5 || (strcmp(argv[1], "absolute") && strcmp(argv[1], "relative"))) {
        fprintf(stderr, "usage: periodic absolute|relative iterations period_us work_us\n");
        return 2;
    }
    int absolute = !strcmp(argv[1], "absolute");
    int64_t count = parse(argv[2], 1, 2000);
    int64_t period = parse(argv[3], 100, 100000) * 1000;
    int64_t work = parse(argv[4], 0, 100000) * 1000;
    if (count * (period + work) > INT64_C(10000000000)) {
        fprintf(stderr, "requested nominal experiment exceeds 10 seconds\n"); return 2;
    }
    struct sample *samples = calloc((size_t)count, sizeof(*samples));
    if (!samples) { perror("calloc"); return 1; }
    /* Touch storage before starting; this does not lock it in RAM. */
    for (int64_t i = 0; i < count; ++i) samples[i].release = -1;
    int policy = sched_getscheduler(0);
    if (policy < 0) { perror("sched_getscheduler"); free(samples); return 1; }
    int64_t start = now_ns();
    for (int64_t i = 0; i < count; ++i) {
        samples[i].release = start + (i + 1) * period;
        sleep_ns(absolute ? samples[i].release : period, absolute);
        samples[i].wake = now_ns();
        /* Deliberately small synthetic wall-time work, not a CPU cycle model. */
        int64_t end = samples[i].wake + work;
        while (now_ns() < end) { }
        samples[i].finish = now_ns();
    }
    /* Printing is outside the measured loop. */
    printf("{\"mode\":\"%s\",\"clock\":\"CLOCK_MONOTONIC\",\"scheduler_policy\":%d,"
           "\"start_ns\":%" PRId64 ",\"period_ns\":%" PRId64 ",\"work_ns\":%" PRId64 ",\"samples\":[",
           argv[1], policy, start, period, work);
    for (int64_t i = 0; i < count; ++i) {
        printf("%s{\"release_ns\":%" PRId64 ",\"wake_ns\":%" PRId64 ",\"finish_ns\":%" PRId64 "}",
               i ? "," : "", samples[i].release, samples[i].wake, samples[i].finish);
    }
    puts("]}");
    free(samples);
    return 0;
}
