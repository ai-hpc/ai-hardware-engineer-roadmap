/* Linux API/synchronization demonstration, not a latency benchmark. */
#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static void check_thread(int code, const char *operation)
{
    if (code != 0) {
        fprintf(stderr, "%s: %s\n", operation, strerror(code));
        exit(EXIT_FAILURE);
    }
}

static int parse_cpu(const char **cursor, long *value)
{
    if (!isdigit((unsigned char)**cursor)) return -1;
    char *end;
    errno = 0;
    *value = strtol(*cursor, &end, 10);
    if (errno || *value < 0 || *value >= CPU_SETSIZE) return -1;
    *cursor = end;
    return 0;
}

static int set_cpu_affinity(const char *text)
{
    cpu_set_t requested, allowed, obtained;
    CPU_ZERO(&requested);
    const char *cursor = text;
    do {
        long first, last;
        if (parse_cpu(&cursor, &first) != 0) goto invalid;
        last = first;
        if (*cursor == '-') {
            ++cursor;
            if (parse_cpu(&cursor, &last) != 0 || last < first) goto invalid;
        }
        for (long cpu = first; cpu <= last; ++cpu) CPU_SET(cpu, &requested);
        if (!*cursor) break;
        if (*cursor++ != ',' || !*cursor) goto invalid;
    } while (*cursor);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        perror("sched_getaffinity"); return -1;
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &requested) && !CPU_ISSET(cpu, &allowed)) {
            fprintf(stderr, "Requested CPU %d is outside the current allowed mask\n", cpu);
            return -1;
        }
    }
    if (sched_setaffinity(0, sizeof(requested), &requested) != 0 ||
        sched_getaffinity(0, sizeof(obtained), &obtained) != 0) {
        perror("CPU affinity"); return -1;
    }
    if (!CPU_EQUAL(&requested, &obtained)) {
        fprintf(stderr, "Obtained affinity differs from request\n"); return -1;
    }
    printf("affinity=%s obtained_count=%d\n", text, CPU_COUNT(&obtained));
    return 0;
invalid:
    fprintf(stderr, "Invalid CPU list: %s\n", text);
    return -1;
}

/* Volatile accesses retain these writes; volatile is not thread synchronization. */
static volatile unsigned char data_pages[65536];
static void touch_pages(volatile unsigned char *buffer, size_t bytes)
{
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) { fprintf(stderr, "Unknown page size\n"); exit(EXIT_FAILURE); }
    for (size_t offset = 0; offset < bytes; offset += (size_t)page) buffer[offset] = 1;
    buffer[bytes - 1] = 1;
}

typedef struct {
    pthread_mutex_t mutex;
    pthread_rwlock_t config_lock;
    pthread_cond_t condition;
    int ready;
    int counter;
    int config;
} Shared;

static void *worker(void *argument)
{
    Shared *shared = argument;
    volatile unsigned char own_stack_pages[8192];
    touch_pages(own_stack_pages, sizeof(own_stack_pages));
    check_thread(pthread_mutex_lock(&shared->mutex), "lock");
    shared->ready = 1;
    check_thread(pthread_cond_signal(&shared->condition), "signal");
    check_thread(pthread_mutex_unlock(&shared->mutex), "unlock");
    for (int i = 0; i < 10000; ++i) {
        check_thread(pthread_rwlock_rdlock(&shared->config_lock), "read lock");
        int config = shared->config;
        check_thread(pthread_rwlock_unlock(&shared->config_lock), "read unlock");
        if (config != 100) { fprintf(stderr, "Unexpected config\n"); exit(EXIT_FAILURE); }
        check_thread(pthread_mutex_lock(&shared->mutex), "lock");
        ++shared->counter;
        check_thread(pthread_mutex_unlock(&shared->mutex), "unlock");
        struct timespec observed;
        if (clock_gettime(CLOCK_MONOTONIC, &observed) != 0) {
            perror("clock_gettime"); exit(EXIT_FAILURE);
        }
        /* No periodic schedule or wakeup-latency measurement is claimed. */
    }
    return NULL;
}

static void usage(const char *program)
{
    printf("Usage: %s [--cpu 2,3-4] [--rt] [--lock-memory] [--no-rt]\n", program);
    puts("Default: inherited normal environment; --rt requests FIFO priority 80.");
    puts("RT and memory locking depend on permissions/resource limits; failures exit nonzero.");
    puts("--no-rt clears prior --rt and --lock-memory options (options apply left to right).");
}

int main(int argc, char **argv)
{
    int use_rt = 0, lock_memory = 0;
    const char *cpus = NULL;
    static const struct option options[] = {
        {"rt", no_argument, NULL, 'r'}, {"lock-memory", no_argument, NULL, 'l'},
        {"cpu", required_argument, NULL, 'c'}, {"no-rt", no_argument, NULL, 'n'},
        {"help", no_argument, NULL, 'h'}, {NULL, 0, NULL, 0}
    };
    int option;
    while ((option = getopt_long(argc, argv, "rlc:nh", options, NULL)) != -1) {
        if (option == 'r') use_rt = 1;
        else if (option == 'l') lock_memory = 1;
        else if (option == 'c') cpus = optarg;
        else if (option == 'n') use_rt = lock_memory = 0;
        else if (option == 'h') { usage(argv[0]); return EXIT_SUCCESS; }
        else return EXIT_FAILURE;
    }
    if (optind != argc) { usage(argv[0]); return EXIT_FAILURE; }
    printf("pid=%ld tid=%ld requested_fifo=%d requested_mlock=%d\n",
           (long)getpid(), (long)gettid(), use_rt, lock_memory);
    if (cpus && set_cpu_affinity(cpus) != 0) return EXIT_FAILURE;
    if (use_rt) {
        struct sched_param requested = {.sched_priority = 80};
        if (sched_setscheduler(0, SCHED_FIFO, &requested) != 0) {
            perror("sched_setscheduler"); return EXIT_FAILURE;
        }
    }
    struct sched_param obtained;
    int policy = sched_getscheduler(0);
    if (policy < 0 || sched_getparam(0, &obtained) != 0) {
        perror("scheduler query"); return EXIT_FAILURE;
    }
    printf("obtained_policy=%d obtained_priority=%d\n", policy, obtained.sched_priority);
    if (lock_memory && mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall"); return EXIT_FAILURE;
    }
    printf("memory_locked=%d\n", lock_memory);
    touch_pages(data_pages, sizeof(data_pages));
    Shared shared = {0};
    check_thread(pthread_mutex_init(&shared.mutex, NULL), "mutex init");
    check_thread(pthread_rwlock_init(&shared.config_lock, NULL), "rwlock init");
    check_thread(pthread_cond_init(&shared.condition, NULL), "condition init");
    check_thread(pthread_rwlock_wrlock(&shared.config_lock), "write lock");
    shared.config = 100;
    check_thread(pthread_rwlock_unlock(&shared.config_lock), "write unlock");
    pthread_t thread;
    check_thread(pthread_create(&thread, NULL, worker, &shared), "pthread_create");
    check_thread(pthread_mutex_lock(&shared.mutex), "lock");
    while (!shared.ready)
        check_thread(pthread_cond_wait(&shared.condition, &shared.mutex), "wait");
    check_thread(pthread_mutex_unlock(&shared.mutex), "unlock");
    check_thread(pthread_join(thread, NULL), "join");
    printf("Final counter = %d\n", shared.counter);
    check_thread(pthread_cond_destroy(&shared.condition), "condition destroy");
    check_thread(pthread_rwlock_destroy(&shared.config_lock), "rwlock destroy");
    check_thread(pthread_mutex_destroy(&shared.mutex), "mutex destroy");
    if (lock_memory && munlockall() != 0) { perror("munlockall"); return EXIT_FAILURE; }
    return shared.counter == 10000 ? EXIT_SUCCESS : EXIT_FAILURE;
}
