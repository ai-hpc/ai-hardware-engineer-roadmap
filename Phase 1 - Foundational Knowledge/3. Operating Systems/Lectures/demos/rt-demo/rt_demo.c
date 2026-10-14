/*
 * rt_demo.c — Small user-space demo for OS Lectures 1–9.
 *
 * It shows, in one program:
 * - how a normal process asks the kernel for scheduling help
 * - how a thread can be pinned to specific CPUs
 * - how memory locking and pre-faulting reduce surprise page faults
 * - how threads coordinate with mutexes, rwlocks, and condition variables
 *
 * Build: make (from this directory)
 * Run:   ./rt_demo [--rt] [--lock-memory] [--cpu 2,3] [--no-rt]
 *        sudo ./rt_demo --rt --lock-memory --cpu 1   # full RT setup
 */
#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

/* ---------- L2: process/thread identity ---------- */
static void print_process_info(void)
{
	pid_t pid = getpid();
	pid_t tid = gettid();
	printf("[L2] Process: pid=%d tgid=%d tid=%d\n", pid, pid, tid);
}

/* ---------- L8: CPU affinity ---------- */
static int set_cpu_affinity(const char *cpu_list)
{
	cpu_set_t mask;
	CPU_ZERO(&mask);
	char *buf = strdup(cpu_list);
	if (!buf) return -1;
	char *tok = strtok(buf, ",");
	while (tok) {
		unsigned int a, b;
		if (sscanf(tok, "%u-%u", &a, &b) == 2) {
			for (; a <= b && a < 1024; a++)
				CPU_SET(a, &mask);
		} else if (sscanf(tok, "%u", &a) == 1 && a < 1024) {
			CPU_SET(a, &mask);
		}
		tok = strtok(NULL, ",");
	}
	free(buf);
	if (CPU_COUNT(&mask) == 0) {
		fprintf(stderr, "Invalid or empty CPU list: %s\n", cpu_list);
		return -1;
	}
	if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
		perror("sched_setaffinity");
		return -1;
	}
	printf("[L8] CPU affinity set to %s\n", cpu_list);
	return 0;
}

/* ---------- L6/L7: scheduler and memory lock ---------- */
static int set_rt_scheduler(int enable)
{
	if (!enable)
		return 0;
	struct sched_param param = { .sched_priority = 80 };
	if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
		perror("sched_setscheduler(SCHED_FIFO)");
		return -1;
	}
	printf("[L6/L7] Scheduler: SCHED_FIFO priority 80\n");
	return 0;
}

static int lock_all_memory(int enable)
{
	if (!enable)
		return 0;
	if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
		perror("mlockall");
		return -1;
	}
	printf("[L7] mlockall(MCL_CURRENT | MCL_FUTURE)\n");
	return 0;
}

/* Pre-fault: touch memory so no page fault in RT loop (L7) */
#define PREFAULT_SIZE 65536
static char prefault_buf[PREFAULT_SIZE];
static void prefault_pages(void)
{
	memset(prefault_buf, 0, sizeof(prefault_buf));
	volatile char stack_probe[8192];
	memset((void *)stack_probe, 0, sizeof(stack_probe));
	printf("[L7] Pre-faulted buffer and stack\n");
}

/* ---------- L9: shared state and synchronization ---------- */
typedef struct {
	pthread_mutex_t mutex;       /* protects counter and ready flag */
	pthread_rwlock_t rwlock;    /* protects config (read-heavy) */
	pthread_cond_t cond;        /* completion-style: worker ready */
	int counter;
	int worker_ready;
	int run_iterations;
	volatile int stop;
	/* Config (read by worker under read lock) */
	int config_value;
} shared_t;

static void *worker_thread(void *arg)
{
	shared_t *s = (shared_t *)arg;
	struct timespec ts;

/* Signal that the worker is ready, like a simple completion event (L9). */
	pthread_mutex_lock(&s->mutex);
	s->worker_ready = 1;
	pthread_cond_signal(&s->cond);
	pthread_mutex_unlock(&s->mutex);

	/* RT-style loop: avoid malloc and use pre-faulted data (L7). */
	while (!s->stop && s->run_iterations > 0) {
		/* L9: read lock for config (many readers OK) */
		pthread_rwlock_rdlock(&s->rwlock);
		int cfg = s->config_value;
		pthread_rwlock_unlock(&s->rwlock);

		/* Short critical section: update counter (L9 mutex) */
		pthread_mutex_lock(&s->mutex);
		s->counter++;
		pthread_mutex_unlock(&s->mutex);

		/* Simulate work; clock_gettime is a normal time source here (L4). */
		clock_gettime(CLOCK_MONOTONIC, &ts);
		(void)ts;
		(void)cfg;

		s->run_iterations--;
	}
	return NULL;
}

static void run_demo(int use_rt, int lock_mem, const char *cpus)
{
	shared_t sh = { 0 };
	pthread_t th;
	int ret;

	pthread_mutex_init(&sh.mutex, NULL);
	pthread_rwlock_init(&sh.rwlock, NULL);
	pthread_cond_init(&sh.cond, NULL);
	sh.config_value = 42;
	sh.run_iterations = 10000;

	print_process_info();

	if (cpus && set_cpu_affinity(cpus) != 0)
		return;
	if (set_rt_scheduler(use_rt) != 0)
		return;
	if (lock_all_memory(lock_mem) != 0)
		return;
	prefault_pages();

	/* Update shared config briefly under the write lock (L9). */
	pthread_rwlock_wrlock(&sh.rwlock);
	sh.config_value = 100;
	pthread_rwlock_unlock(&sh.rwlock);

	ret = pthread_create(&th, NULL, worker_thread, &sh);
	if (ret != 0) {
		fprintf(stderr, "pthread_create: %d\n", ret);
		return;
	}

	/* Wait until the worker tells us it has started (L9). */
	pthread_mutex_lock(&sh.mutex);
	while (!sh.worker_ready)
		pthread_cond_wait(&sh.cond, &sh.mutex);
	pthread_mutex_unlock(&sh.mutex);
	printf("[L9] Worker signaled ready\n");

	pthread_join(th, NULL);
	printf("[L9] Final counter = %d\n", sh.counter);
}

static void usage(const char *prog)
{
	printf("Usage: %s [OPTIONS]\n", prog);
	printf("  --rt           Use SCHED_FIFO priority 80 (needs root)\n");
	printf("  --lock-memory  mlockall (needs root)\n");
	printf("  --cpu <list>   CPU affinity, e.g. 1 or 2-4\n");
	printf("  --no-rt        Default: no RT, no mlock (run without root)\n");
	printf("\nExample: sudo %s --rt --lock-memory --cpu 1\n", prog);
}

int main(int argc, char **argv)
{
	int use_rt = 0, lock_mem = 0;
	const char *cpus = NULL;
	static struct option opts[] = {
		{ "rt",          no_argument,       0, 'r' },
		{ "lock-memory", no_argument,       0, 'l' },
		{ "cpu",         required_argument, 0, 'c' },
		{ "no-rt",       no_argument,       0, 'n' },
		{ "help",        no_argument,       0, 'h' },
		{ 0, 0, 0, 0 }
	};
	int c, idx;
	while ((c = getopt_long(argc, argv, "rlc:nh", opts, &idx)) != -1) {
		if (c == 'r') use_rt = 1;
		else if (c == 'l') lock_mem = 1;
		else if (c == 'c') cpus = optarg;
		else if (c == 'n') use_rt = 0, lock_mem = 0;
		else if (c == 'h') { usage(argv[0]); return 0; }
	}

	run_demo(use_rt, lock_mem, cpus);
	return 0;
}
