/* SPDX-License-Identifier: MIT
 * Small packaging exercise: one clock domain, integer milliseconds, no actuation.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int parse_ms(const char *text, uint64_t *value)
{
    char *end;
    unsigned long long parsed;
    if (*text == '\0') return 0;
    for (const char *p = text; *p; ++p)
        if (*p < '0' || *p > '9') return 0;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || parsed > UINT64_MAX) return 0;
    *value = (uint64_t)parsed;
    return 1;
}

int main(int argc, char **argv)
{
    uint64_t captured, completed, limit, age;
    if (argc != 4 || !parse_ms(argv[1], &captured) ||
        !parse_ms(argv[2], &completed) || !parse_ms(argv[3], &limit)) {
        fputs("usage: observation-check CAPTURE_MS COMPLETE_MS LIMIT_MS\n", stderr);
        return 64;
    }
    if (completed < captured) {
        puts("{\"status\":\"clock_error\"}");
        return 65;
    }
    age = completed - captured; /* Safe after checking the ordering. */
    printf("{\"status\":\"%s\",\"age_ms\":%" PRIu64 "}\n",
           age <= limit ? "fresh" : "stale", age);
    return age <= limit ? 0 : 2;
}
