#pragma once
#include <stdio.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include "postgres.h"

#define GEMBED_TELEMETRY_LOG "/dev/shm/gembed_telemetry_log"

static inline FILE* _gembed_tlog(void)
{
    static FILE *_f = NULL;
    if (_f) return _f;

    int fd = open(GEMBED_TELEMETRY_LOG, O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd < 0)
    {
        elog(LOG, "GEMBED TELEMETRY: open(%s) failed: %m", GEMBED_TELEMETRY_LOG);
        return NULL;
    }

    _f = fdopen(fd, "a");
    if (!_f)
    {
        elog(LOG, "GEMBED TELEMETRY: fdopen() failed: %m");
        close(fd);
        return NULL;
    }

    return _f;
}

#define TELEMETRY_LOG(label, n) do \
{ \
    FILE *f = _gembed_tlog(); \
    if (f) \
    { \
        struct timespec ts; \
        clock_gettime(CLOCK_REALTIME, &ts); \
        if (fprintf(f, "%lld\t%s\t%d\n", \
            (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL, \
            label, (int)n) < 0) \
        { \
            elog(LOG, "GEMBED TELEMETRY: fprintf failed: %m"); \
        } \
        fflush(f); \
    } \
} while (0)
